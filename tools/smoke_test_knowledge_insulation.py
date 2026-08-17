#!/usr/bin/env python
"""Smoke test for knowledge insulation (`--policy.knowledge_insulation=true`).

Run before any long job:

    source /home/gpuuser/miniforge3/etc/profile.d/conda.sh && conda activate smolvla
    python tools/smoke_test_knowledge_insulation.py

Ten checks, in the order they can break:

  1. FAST tokenization round-trips through the LM vocabulary mapping.
  2. Insulation does not change the action expert: the split attention matches the stock joint
     attention to bf16 precision.
  3. The postfix is invisible to the action expert: the flow-matching loss has exactly zero
     gradient w.r.t. the ground-truth action tokens (no label leakage).
  4. The flow-matching loss alone puts NO gradient on any VLM parameter, and still trains the
     expert.
  5. The FAST cross-entropy alone DOES put gradient on the VLM (including `lm_head`, which is
     frozen without this flag).
  6. The cross-entropy predicts the NEXT token — an off-by-one would let each position see its
     own target, and the loss would fall convincingly while teaching nothing.
  7. The attention mask and position ids match the design, read off the real forward call.
  8. The two losses own disjoint parameter sets, with nothing left untrained.
  9. The cross-entropy actually falls when optimized on a fixed batch.
 10. Sampling still works with the flag on, so the checkpoint can be evaluated.

Synthetic data throughout: this tests the mechanism, not the corpus.
"""

import argparse
import contextlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching  # noqa: E402

CHUNK = 10
ACTION_DIM = 12
BATCH = 2
PASS, FAIL = "PASS", "FAIL"
results = []


def report(name, ok, detail=""):
    results.append(ok)
    print(f"[{PASS if ok else FAIL}] {name}{'  ' + detail if detail else ''}")


def build_model(knowledge_insulation, device, num_layers, seed=0):
    torch.manual_seed(seed)
    config = SmolVLAConfig(
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        num_vlm_layers=num_layers,
        self_attn_every_n_layers=2,
        attention_mode="cross_attn",
        freeze_vision_encoder=False,
        train_expert_only=False,
        load_vlm_weights=True,
        knowledge_insulation=knowledge_insulation,
        tokenizer_max_length=16,
    )
    config.device = device
    model = VLAFlowMatching(config).to(device)
    model.train()
    return model


def make_batch(model, device, seed=0):
    torch.manual_seed(seed)
    config = model.config
    images = [torch.rand(BATCH, 3, 512, 512, device=device) * 2 - 1]
    img_masks = [torch.ones(BATCH, dtype=torch.bool, device=device)]
    lang_tokens = torch.randint(10, 1000, (BATCH, config.tokenizer_max_length), device=device)
    lang_masks = torch.ones_like(lang_tokens, dtype=torch.bool)
    state = torch.randn(BATCH, config.max_state_dim, device=device)
    # Smooth, like a real trajectory — FAST is a compression codec, so white noise would tokenize
    # into an unlearnably long and unpredictable id stream.
    raw_actions = torch.tanh(torch.randn(BATCH, CHUNK, ACTION_DIM, device=device).cumsum(dim=1) * 0.3)
    actions = torch.zeros(BATCH, CHUNK, config.max_action_dim, device=device)
    actions[..., :ACTION_DIM] = raw_actions
    noise = torch.randn_like(actions)
    time = torch.full((BATCH,), 0.5, device=device)
    return {
        "images": images,
        "img_masks": img_masks,
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
        "state": state,
        "raw_actions": raw_actions,
        "actions": actions,
        "noise": noise,
        "time": time,
    }


def run_forward(model, b, fast_postfix=None):
    return model.forward(
        b["images"],
        b["img_masks"],
        b["lang_tokens"],
        b["lang_masks"],
        b["state"],
        b["actions"],
        b["noise"],
        b["time"],
        fast_postfix,
    )


@contextlib.contextmanager
def patched_postfix_embedding(model, postfix_tokens):
    """Add a zero leaf tensor to the postfix embeddings so autograd can be asked about them."""
    original = model.vlm_with_expert.embed_language_tokens
    hidden = model.vlm_with_expert.config.text_config.hidden_size
    delta = torch.zeros(
        postfix_tokens.shape[0],
        postfix_tokens.shape[1],
        hidden,
        device=postfix_tokens.device,
        dtype=torch.float32,
        requires_grad=True,
    )

    def patched(tokens):
        emb = original(tokens)
        if tokens.shape == postfix_tokens.shape and torch.equal(tokens, postfix_tokens):
            emb = emb + delta.to(dtype=emb.dtype)
        return emb

    model.vlm_with_expert.embed_language_tokens = patched
    try:
        yield delta
    finally:
        model.vlm_with_expert.embed_language_tokens = original


@contextlib.contextmanager
def capture_vlm_forward(model):
    """Record the attention mask and position ids the model actually hands the transformer."""
    original = model.vlm_with_expert.forward
    captured = {}

    def patched(**kwargs):
        captured["attention_mask"] = kwargs["attention_mask"]
        captured["position_ids"] = kwargs["position_ids"]
        return original(**kwargs)

    model.vlm_with_expert.forward = patched
    try:
        yield captured
    finally:
        model.vlm_with_expert.forward = original


def partition_by_gradient(model, batch, postfix):
    """Split trainable parameters by which of the two losses actually reaches them."""

    def touched(loss_fn):
        model.zero_grad(set_to_none=True)
        flow, aux = run_forward(model, batch, postfix)
        loss_fn(flow, aux).backward()
        return {
            n
            for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
        }

    flow_touched = touched(lambda flow, aux: flow.mean())
    ce_touched = touched(lambda flow, aux: aux["fast_ce_loss"])
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    return (
        flow_touched - ce_touched,
        ce_touched - flow_touched,
        flow_touched & ce_touched,
        trainable - flow_touched - ce_touched,
    )


def vlm_parameters(model):
    return [(n, p) for n, p in model.vlm_with_expert.vlm.named_parameters() if p.requires_grad]


def expert_parameters(model):
    return [(n, p) for n, p in model.vlm_with_expert.lm_expert.named_parameters() if p.requires_grad]


def grad_norm(params):
    total = 0.0
    for _, p in params:
        if p.grad is not None:
            total += p.grad.detach().float().pow(2).sum().item()
    return total**0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-layers", type=int, default=4, help="Truncated VLM depth, for speed.")
    parser.add_argument("--ce-steps", type=int, default=150)
    parser.add_argument("--ce-lr", type=float, default=3e-4)
    args = parser.parse_args()
    device = args.device

    print(f"device={device} num_vlm_layers={args.num_layers}\n")

    ki = build_model(True, device, args.num_layers)
    batch = make_batch(ki, device)
    postfix = ki.fast_tokenizer.encode(batch["raw_actions"])

    # ---- 1. tokenizer round-trip -------------------------------------------------------------
    decoded = ki.fast_tokenizer.decode(postfix["tokens"], postfix["loss_masks"], CHUNK, ACTION_DIM)
    decoded = torch.as_tensor(decoded, device=device, dtype=torch.float32)
    err = (decoded - batch["raw_actions"]).abs().max().item()
    report(
        "FAST tokens round-trip through the LM vocab",
        err < 0.1,
        f"max |a - decode(encode(a))| = {err:.4f}, "
        f"postfix len {postfix['tokens'].shape[1]} (mean {postfix['stats']['fast_tokens_mean']:.1f} FAST ids)",
    )

    # ---- 2. insulation does not move the action expert ---------------------------------------
    with torch.no_grad():
        losses_ki, _ = run_forward(ki, batch)
    plain = build_model(False, device, args.num_layers)
    plain.load_state_dict(ki.state_dict())
    with torch.no_grad():
        losses_plain, _ = run_forward(plain, batch)
    delta = (losses_ki - losses_plain).abs().max().item()
    scale = losses_plain.abs().max().item()
    # Loose: the split changes the reduction order of bf16 matmuls, so the two paths agree only to
    # bf16 precision. Check 3 is the exact statement.
    report(
        "split attention reproduces the stock action path",
        delta <= 5e-3 * max(scale, 1.0),
        f"max |dloss| = {delta:.2e} (loss scale {scale:.2f}, bf16)",
    )
    del plain

    # ---- 3. no label leakage from the postfix ------------------------------------------------
    # Exact rather than numerical: perturb the postfix embeddings by a leaf tensor and ask
    # autograd whether the flow-matching loss depends on it at all. Zero gradient is proof that
    # no path exists from the ground-truth action tokens to the action expert.
    with patched_postfix_embedding(ki, postfix["tokens"]) as delta_leaf:
        flow_losses, aux = run_forward(ki, batch, postfix)
        (leak_grad,) = torch.autograd.grad(
            flow_losses.mean(), delta_leaf, retain_graph=True, allow_unused=True
        )
        (ce_grad,) = torch.autograd.grad(aux["fast_ce_loss"], delta_leaf, allow_unused=True)
    leak = 0.0 if leak_grad is None else leak_grad.abs().max().item()
    ce_dep = 0.0 if ce_grad is None else ce_grad.abs().max().item()
    report(
        "action expert cannot read the ground-truth action tokens",
        leak == 0.0 and ce_dep > 0.0,
        f"d(flow_loss)/d(postfix) = {leak:.3e} (must be 0), "
        f"d(ce)/d(postfix) = {ce_dep:.3e} (must not be, or the probe missed)",
    )

    # ---- 4. flow-matching gradient stops before the VLM --------------------------------------
    ki.zero_grad(set_to_none=True)
    losses, aux = run_forward(ki, batch, postfix)
    losses.mean().backward()
    vlm_g = grad_norm(vlm_parameters(ki))
    expert_g = grad_norm(expert_parameters(ki))
    with_grad = [n for n, p in vlm_parameters(ki) if p.grad is not None and p.grad.abs().sum() > 0]
    report(
        "flow-matching loss puts no gradient on the VLM",
        vlm_g == 0.0 and expert_g > 0.0,
        f"|g_vlm| = {vlm_g:.3e} over {len(vlm_parameters(ki))} params, |g_expert| = {expert_g:.3e}"
        + (f"  LEAKING: {with_grad[:5]}" if with_grad else ""),
    )

    # control: without the flag the same loss does reach the VLM
    control = build_model(False, device, args.num_layers)
    control.load_state_dict(ki.state_dict())
    control.zero_grad(set_to_none=True)
    losses_c, _ = run_forward(control, batch)
    losses_c.mean().backward()
    control_g = grad_norm(vlm_parameters(control))
    report(
        "control: without the flag the flow loss DOES reach the VLM",
        control_g > 0.0,
        f"|g_vlm| = {control_g:.3e}",
    )
    del control

    # ---- 5. the token loss is what trains the VLM --------------------------------------------
    ki.zero_grad(set_to_none=True)
    _, aux = run_forward(ki, batch, postfix)
    aux["fast_ce_loss"].backward()
    vlm_g = grad_norm(vlm_parameters(ki))
    lm_head = ki.vlm_with_expert.vlm.lm_head.weight
    vision = ki.vlm_with_expert.get_vlm_model().vision_model
    vision_g = grad_norm([(n, p) for n, p in vision.named_parameters() if p.requires_grad])
    report(
        "FAST cross-entropy trains the VLM (lm_head included)",
        vlm_g > 0.0 and lm_head.requires_grad and lm_head.grad is not None and lm_head.grad.abs().sum() > 0,
        f"|g_vlm| = {vlm_g:.3e}, |g_vision| = {vision_g:.3e}, lm_head.requires_grad={lm_head.requires_grad}",
    )

    # ---- 6. the cross-entropy is genuinely next-token ----------------------------------------
    # An off-by-one in the shift would let each position see the token it is asked to predict.
    # The CE would then collapse to ~0 within a few steps and look like spectacular learning while
    # teaching the VLM nothing. Exact test: with a correct shift, the embedding of postfix token j
    # can only influence the predictions of tokens after j — so the LAST postfix token, which
    # nothing follows, must have exactly zero gradient, while the first must have some.
    with patched_postfix_embedding(ki, postfix["tokens"]) as delta_leaf:
        _, aux = run_forward(ki, batch, postfix)
        (ce_grad,) = torch.autograd.grad(aux["fast_ce_loss"], delta_leaf)
    last_real = postfix["pad_masks"][0].nonzero()[-1].item()
    self_leak = ce_grad[0, last_real].abs().max().item()
    first_dep = ce_grad[0, 0].abs().max().item()
    report(
        "cross-entropy predicts the NEXT token, not the current one",
        self_leak == 0.0 and first_dep > 0.0,
        f"d(ce)/d(last postfix token) = {self_leak:.3e} (must be 0 — nothing is predicted from it), "
        f"d(ce)/d(first) = {first_dep:.3e}",
    )

    # ---- 7. the masks say what the design claims ---------------------------------------------
    # Read off the tensor the model actually passes to the transformer, not a re-derivation.
    with capture_vlm_forward(ki) as captured:
        run_forward(ki, batch, postfix)
        with_postfix = dict(captured)
    with capture_vlm_forward(ki) as captured:
        run_forward(ki, batch)
        without_postfix = dict(captured)

    mask = with_postfix["attention_mask"]
    postfix_len = postfix["tokens"].shape[1]
    total_len = mask.shape[1]
    suffix_len = CHUNK
    prefix_len = total_len - suffix_len
    core_len = prefix_len - postfix_len
    post = slice(core_len, prefix_len)

    expert_blind = not mask[:, prefix_len:, post].any()
    core_blind = not mask[:, :core_len, post].any()
    postfix_block = mask[:, post, post]
    causal = torch.tril(torch.ones_like(postfix_block[0]))
    postfix_causal = bool((postfix_block[0].float() <= causal).all())
    postfix_reads_core = bool(mask[:, post, :core_len].any())
    suffix_pos_same = torch.equal(
        with_postfix["position_ids"][:, prefix_len:], without_postfix["position_ids"][:, core_len:]
    )
    suffix_core_same = torch.equal(
        with_postfix["attention_mask"][:, prefix_len:, :core_len],
        without_postfix["attention_mask"][:, core_len:, :core_len],
    )
    report(
        "attention mask and position ids match the design",
        expert_blind and core_blind and postfix_causal and postfix_reads_core and suffix_pos_same
        and suffix_core_same,
        f"expert->postfix blind={expert_blind}, core->postfix blind={core_blind}, "
        f"postfix causal={postfix_causal} reads core={postfix_reads_core}, "
        f"suffix position ids unchanged={suffix_pos_same}, suffix->core mask unchanged={suffix_core_same}",
    )

    # ---- 8. the two objectives own disjoint parameters ----------------------------------------
    flow_only, ce_only, both, neither = partition_by_gradient(ki, batch, postfix)
    expected_ce = ("vlm_with_expert.vlm", "state_proj")
    expected_flow = ("lm_expert", "action_in_proj", "action_out_proj", "action_time_mlp")
    ce_ok = all(any(k in n for k in expected_ce) for n in ce_only)
    flow_ok = all(any(k in n for k in expected_flow) for n in flow_only)
    report(
        "each loss trains exactly the parameters it should",
        ce_ok and flow_ok and not both and not neither,
        f"CE-only {len(ce_only)}, flow-only {len(flow_only)}, both {len(both)}, neither {len(neither)}"
        + (f"  BOTH: {sorted(both)[:3]}" if both else "")
        + (f"  NEITHER (DDP unused): {sorted(neither)[:3]}" if neither else ""),
    )

    # ---- 9. the cross-entropy falls ----------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [p for p in ki.parameters() if p.requires_grad], lr=args.ce_lr, weight_decay=0.0
    )
    history = []
    for step in range(args.ce_steps):
        optimizer.zero_grad(set_to_none=True)
        flow, aux = run_forward(ki, batch, postfix)
        loss = flow.mean() + aux["fast_ce_loss"]
        loss.backward()
        optimizer.step()
        history.append((aux["fast_ce_loss"].item(), aux["fast_token_accuracy"]))
        if step % 5 == 0 or step == args.ce_steps - 1:
            print(f"      step {step:3d}  ce {history[-1][0]:7.4f}  token_acc {history[-1][1]:.3f}")
    first, last = history[0][0], history[-1][0]
    report(
        "FAST cross-entropy falls when optimized",
        last < first * 0.9,
        f"{first:.4f} -> {last:.4f} in {args.ce_steps} steps, token_acc {history[0][1]:.3f} -> {history[-1][1]:.3f}",
    )

    # ---- 10. inference is untouched -----------------------------------------------------------
    # The postfix exists only during training; sampling must still run, and run identically, or a
    # knowledge-insulated checkpoint cannot be evaluated.
    ki.eval()
    with torch.no_grad():
        sampled = ki.sample_actions(
            batch["images"],
            batch["img_masks"],
            batch["lang_tokens"],
            batch["lang_masks"],
            batch["state"],
            noise=batch["noise"],
        )
    report(
        "inference path still runs with the flag on",
        tuple(sampled.shape) == (BATCH, CHUNK, ki.config.max_action_dim) and torch.isfinite(sampled).all(),
        f"sample_actions -> {tuple(sampled.shape)}, finite={bool(torch.isfinite(sampled).all())}",
    )

    print()
    if all(results):
        print(f"all {len(results)} checks passed")
        return 0
    print(f"{results.count(False)}/{len(results)} checks FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
