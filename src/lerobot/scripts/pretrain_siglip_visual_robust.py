#!/usr/bin/env python
"""Pre-train the SigLIP tower alone for embodiment invariance, before any policy training.

The visual-robust export renders the same frame with all three robots, so a positive pair differs
ONLY in which arm is in the picture. Pulling those together and pushing other frames apart trains
a tower that reads the scene rather than the robot. The result is written as a bare vision-tower
state dict, which SmolVLA loads with `--policy.vision_encoder_path`.

Note the direction: this makes the encoder embodiment-INVARIANT (it stops being able to tell the
robots apart). Training an encoder that IDENTIFIES the embodiment is the opposite objective and is
not what this script does.

    python src/lerobot/scripts/pretrain_siglip_visual_robust.py --steps 2000

Two failure modes this script is built to expose rather than hide:

* **The loss can satisfy itself where the policy never looks.** The gap metric and the mean-pooled
  contrastive objective are the same statistic, so training one and reporting the other is
  circular. `--pool tokens` contrasts per-patch features instead, and the evaluation always reports
  BOTH the pooled gap and a per-token gap, so a pooled-only win is visible as a gap between them.
  This is the failure that made the alignment objective inert (CLAUDE.md section 3).
* **This corpus is small** (~324 episodes, one task per embodiment). A tower trained on it alone
  can drift a long way from pretrained and lose general features, so relative weight drift is
  reported every evaluation next to the gap, and `--l2sp` can hold it back.
"""

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad  # noqa: E402
from lerobot.scripts.lerobot_train_with_visual_robust import (  # noqa: E402
    SameEpisodeBatchSampler,
    VisualRobustStateHead,
    _select_visual_robust_image_keys,
    _supervised_contrastive_loss,
    build_episode_position_offsets,
    compute_eef_state_normalizer,
    make_visual_robust_contrastive_loader,
    visual_robust_state_loss_from_tokens,
)

VLM_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def log(msg: str) -> None:
    print(f"[pretrain_siglip] {msg}", flush=True)


def prepare_views(batch, image_keys, device, resize=(512, 512)):
    """[B, V, 3, H, W] prepared exactly as the policy prepares them, so the features transfer."""
    images = []
    for key in image_keys:
        img = batch[key].to(device, non_blocking=True)
        img = img[:, -1] if img.ndim == 5 else img
        img = resize_with_pad(img, *resize, pad_value=0)
        images.append(img * 2.0 - 1.0)
    return torch.stack(images, dim=1)


def reference_eval_batches(root: Path, batch_size: int, n_batches: int, seed: int,
                           image_prefix: str = "observation.image.", include_views=None):
    """Rebuild tools/compare_siglip_all_checkpoints.py's evaluation batches, step for step.

    Same seed, same dataset order (sorted glob), same sampler, same image prep. Without this the
    gap reported here would not be comparable to the published table: centering is per-batch over
    those 24 images, so a different draw of frames moves the number a long way -- the pretrained
    tower reads -0.0501 on the table's batches and +0.2234 on an arbitrary other draw.

    These frames are part of the training corpus, so this is a probe of what the encoder has
    learned to represent, not a held-out generalization measure.
    """
    torch.manual_seed(seed)
    repo_ids = sorted(f"{p.parent.parent.parent.name}/lerobot" for p in root.glob("*/lerobot/meta/info.json"))
    ds = MultiLeRobotDataset(
        repo_ids, root=root, delta_timestamps={r: None for r in repo_ids},
        visual_cue_mode="vanilla", use_wrist_cam=False, use_state=True, cache_in_memory=False,
    )
    sampler = SameEpisodeBatchSampler(ds.meta_episodes, batch_size=batch_size, shuffle=True)
    batches, it = [], iter(sampler)
    for _ in range(n_batches):
        items = [ds[i] for i in next(it)]
        batch = {k: torch.stack([x[k] for x in items]) for k in items[0] if torch.is_tensor(items[0][k])}
        keys = _select_visual_robust_image_keys(
            batch, image_prefix=image_prefix, include_views=include_views
        )
        views = prepare_views(batch, keys, "cpu")
        batches.append(views)
    return batches


def encode(vision_model, flat_images, chunk, grad=True, dtype=torch.float32):
    outputs = []
    for piece in flat_images.split(chunk, dim=0):
        with nullcontext() if grad else torch.no_grad():
            outputs.append(
                vision_model(pixel_values=piece.to(dtype=dtype), patch_attention_mask=None).last_hidden_state
            )
    return torch.cat(outputs, dim=0)


def gap_stats(features, n_frames, n_views, center):
    """Positive/negative cosine gap, identical in definition to tools/compare_siglip_all_checkpoints."""
    f = features.float()
    if center:
        f = f - f.mean(dim=0, keepdim=True)
    f = F.normalize(f, dim=-1)
    sim = f @ f.T
    frame_of = torch.arange(n_frames, device=f.device).repeat_interleave(n_views)
    same = frame_of[:, None] == frame_of[None, :]
    eye = torch.eye(len(f), dtype=torch.bool, device=f.device)
    pos = sim[same & ~eye].mean().item()
    neg = sim[~same].mean().item()
    return pos, neg, pos - neg


@torch.no_grad()
def evaluate(vision_model, eval_batches, device, chunk):
    """Pooled and per-token gaps, on frozen held-out batches so the numbers are comparable."""
    vision_model.eval()
    pooled, token = [], []
    for views in eval_batches:
        n_frames, n_views = views.shape[:2]
        tokens = encode(vision_model, views.flatten(0, 1).to(device), chunk, grad=False)
        pooled.append(gap_stats(tokens.mean(dim=1), n_frames, n_views, center=True))
        # Per-token: the policy reads a compressed version of the token grid, not this mean, so a
        # pooled win that does not show up here has not touched what the policy consumes.
        token.append(gap_stats(tokens.flatten(1), n_frames, n_views, center=True))
    vision_model.train()
    mean = lambda rows, i: sum(r[i] for r in rows) / len(rows)  # noqa: E731
    return {
        "pooled_pos": mean(pooled, 0), "pooled_neg": mean(pooled, 1), "pooled_gap": mean(pooled, 2),
        "token_pos": mean(token, 0), "token_neg": mean(token, 1), "token_gap": mean(token, 2),
    }


def relative_drift(model, reference):
    num = sum((p.detach().float() - reference[n]).pow(2).sum() for n, p in model.named_parameters())
    den = sum(t.pow(2).sum() for t in reference.values())
    return float((num / den) ** 0.5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO_ROOT / "dataset_git/visual_robust_new_barx/new_barx"))
    ap.add_argument(
        "--repo-ids",
        default="PandaOmron_TurnOnSinkFaucet/lerobot,IIWAOmron_PnPCounterToSink/lerobot,"
        "UR5eOmron_PnPSinkToCounter/lerobot",
    )
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8, help="frames per step; each yields 3 views")
    ap.add_argument("--max-views", type=int, default=3)
    ap.add_argument("--image-prefix", default="observation.image.",
                    help="'observation.image.' for the 3-embodiment export, 'observation.images.' for the "
                         "6-embodiment one, whose keys are observation.images.<Emb>.<camera>")
    ap.add_argument("--include-views", default=None,
                    help="Comma-separated substrings naming which views to use. REQUIRED for the "
                         "6-embodiment export: its keys carry both cameras, and matching the prefix "
                         "alone would pull the wrist view into the same positive group as the front "
                         "one -- which would train away the viewpoint distinction instead of the "
                         "robot's identity.")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--objective", choices=["contrastive", "eef", "both"], default="contrastive",
                    help="contrastive: pull the three renders of a frame together. "
                         "eef: regress the end-effector pose, which every render of a frame shares. "
                         "both: sum of the two, weighted by --eef-weight")
    ap.add_argument("--eef-weight", type=float, default=1.0)
    ap.add_argument("--state-pool", choices=["mean", "attn"], default="attn",
                    help="How the EEF head reads the token grid. 'attn' beats 'mean' decisively on "
                         "this target (CLAUDE.md section 3): a mean over 1024 tokens is close to "
                         "position-blind, while attention weights over position-tagged tokens ARE "
                         "the localisation.")
    ap.add_argument("--state-hidden", type=int, default=512)
    ap.add_argument("--state-layers", type=int, default=2)
    ap.add_argument("--rotation-weight", type=float, default=1.0)
    ap.add_argument("--gripper-weight", type=float, default=1.0)
    ap.add_argument("--pool", choices=["mean", "tokens"], default="mean",
                    help="'mean' contrasts the pooled feature (what the gap metric measures); "
                         "'tokens' contrasts each patch position, which is closer to what the policy reads")
    ap.add_argument("--l2sp", type=float, default=0.0, help="weight on ||w - w_pretrained||^2")
    ap.add_argument("--encoder-chunk", type=int, default=12)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=3)
    ap.add_argument("--eval-batch-size", type=int, default=8,
                    help="Held at 8 to match the published table; changing it changes the centering "
                         "and makes the gap incomparable to tools/compare_siglip_all_checkpoints.py")
    ap.add_argument("--same-episode-negatives", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--cache-in-memory", type=lambda s: s.lower() != "false", default=False)
    ap.add_argument("--video-backend", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "outputs/siglip_pretrain"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForImageTextToText

    log(f"loading {VLM_MODEL} vision tower (fp32) ...")
    # fp32 on purpose. The policy keeps its VLM in bf16, where any parameter whose accumulated
    # update stays under half a ULP never moves at all -- 38-45 tensors per checkpoint, almost all
    # LayerNorm scales (CLAUDE.md section 8). A dedicated pre-training run has no reason to inherit
    # that; the tower is cast to bf16 only when the policy loads it.
    vision_model = (
        AutoModelForImageTextToText.from_pretrained(VLM_MODEL, dtype=torch.float32).model.vision_model.to(device)
    )
    vision_model.train()
    reference = {n: p.detach().float().clone() for n, p in vision_model.named_parameters()}
    log(f"{sum(p.numel() for p in vision_model.parameters()) / 1e6:.1f}M parameters")

    loader = make_visual_robust_contrastive_loader(
        root=args.root,
        repo_ids=args.repo_ids,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_in_memory=args.cache_in_memory,
        video_backend=args.video_backend,
        same_episode_negatives=args.same_episode_negatives,
    )
    log(f"dataset ready: {args.repo_ids}")

    include_views = tuple(v.strip() for v in (args.include_views or "").split(",") if v.strip()) or None
    eval_batches = reference_eval_batches(
        Path(args.root), args.eval_batch_size, args.eval_batches, args.seed,
        image_prefix=args.image_prefix, include_views=include_views,
    )
    log(f"{len(eval_batches)} evaluation batches, matched to tools/compare_siglip_all_checkpoints.py "
        f"({eval_batches[0].shape[0]} frames x {eval_batches[0].shape[1]} views each)")

    state_head = None
    normalizer = None
    episode_offsets = None
    trainable = list(vision_model.parameters())
    if args.objective in ("eef", "both"):
        dataset = loader.dataset
        # Per-episode centering, without which the target is world-frame fingertip position -- which
        # a robot-mounted camera genuinely cannot see, and a head asked for it plateaus at the
        # between-episode std of 1.57 m (CLAUDE.md section 1).
        episode_offsets = build_episode_position_offsets(dataset, "observation.state", slice(0, 3))
        normalizer = compute_eef_state_normalizer(
            dataset, "observation.state", quat_slice=slice(3, 7), episode_offsets=episode_offsets
        )
        state_head = VisualRobustStateHead(
            in_dim=vision_model.config.hidden_size,
            hidden_dim=args.state_hidden,
            out_dim=8,
            num_layers=args.state_layers,
            pool=args.state_pool,
        ).to(device)
        trainable += list(state_head.parameters())
        log(f"EEF head: pool={args.state_pool}, {sum(p.numel() for p in state_head.parameters()) / 1e6:.2f}M params, "
            f"{int(normalizer[2].sum())}/8 dimensions active (constant ones are zero-weighted)")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(1, args.warmup))
    )

    start = evaluate(vision_model, eval_batches, device, args.encoder_chunk)
    log(f"step     0  pooled gap {start['pooled_gap']:+.4f}  token gap {start['token_gap']:+.4f}  "
        f"drift 0.00000   (pretrained starting point)")
    history = [{"step": 0, "drift": 0.0, **start}]

    step = 0
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            keys = _select_visual_robust_image_keys(
            batch, args.image_prefix, max_views=args.max_views, include_views=include_views
        )
            if len(keys) < 2:
                continue
            views = prepare_views(batch, keys, device)
            n_frames, n_views = views.shape[:2]

            tokens = encode(vision_model, views.flatten(0, 1), args.encoder_chunk)
            labels = torch.arange(n_frames, device=device).repeat_interleave(n_views)

            loss = torch.zeros((), device=device)
            extra = {}
            if args.objective in ("contrastive", "both"):
                if args.pool == "mean":
                    contrastive = _supervised_contrastive_loss(tokens.mean(dim=1), labels, args.temperature)
                else:
                    # One contrastive problem per patch position, averaged. Same positives, but the
                    # objective now has to hold at every location the connector will later read from.
                    n_tokens = tokens.shape[1]
                    per_token = [
                        _supervised_contrastive_loss(tokens[:, i], labels, args.temperature)
                        for i in range(0, n_tokens, max(1, n_tokens // 16))
                    ]
                    contrastive = sum(per_token) / len(per_token)
                loss = loss + contrastive
                extra["contrastive"] = float(contrastive.detach())
            if args.objective in ("eef", "both"):
                # Same loss the policy-side auxiliary term uses, sharing one implementation.
                state_loss, state_metrics = visual_robust_state_loss_from_tokens(
                    tokens=tokens,
                    batch=batch,
                    head=state_head,
                    normalizer=normalizer,
                    device=device,
                    num_views=n_views,
                    episode_offsets=episode_offsets,
                    rotation_weight=args.rotation_weight,
                    gripper_weight=args.gripper_weight,
                )
                loss = loss + args.eef_weight * state_loss
                extra["eef"] = float(state_loss.detach())
                extra["pos_err_m"] = state_metrics["visual_robust_state_pos_err_m"]
                extra["rot_err_deg"] = state_metrics["visual_robust_state_rot_err_deg"]
                extra["view_spread_m"] = state_metrics["visual_robust_state_view_spread_m"]

            contrastive_value = float(loss.detach())
            if args.l2sp:
                penalty = sum(
                    (p.float() - reference[n]).pow(2).sum() for n, p in vision_model.named_parameters()
                )
                loss = loss + args.l2sp * penalty

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 10.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % args.eval_every == 0 or step == args.steps:
                stats = evaluate(vision_model, eval_batches, device, args.encoder_chunk)
                drift = relative_drift(vision_model, reference)
                history.append({"step": step, "drift": drift, "loss": contrastive_value, **extra, **stats})
                detail = "".join(
                    f"  {k} {v:6.3f}" for k, v in extra.items() if k in ("contrastive", "eef", "pos_err_m")
                )
                log(
                    f"step {step:5d}  loss {contrastive_value:6.3f}{detail}  "
                    f"pooled gap {stats['pooled_gap']:+.4f}  token gap {stats['token_gap']:+.4f}  "
                    f"drift {drift:.5f}  |g| {float(grad_norm):.2f}"
                )

    if state_head is not None:
        torch.save(state_head.state_dict(), out_dir / "eef_head.pt")

    tower_path = out_dir / "vision_tower.safetensors"
    from safetensors.torch import save_file

    save_file({n: p.detach().cpu() for n, p in vision_model.state_dict().items()}, tower_path)
    (out_dir / "pretrain_info.json").write_text(
        json.dumps({"args": vars(args), "history": history}, indent=2, default=str)
    )
    log(f"wrote {tower_path}")
    log(f"wrote {out_dir / 'pretrain_info.json'}")
    log("")
    log("use it with:  --policy.vision_encoder_path=" + str(tower_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
