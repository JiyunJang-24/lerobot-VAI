#!/usr/bin/env python
"""Experiment 1: can counterfactual supervision buy SELECTIVE invariance?

Four methods behind --method, all on the same backbone, the same split, the same batch shape, so
the only thing that differs is what the alignment gradient is applied to.

    A  frozen      stock SigLIP, no training. The reference.
    B  global      supervised InfoNCE on the mean-pooled feature. What was tried before.
    C  selective   same loss, but the feature is pooled over the ROBOT REGION only, using masks
                   derived from the cross-embodiment median. Non-robot patches get no alignment
                   gradient, so the scene is not asked to become embodiment-invariant.
    D  selective+  C plus an EEF head on the robot-pooled feature, so alignment cannot be bought
                   by discarding where the arm is.
    E  selective++ D plus an anchor holding NON-robot patch features near the frozen teacher.
                   C and D constrain nothing outside the mask; E is the explicit defence against
                   the scene drifting while nobody is watching.

Masks are used ONLY here. The trained encoder is an ordinary image -> features map, so a
downstream policy never needs one.

    python -m lerobot.scripts.exp1_train --method C --out outputs/exp1/C
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad  # noqa: E402
from lerobot.scripts.exp1_data import Exp1Config, Exp1Dataset  # noqa: E402

IMAGE_SIZE = 512
PATCH = 16
GRID = IMAGE_SIZE // PATCH


def log(msg: str) -> None:
    print(f"[exp1] {msg}", flush=True)


# --------------------------------------------------------------------------------------------
# backbone


def load_tower(path: str, device, dtype=torch.float32):
    from transformers import AutoModelForImageTextToText

    vlm = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", dtype=dtype)
    tower = vlm.model.vision_model
    if path:
        from safetensors.torch import load_file
        tower.load_state_dict(load_file(path), strict=True)
    return tower.to(device)


def prepare(images: np.ndarray, device) -> torch.Tensor:
    x = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float() / 255.0
    return resize_with_pad(x, IMAGE_SIZE, IMAGE_SIZE, pad_value=0) * 2.0 - 1.0


def prepare_masks(masks: np.ndarray, device) -> torch.Tensor:
    """Mask -> per-patch weight in [0,1], padded exactly like the image."""
    m = torch.from_numpy(masks).to(device).float().unsqueeze(1)
    m = resize_with_pad(m, IMAGE_SIZE, IMAGE_SIZE, pad_value=0)
    w = F.avg_pool2d(m, PATCH, PATCH)  # (B,1,GRID,GRID) = fraction of the patch that is robot
    return w.flatten(1)


def masked_pool(tokens: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w = weights.unsqueeze(-1)
    return (tokens * w).sum(1) / w.sum(1).clamp_min(1e-4)


# --------------------------------------------------------------------------------------------
# losses


def supervised_contrastive(features: torch.Tensor, labels: torch.Tensor, temperature: float):
    """Same form as the existing visual-robust loss, kept identical for comparability."""
    features = F.normalize(features.float(), dim=-1)
    logits = (features @ features.T) / temperature
    eye = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    logits = logits.masked_fill(eye, torch.finfo(logits.dtype).min)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    counts = positive.sum(1)
    valid = counts > 0
    if not valid.any():
        return features.new_zeros(())
    return -((log_prob * positive).sum(1)[valid] / counts[valid]).mean()


def quat_to_6d(q: torch.Tensor) -> torch.Tensor:
    """xyzw quaternion -> the first two rotation-matrix columns (Zhou et al. 6D).

    Euler angles are unusable as a regression target (wrap-around, gimbal); the repo stores
    quaternions, which are double-cover, so the sign flips randomly. 6D is continuous and has
    neither problem.
    """
    x, y, z, w = q.unbind(-1)
    n = torch.stack([x, y, z, w], -1).norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = (torch.stack([x, y, z, w], -1) / n).unbind(-1)
    c0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], -1)
    c1 = torch.stack([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)], -1)
    return torch.cat([c0, c1], -1)


class EefHead(torch.nn.Module):
    """Small MLP: pooled feature -> xyz (3) + 6D rotation (6) + gripper (1)."""

    def __init__(self, in_dim: int, hidden: int = 512):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 10))

    def forward(self, f):
        return self.net(f)


def eef_targets(eef: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    xyz = (eef[:, :3] - mean) / std
    rot = quat_to_6d(eef[:, 3:7])
    grip = eef[:, 7:8]
    return torch.cat([xyz, rot, grip], -1)


# --------------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=["A", "B", "C", "D", "E"])
    ap.add_argument("--split", type=Path, default=Path("outputs/exp1/split.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--states-per-batch", type=int, default=8)
    ap.add_argument("--embodiments-per-state", type=int, default=6)
    ap.add_argument("--lambda-eef", type=float, default=1.0)
    ap.add_argument("--lambda-anchor", type=float, default=1.0)
    ap.add_argument("--encoder-chunk", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    # scaling axes, all optional
    ap.add_argument("--max-train-scenes", type=int, default=0)
    ap.add_argument("--max-train-embodiments", type=int, default=0)
    ap.add_argument("--states-per-scene", type=int, default=0)
    ap.add_argument("--total-state-budget", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = Exp1Config(
        split=args.split, states_per_batch=args.states_per_batch,
        embodiments_per_state=args.embodiments_per_state,
        max_train_scenes=args.max_train_scenes,
        max_train_embodiments=args.max_train_embodiments,
        states_per_scene=args.states_per_scene,
        total_state_budget=args.total_state_budget, seed=args.seed)
    data = Exp1Dataset(cfg, "train")
    log(f"method {args.method}: {len(data.scenes)} scenes x {len(data.embodiments)} embodiments, "
        f"{data.n_states} usable states")

    tower = load_tower("", device)
    (args.out / "config.json").write_text(json.dumps(
        {"args": vars(args) | {"split": str(args.split), "out": str(args.out)},
         "data": asdict(cfg) | {"split": str(cfg.split)},
         "scenes": data.scenes, "embodiments": data.embodiments}, indent=2, default=str))

    if args.method == "A":
        # Nothing to train. Save the stock tower so evaluation treats every method identically.
        from safetensors.torch import save_file
        save_file({k: v.contiguous().cpu() for k, v in tower.state_dict().items()},
                  args.out / "vision_tower.safetensors")
        log("method A is the untrained reference; wrote the stock tower and stopped")
        return 0

    teacher = None
    if args.method == "E":
        teacher = load_tower("", device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    head = EefHead(768).to(device) if args.method in ("D", "E") else None
    params = list(tower.parameters()) + (list(head.parameters()) if head else [])
    opt = torch.optim.AdamW(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)))

    # EEF normaliser from the training scenes, so the regression target is unit-scale
    rng = np.random.default_rng(args.seed)
    sample = np.concatenate([data.sample_batch(rng)["eef"] for _ in range(16)])
    mean = torch.tensor(sample[:, :3].mean(0), device=device)
    std = torch.tensor(sample[:, :3].std(0) + 1e-6, device=device)
    log(f"eef xyz mean {mean.tolist()} std {std.tolist()}")

    history = []
    t0 = time.time()
    for step in range(args.steps):
        batch = data.sample_batch(rng)
        pixels = prepare(batch["images"], device)
        labels = torch.from_numpy(batch["labels"]).to(device)
        weights = prepare_masks(batch["masks"], device)

        tokens = []
        for piece in pixels.split(args.encoder_chunk):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens.append(tower(pixel_values=piece,
                                    patch_attention_mask=None).last_hidden_state)
        tokens = torch.cat(tokens).float()

        if args.method == "B":
            pooled = tokens.mean(1)
        else:
            pooled = masked_pool(tokens, weights)
        align = supervised_contrastive(pooled, labels, args.temperature)
        loss = align
        terms = {"align": float(align)}

        if head is not None:
            eef = torch.from_numpy(batch["eef"]).to(device)
            pred = head(pooled)
            target = eef_targets(eef, mean, std)
            eef_loss = F.l1_loss(pred[:, :3], target[:, :3]) + F.l1_loss(pred[:, 3:9], target[:, 3:9])
            loss = loss + args.lambda_eef * eef_loss
            terms["eef"] = float(eef_loss)

        if teacher is not None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ref = torch.cat([teacher(pixel_values=p, patch_attention_mask=None).last_hidden_state
                                 for p in pixels.split(args.encoder_chunk)]).float()
            scene_w = (1.0 - weights).unsqueeze(-1)
            # relative drift: absolute feature MSE is ~1e-4 at init and would need a lambda tuned
            # to the backbone's scale. Dividing by the teacher's own energy makes the term read as
            # "fraction of the scene representation that moved", so lambda 1.0 means "fully on".
            drift = (((tokens - ref) ** 2).mean(-1, keepdim=True) * scene_w).sum() / scene_w.sum().clamp_min(1e-4)
            anchor = drift / (ref ** 2).mean().clamp_min(1e-6)
            loss = loss + args.lambda_anchor * anchor
            terms["anchor"] = float(anchor)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        sched.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            robot_frac = float(weights.mean())
            msg = " ".join(f"{k} {v:.4f}" for k, v in terms.items())
            log(f"step {step:5d}/{args.steps} loss {float(loss):.4f} {msg} "
                f"|g| {float(gnorm):.2f} robot_frac {robot_frac:.3f} "
                f"({time.time() - t0:.0f}s)")
            history.append({"step": step, "loss": float(loss), **terms,
                            "grad_norm": float(gnorm), "robot_frac": robot_frac})

    from safetensors.torch import save_file
    save_file({k: v.contiguous().cpu() for k, v in tower.state_dict().items()},
              args.out / "vision_tower.safetensors")
    if head is not None:
        torch.save(head.state_dict(), args.out / "eef_head.pt")
    (args.out / "history.json").write_text(json.dumps(history, indent=1))
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
