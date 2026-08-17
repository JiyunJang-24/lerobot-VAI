#!/usr/bin/env python
"""Measure how much GPU memory a barx front-only training step needs, with the SigLIP vision tower
frozen vs trainable.

Purpose: decide whether a `--policy.freeze_vision_encoder=true` baseline can be squeezed onto the
same GPUs as a live 8-GPU run. Freezing removes the vision tower's gradients, its Adam moments and
(because no leaf under it requires grad) its whole backward activation graph, so the saving is much
larger than the parameter count alone suggests.

Runs one real forward+backward per (freeze, batch_size) and reports torch.cuda.max_memory_allocated.
The number is per-process, i.e. per GPU under DDP. Add ~1-2 GB of CUDA context/allocator slack when
comparing against nvidia-smi.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/probe_frozen_encoder_memory.py \
        --checkpoint outputs/train/.../checkpoints/050000/pretrained_model \
        --batch-sizes 16,32,48,64
"""

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402


def log(msg: str) -> None:
    print(f"[probe_frozen] {msg}", flush=True)


def set_freeze(policy: SmolVLAPolicy, freeze: bool) -> None:
    """Flip the vision tower between frozen and trainable exactly the way the policy config does."""
    vlm_with_expert = policy.model.vlm_with_expert
    vlm_with_expert.freeze_vision_encoder = freeze
    vision = vlm_with_expert.get_vlm_model().vision_model
    for params in vision.parameters():
        params.requires_grad = not freeze
    vlm_with_expert.set_requires_grad()


def trainable_params(policy) -> int:
    return sum(p.numel() for p in policy.parameters() if p.requires_grad)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "dataset_git/barx_frontonly_p900_i1000_u1000/raw")
    ap.add_argument("--batch-sizes", type=str, default="16,32,48,64")
    ap.add_argument("--freeze-modes", type=str, default="true,false")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        log("no CUDA device visible")
        return 1
    device = torch.device("cuda")
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    freeze_modes = [x.strip().lower() == "true" for x in args.freeze_modes.split(",")]
    max_bs = max(batch_sizes)

    repo_ids = sorted(p.parent.parent.name for p in args.root.glob("*/meta/info.json"))
    log(f"dataset repo_ids: {repo_ids}")
    log(f"loading policy from {args.checkpoint} ...")
    policy = SmolVLAPolicy.from_pretrained(str(args.checkpoint)).to(device)

    # The action chunk dominates the expert's sequence length, so it has to match training exactly.
    delta_timestamps = {}
    for repo_id in repo_ids:
        meta = LeRobotDatasetMetadata(repo_id, root=args.root / repo_id)
        delta_timestamps[repo_id] = resolve_delta_timestamps(policy.config, meta)
    ds = MultiLeRobotDataset(
        repo_ids,
        root=args.root,
        delta_timestamps=delta_timestamps,
        visual_cue_mode="vanilla",
        use_wrist_cam=False,
        use_state=True,
        cache_in_memory=False,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=max_bs, shuffle=True, num_workers=4)
    raw_batch = next(iter(loader))
    log(f"batch keys: {sorted(k for k in raw_batch if torch.is_tensor(raw_batch[k]))}")

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": device.type}},
    )
    batch = preprocessor(raw_batch)

    rows = []
    for freeze in freeze_modes:
        set_freeze(policy, freeze)
        n_train = trainable_params(policy)
        log(f"freeze_vision_encoder={freeze}  trainable params={n_train / 1e6:.1f}M")
        for bs in batch_sizes:
            sub = {
                k: (v[:bs] if torch.is_tensor(v) or isinstance(v, list) else v)
                for k, v in batch.items()
            }
            optimizer = torch.optim.AdamW(
                [p for p in policy.parameters() if p.requires_grad], lr=1e-4
            )
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                policy.train()
                # two warm-up steps: the first allocates Adam state, the second reaches steady state
                for _ in range(2):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss, _ = policy.forward(sub)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                peak = torch.cuda.max_memory_allocated() / 1024**3
                # Timed steps. What this is for: deciding between 8 GPUs at batch B and 4 GPUs at
                # batch 2B, which are the same effective batch. If step time scales linearly with the
                # per-GPU batch the two arrangements have identical throughput; sub-linear scaling
                # (better GPU utilisation at the larger batch) favours splitting the GPUs.
                torch.cuda.synchronize()
                started = time.perf_counter()
                n_timed = 3
                for _ in range(n_timed):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss, _ = policy.forward(sub)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                step_s = (time.perf_counter() - started) / n_timed
                rows.append((freeze, bs, peak, step_s))
                log(
                    f"  batch={bs:3d}  peak={peak:6.2f} GiB  step={step_s:.3f}s  "
                    f"({bs / step_s:6.1f} samples/s/GPU)"
                )
            except torch.cuda.OutOfMemoryError:
                rows.append((freeze, bs, float("nan"), float("nan")))
                log(f"  batch={bs:3d}  OOM")
            del optimizer
            policy.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    log("")
    log(f"{'freeze':>7s} {'batch':>6s} {'peak GiB':>9s} {'step s':>8s} {'smp/s/GPU':>10s}")
    for freeze, bs, peak, step_s in rows:
        rate = bs / step_s if step_s == step_s and step_s else float("nan")
        log(f"{str(freeze):>7s} {bs:6d} {peak:9.2f} {step_s:8.3f} {rate:10.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
