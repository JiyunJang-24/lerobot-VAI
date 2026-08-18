#!/usr/bin/env python
"""How far did policy training move the SigLIP tower away from its pretrained weights?

This is the quantity the L2-SP penalty (dataset.vision_l2sp_weight) acts on, so it sets the scale
for choosing that coefficient: it says what "unregularised drift" is worth numerically, and later
runs can be compared against it.

Reports, for every discovered barx front-only checkpoint:
    ||w - w0||^2                  the raw penalty value at the end of training
    ||w - w0|| / ||w0||           unitless relative drift -- the readable one

Runs on CPU.

Usage:
    python tools/measure_vision_weight_drift.py
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def log(msg: str) -> None:
    print(f"[vision_drift] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-glob", type=str,
                    default="outputs/train/*/*barx_frontonly*/checkpoints/050000/pretrained_model")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    log("loading pretrained SmolVLM2 vision tower ...")
    pretrained = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", dtype=torch.float32
    ).model.vision_model.eval()
    reference = {name: p.detach().float() for name, p in pretrained.named_parameters()}
    ref_sq = float(sum(t.pow(2).sum() for t in reference.values()))
    n_params = sum(t.numel() for t in reference.values())
    log(f"reference: {n_params / 1e6:.1f}M params, ||w0||^2 = {ref_sq:.4g}")
    log("")

    rows = []
    for path in sorted(REPO_ROOT.glob(args.ckpt_glob)):
        cfg = json.loads((path / "train_config.json").read_text())
        ds, pol = cfg["dataset"], cfg["policy"]
        w = ds.get("visual_robust_contrastive_weight", 0.0)
        obj = ds.get("visual_robust_front_objective", "none")
        label = "baseline (no aux)" if not w else f"{obj} w={w}"
        if pol.get("freeze_vision_encoder"):
            label += " frozen"
        if ds.get("vision_l2sp_weight"):
            label += f" l2sp={ds['vision_l2sp_weight']}"
        if pol.get("knowledge_insulation"):
            # No dataset-side weight identifies these, so without this both KI runs print as
            # "baseline (no aux)" and the two rows are indistinguishable.
            label += f" KI {pol.get('ki_objective', 'fast')}"

        tower = (
            SmolVLAPolicy.from_pretrained(str(path))
            .model.vlm_with_expert.get_vlm_model()
            .vision_model.float()
            .cpu()
        )
        drift_sq = 0.0
        missing = 0
        for name, p in tower.named_parameters():
            if name not in reference:
                missing += 1
                continue
            drift_sq += float((p.detach().float() - reference[name]).pow(2).sum())
        if missing:
            log(f"  WARNING: {missing} parameters of {label} had no match in the reference")
        rows.append((label, drift_sq, (drift_sq / ref_sq) ** 0.5))
        log(f"  {label:28s} ||w-w0||^2={drift_sq:12.4g}   relative drift={rows[-1][2]:.5f}")

    log("")
    log(f"{'checkpoint':28s} {'||w-w0||^2':>14s} {'rel drift':>11s}")
    for label, drift_sq, rel in sorted(rows, key=lambda r: r[2]):
        log(f"{label:28s} {drift_sq:14.4g} {rel:11.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
