#!/usr/bin/env python
"""Re-score the pose-ladder models on ONE common evaluation set.

Each ladder run trains AND is scored on pairs drawn from its own pose subset, so the 48-pose run is
tested on an easier motion distribution than the 144-pose run: 936 distinct pose pairs against
4000, and a narrower range of rotations. Comparing those raw numbers measures task difficulty as
much as model quality, which is the same confound the embodiment ladder had with its shrinking
held-out set (CLAUDE.md 9.4).

This scores every model on the SAME pairs -- drawn from all 144 poses, on held-out embodiments --
so the only thing that differs is what each model was trained on.

    python tools/eval_pose_ladder_common.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.scripts.motion_data import (  # noqa: E402
    build_table, geodesic_deg, load_cache, motion_labels, pose_states, sample_pose_pairs,
)
from lerobot.scripts.train_motion_prediction import (  # noqa: E402
    MotionHead, build_index, evaluate, load_tower,
)

SUBSET = "56combo_144_bg12_closed"


def log(msg: str) -> None:
    print(f"[common] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="pose48,pose96,pose144")
    ap.add_argument("--eval-samples", type=int, default=128)
    ap.add_argument("--pose-pairs", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/pose_ladder_common.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = build_table([SUBSET])
    states = pose_states(table)
    index = build_index(table, [SUBSET])
    cache = load_cache([SUBSET])

    # ONE pair pool, from ALL 144 poses, fixed by seed -- identical for every model scored.
    rng = np.random.default_rng(20260904)
    pairs = sample_pose_pairs(states, args.pose_pairs, 0.15, rng)
    pos_scale = float(np.linalg.norm(motion_labels(states, pairs)[0], axis=1).std())
    log(f"common eval pool: {len(pairs)} pairs from all {len(states)} poses")

    results = {"eval_pairs": int(len(pairs)), "n_poses_in_eval": int(len(states)), "runs": {}}
    for name in args.runs.split(","):
        run_dir = REPO_ROOT / "outputs/motion_prediction" / name
        info = json.loads((run_dir / "results.json").read_text())
        heldout = info["heldout_embodiments"]
        tower = load_tower(str(run_dir / "vision_tower.safetensors"), device)
        head = MotionHead(int(tower.config.hidden_size)).to(device)
        head.load_state_dict(torch.load(run_dir / "motion_head.pt")["head"])
        tower.eval(), head.eval()
        rows = evaluate(tower, head, index, pairs, states, heldout, cache, device,
                        np.random.default_rng(7), args.eval_samples, 24, True, pos_scale)
        summary = {
            "n_poses_trained_on": info["n_poses_used"],
            "translation_mae_m": float(np.mean([r["translation_mae_m"] for r in rows])),
            "translation_dir_cos": float(np.mean([r["translation_dir_cos"] for r in rows])),
            "rotation_err_deg": float(np.mean([r["rotation_err_deg"] for r in rows])),
            "gripper_acc": float(np.mean([r["gripper_acc"] for r in rows])),
        }
        results["runs"][name] = summary
        log(f"{name:<8} trained on {summary['n_poses_trained_on']:>3} poses -> "
            f"{summary['translation_mae_m'] * 100:5.2f} cm  cos {summary['translation_dir_cos']:.3f}  "
            f"rot {summary['rotation_err_deg']:5.1f} deg")
        del tower, head
        torch.cuda.empty_cache()

    mean_pred = motion_labels(states, pairs)[0].mean(0)
    results["baseline_translation_mae_m"] = float(
        np.abs(motion_labels(states, pairs)[0] - mean_pred).mean())
    log(f"predict-the-mean baseline on this pool: {results['baseline_translation_mae_m'] * 100:.2f} cm")
    args.out.write_text(json.dumps(results, indent=2))
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
