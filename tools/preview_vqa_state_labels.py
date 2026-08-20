#!/usr/bin/env python
"""Render the VQA gripper-state labels next to the frames they describe.

The point is to see, with your own eyes, that the sentence tracks the arm: that "high" rises as the
gripper lifts, that "closed" appears at the grasp, and that all six embodiment renders of one
instant carry the SAME answer -- which is the entire invariance mechanism.

    python tools/preview_vqa_state_labels.py --episode 0 --frames 6
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset  # noqa: E402
from lerobot.policies.smolvla.vqa_state_text import (  # noqa: E402
    VQAStateTokenizer,
    quaternion_xyzw_to_yaw_deg,
    wrap_deg,
)

DEFAULT_ROOT = REPO_ROOT / "dataset_git/visual_robust_new_barx_ur5e/new_barx"
REPO_ID = "UR5eOmron_PnPSinkToCounter/lerobot"


class _NoTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frames", type=int, default=6, help="evenly spaced frames across the episode")
    ap.add_argument("--position-resolution-cm", type=float, default=1.0)
    ap.add_argument("--yaw-resolution-deg", type=float, default=5.0)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/vqa_labels_preview.png")
    args = ap.parse_args()

    # MultiLeRobotDataset, not LeRobotDataset: the latter checks the hub for a matching revision
    # and 404s on a local-only tree.
    dataset = MultiLeRobotDataset(
        [args.repo_id], root=args.root, delta_timestamps={args.repo_id: None},
        visual_cue_mode="vanilla", use_wrist_cam=False, use_state=True, cache_in_memory=False,
    )
    labeller = VQAStateTokenizer(
        _NoTokenizer(),
        position_resolution_cm=args.position_resolution_cm,
        yaw_resolution_deg=args.yaw_resolution_deg,
    )

    episodes = dataset.meta_episodes
    start = int(episodes["dataset_from_index"][args.episode])
    stop = int(episodes["dataset_to_index"][args.episode])
    indices = np.linspace(start, stop - 1, args.frames).astype(int)
    print(f"episode {args.episode}: frames {start}..{stop - 1}, sampling {list(indices)}")

    probe = dataset[start]
    reference = probe["observation.state"].numpy()
    view_keys = sorted(
        k for k, v in probe.items()
        if k.startswith("observation.image") and torch.is_tensor(v) and v.ndim == 3 and "wrist" not in k
        and "eye_in_hand" not in k
    )
    print(f"{len(view_keys)} front views: {[k.split('.')[2] for k in view_keys]}\n")

    rows = []
    for idx in indices:
        item = dataset[int(idx)]
        state = item["observation.state"].numpy()
        sentence = labeller.sentences(state[None, :], reference[None, :])[0]
        dyaw = wrap_deg(
            quaternion_xyzw_to_yaw_deg(state[None, 3:7]) - quaternion_xyzw_to_yaw_deg(reference[None, 3:7])
        )[0]
        print(f"  frame {int(idx) - start:4d}  raw dx={100 * (state[0] - reference[0]):+6.1f} "
              f"dy={100 * (state[1] - reference[1]):+6.1f} z={100 * state[2]:6.1f} "
              f"dyaw={dyaw:+7.1f} grip={state[7]:.0f}\n            \"{sentence}\"")
        rows.append((int(idx) - start, [item[k] for k in view_keys], sentence))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows, n_cols = len(rows), len(view_keys)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.6 * n_rows + 0.6 * n_rows))
    axes = np.atleast_2d(axes)
    for r, (frame_no, images, sentence) in enumerate(rows):
        for c, img in enumerate(images):
            ax = axes[r, c]
            ax.imshow(img.permute(1, 2, 0).numpy().clip(0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(view_keys[c].split(".")[2], fontsize=9)
        axes[r, 0].set_ylabel(f"frame {frame_no}", fontsize=9)
        # One answer per row -- the same string for every render, which is the point.
        axes[r, 0].text(
            0, -0.14, f'A: "{sentence}"', transform=axes[r, 0].transAxes,
            fontsize=9, va="top", ha="left", family="monospace",
        )
    fig.suptitle(
        f'Q: "{VQAStateTokenizer.QUESTION}"   —   one answer shared by every embodiment render',
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
