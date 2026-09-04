#!/usr/bin/env python
"""Show exactly what one VLM training sample is: the two images, the prompt, and the answer.

Two sampling policies side by side, because they encode different claims about what the model is
supposed to key on:

  strict      everything except pose and gripper is identical. The visual difference between the
              two frames IS the motion.
  robot-fixed the ROBOT is identical -- both its morphology (embodiment) and its paint
              (color_variant) -- while background and furniture are re-drawn for the second frame.

color_variant is part of the robot, not the scene: cv 0/1/2 repaint the SAME arm silver, yellow
and pink. Letting it change would make the model solve a correspondence puzzle that never happens
in a real trajectory, so it is held fixed in both policies.

    python tools/preview_vlm_motion_batch.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.scripts.motion_data import EEF_ROOT, build_table, sample_pose_pairs  # noqa: E402
from lerobot.scripts.motion_language import (  # noqa: E402
    MotionDescriber, fit_thresholds, motion_deltas,
)
from lerobot.scripts.train_vlm_motion import QUESTION, SUBSETS, build_index  # noqa: E402

N_BG, N_VIEW = 12, 4


def log(msg: str) -> None:
    print(msg, flush=True)


def draw_sample(index, pairs, embodiments, rng, policy: str):
    """One (pos_t, pos_h, meta) triple under the given policy."""
    n_sub = index.shape[2]
    for _ in range(4000):
        emb = int(rng.choice(embodiments))
        i, j = pairs[rng.integers(len(pairs))]
        color = int(rng.integers(index.shape[5]))   # the robot's paint -- same in both frames
        if policy == "strict":
            furn = int(rng.integers(max(1, n_sub // 2))) * 2
            sub_t = furn + int(rng.integers(min(2, n_sub)))
            sub_h = furn + int(rng.integers(min(2, n_sub)))
            bg_t = bg_h = int(rng.integers(N_BG))
        else:  # robot-fixed: background and furniture re-drawn, the robot untouched
            sub_t, sub_h = int(rng.integers(n_sub)), int(rng.integers(n_sub))
            bg_t, bg_h = int(rng.integers(N_BG)), int(rng.integers(N_BG))
        view_t = view_h = int(rng.integers(N_VIEW))
        pos_t = index[emb, i, sub_t, bg_t, view_t, color]
        pos_h = index[emb, j, sub_h, bg_h, view_h, color]
        if pos_t >= 0 and pos_h >= 0:
            return pos_t, pos_h, dict(emb=emb, color=color, pose=(int(i), int(j)),
                                      sub=(sub_t, sub_h), bg=(bg_t, bg_h),
                                      view=(view_t, view_h))
    raise RuntimeError("no renderable sample found")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/vlm_motion_batch.png")
    args = ap.parse_args()

    table = build_table(SUBSETS)
    index = build_index(table, SUBSETS)
    first = table.drop_duplicates("pose").sort_values("pose")
    states = np.stack(first["observation.state"].to_numpy()).astype(np.float64)[:, :7]

    rng = np.random.default_rng(args.seed)
    pairs = sample_pose_pairs(states, 8000, 0.15, rng)
    pad = lambda idx, grip: np.concatenate(  # noqa: E731
        [states[idx][None, :], np.array([[grip]])], axis=1)
    a = np.concatenate([states[pairs[:, 0]], np.zeros((len(pairs), 1))], axis=1)
    b = np.concatenate([states[pairs[:, 1]], np.zeros((len(pairs), 1))], axis=1)
    d_pos, d_euler, _ = motion_deltas(a, b, "fixed")
    describer = MotionDescriber(fit_thresholds(d_pos), fit_thresholds(d_euler),
                                grip_threshold=0.5, frame="fixed")

    heldout = {0, 1, 3, 8, 12, 14, 23, 27, 28, 33, 34, 36, 42, 49}
    train_emb = np.array(sorted(set(range(int(table.embodiment.max()) + 1)) - heldout))

    datasets = {s: LeRobotDataset(f"eef_pairs/{s}", root=EEF_ROOT / s) for s in SUBSETS}
    offsets, running = {}, 0
    for s in SUBSETS:
        offsets[s] = running
        running += int((table.subset == s).sum())

    def frame(cache_pos):
        row = table.iloc[int(cache_pos)]
        local = int(row["row"]) - int(table[table.subset == row["subset"]]["row"].min())
        img = datasets[row["subset"]][local]["observation.images.agentview_right"]
        img = img[-1] if img.ndim == 4 else img
        return (img * 255).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy(), row

    policies = ["strict", "robot-fixed"]
    # The frames are 320x180. Give each cell that aspect explicitly, or tight_layout shrinks the
    # images to slivers inside tall axes and the figure is unreadable.
    fig, axes = plt.subplots(args.rows, 4, figsize=(4 * 3.5, args.rows * 2.75),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.95})
    for r in range(args.rows):
        for c, policy in enumerate(policies):
            pos_t, pos_h, meta = draw_sample(index, pairs, train_emb,
                                             np.random.default_rng(args.seed * 100 + r), policy)
            img_t, row_t = frame(pos_t)
            img_h, row_h = frame(pos_h)
            s_t = pad(meta["pose"][0], 1.0 if meta["sub"][0] % 2 == 0 else 0.0)
            s_h = pad(meta["pose"][1], 1.0 if meta["sub"][1] % 2 == 0 else 0.0)
            answer = describer.describe(s_t, s_h)[0]
            for k, (img, tag) in enumerate([(img_t, "image 1"), (img_h, "image 2")]):
                ax = axes[r, c * 2 + k]
                ax.imshow(img)
                ax.set_xticks([]); ax.set_yticks([])
                names = ["closed", "open", "closed+furn", "open+furn"]
                idx = 0 if k == 0 else 1
                changed = "" if k == 0 else " ".join(
                    w for w, a, b in [("bg", *meta["bg"]), ("view", *meta["view"]),
                                      ("furniture", meta["sub"][0] // 2, meta["sub"][1] // 2)]
                    if a != b)
                head = (f"{tag}   bg{meta['bg'][idx]} view{meta['view'][idx]} "
                        f"{names[meta['sub'][idx]]}")
                if changed:
                    head += f"\n({changed} CHANGED)"
                ax.set_title(head, fontsize=7.5,
                             color="#b3153b" if changed else "black")
            import textwrap

            wrapped = textwrap.fill(f'robot {meta["emb"]} (paint {meta["color"]})  →  "{answer}"',
                                    width=88)
            axes[r, c * 2].text(0.0, -0.12, wrapped, transform=axes[r, c * 2].transAxes,
                                fontsize=7.5, color="#0b3d5c", va="top", linespacing=1.35)
            log(f"[{policy:<11}] robot {meta['emb']:>2} paint {meta['color']}  "
                f"pose {meta['pose']}  bg {meta['bg']}  view {meta['view']}  sub {meta['sub']}")
            log(f'             -> "{answer}"')
    fig.suptitle(
        f"One VLM training sample: [image 1, image 2] + \"{QUESTION}\"  ->  the sentence\n"
        "left = strict (only pose and gripper differ)   •   right = robot-fixed "
        "(background and furniture re-drawn; the ROBOT, paint included, is identical)", fontsize=11)
    for c, policy in enumerate(policies):
        axes[0, c * 2].annotate(policy.upper(), xy=(1.02, 1.42), xycoords="axes fraction",
                                ha="center", fontsize=13, weight="bold",
                                color="#3d5573" if c == 0 else "#b3153b")
    fig.savefig(args.out, dpi=115, bbox_inches="tight")
    log(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
