#!/usr/bin/env python
"""Look at what a contrastive batch on selfws_v2 actually contains: positives, negatives, and the
arm x gripper grid the positives are drawn from.

Unlike tools/explain_siglip_pairs.py, which needs a trained encoder and assumes the 3-view export,
this inspects the DATA only -- no model, no checkpoint. It answers "what would the loss be asked to
pull together and push apart" before any training is run.

Definitions used by the visual-robust losses, and by this script:

    POSITIVE   two renders of the SAME frame with a DIFFERENT embodiment. Same scene, same
               instant, same camera pose -- only the robot differs.
    NEGATIVE   renders of DIFFERENT frames. With same_episode_negatives=true (the default for the
               contrastive term) those other frames come from the SAME episode, which makes them
               deliberately hard: same kitchen, same lighting, most objects in the same place.

selfws_v2 is laid out as an arm x gripper grid, so the positive group has structure the earlier
3-view export did not: pairs that differ only in the gripper (same arm), only in the arm (same
gripper), or in both. Three figures are written:

    <out>/pairs_positive.png    one frame across every embodiment -- the positive group
    <out>/pairs_negative.png    one embodiment across several frames -- the negative set
    <out>/pairs_grid.png        the arm x gripper grid for one frame, laid out as a matrix

    python tools/inspect_selfws_pairs.py --tree kitchen --episode 0
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

from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset  # noqa: E402

DEFAULT_ROOT = Path("/dataset/jiyun/dataset_git/selfws_v2")


def log(msg: str) -> None:
    print(f"[inspect_pairs] {msg}", flush=True)


def split_tag(tag: str) -> tuple[str, str]:
    """'IIWAOmron_R85' -> ('IIWAOmron', 'R85'); 'UR5eOmron' -> ('UR5eOmron', '-')."""
    known = ("_R85", "_R140", "_PG", "_RG", "_AG")
    for suffix in known:
        if tag.endswith(suffix):
            return tag[: -len(suffix)], suffix[1:]
    return tag, "-"


def front_views(item) -> list[str]:
    return sorted(
        k
        for k, v in item.items()
        if k.startswith("observation.images.")
        and torch.is_tensor(v)
        and v.ndim == 3
        and "eye_in_hand" not in k
    )


def show(ax, image, title=None, color=None):
    ax.imshow(image.permute(1, 2, 0).numpy().clip(0, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7, color=color or "black")
    if color:
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--tree", default="kitchen", choices=["kitchen", "no_kitchen"])
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frame-offset", type=int, default=60, help="frame within the episode for positives")
    ap.add_argument("--negatives", type=int, default=5, help="how many other frames to show as negatives")
    ap.add_argument("--negative-stride", type=int, default=40)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/selfws_pairs")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ds = MultiLeRobotDataset(
        [args.tree], root=args.root, delta_timestamps={args.tree: None},
        visual_cue_mode="vanilla", use_wrist_cam=False, use_state=True, cache_in_memory=False,
    )
    episodes = ds.meta_episodes
    start = int(episodes["dataset_from_index"][args.episode])
    stop = int(episodes["dataset_to_index"][args.episode])
    anchor = min(start + args.frame_offset, stop - 1)

    item = ds[anchor]
    keys = front_views(item)
    tags = [k.split(".")[2] for k in keys]
    log(f"{args.tree}: episode {args.episode} spans frames {start}..{stop - 1}, anchor = {anchor}")
    log(f"{len(keys)} front views")

    # ---- figure 1: the positive group -------------------------------------------------------
    cols = 6
    rows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.0 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (key, tag) in enumerate(zip(keys, tags, strict=True)):
        ax = axes[i // cols, i % cols]
        ax.axis("on")
        arm, grip = split_tag(tag)
        show(ax, item[key], f"{arm}\n{grip}", color="tab:green")
    fig.suptitle(
        f"POSITIVE group -- {args.tree}, episode {args.episode}, frame {anchor - start}: "
        f"one instant rendered by all {len(keys)} embodiments\n"
        f"the contrastive loss pulls every pair of these together",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out / "pairs_positive.png", dpi=100, bbox_inches="tight")
    log(f"wrote {args.out / 'pairs_positive.png'}")

    # ---- figure 2: negatives ------------------------------------------------------------------
    neg_indices = [anchor + (i + 1) * args.negative_stride for i in range(args.negatives)]
    neg_indices = [i for i in neg_indices if i < stop]
    pick_tags = tags[: min(4, len(tags))]
    pick_keys = [keys[tags.index(t)] for t in pick_tags]

    n_cols = 1 + len(neg_indices)
    fig, axes = plt.subplots(len(pick_keys), n_cols, figsize=(2.3 * n_cols, 2.0 * len(pick_keys)))
    axes = np.atleast_2d(axes)
    for r, (key, tag) in enumerate(zip(pick_keys, pick_tags, strict=True)):
        show(axes[r, 0], item[key], f"ANCHOR\n{tag}", color="tab:green")
        for c, idx in enumerate(neg_indices, start=1):
            other = ds[int(idx)]
            show(axes[r, c], other[key], f"+{idx - anchor} frames", color="tab:red")
    fig.suptitle(
        f"NEGATIVE pairs -- same embodiment, different frames of episode {args.episode}\n"
        f"green = anchor, red = negative. With same_episode_negatives=true these are the hard "
        f"negatives the loss pushes apart",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(args.out / "pairs_negative.png", dpi=100, bbox_inches="tight")
    log(f"wrote {args.out / 'pairs_negative.png'}")

    # ---- figure 3: the arm x gripper grid -----------------------------------------------------
    parsed = [split_tag(t) for t in tags]
    arms = sorted({a for a, _ in parsed})
    grips = sorted({g for _, g in parsed})
    fig, axes = plt.subplots(len(arms), len(grips), figsize=(2.2 * len(grips), 1.9 * len(arms)))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for (arm, grip), key in zip(parsed, keys, strict=True):
        ax = axes[arms.index(arm), grips.index(grip)]
        ax.axis("on")
        show(ax, item[key])
    for r, arm in enumerate(arms):
        axes[r, 0].set_ylabel(arm, fontsize=9)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
    for c, grip in enumerate(grips):
        axes[0, c].set_title(grip, fontsize=10)
    fig.suptitle(
        f"the positive group is an ARM x GRIPPER grid ({len(arms)} arms x {len(grips)} grippers, "
        f"{len(keys)} present)\n"
        f"a pair along a row differs only in the gripper; along a column, only in the arm",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(args.out / "pairs_grid.png", dpi=100, bbox_inches="tight")
    log(f"wrote {args.out / 'pairs_grid.png'}")

    # ---- console summary ----------------------------------------------------------------------
    n = len(keys)
    print()
    log(f"batch arithmetic for one frame with all {n} views in the positive group:")
    log(f"    positive ordered pairs : {n * (n - 1):5d}   (same frame, different embodiment)")
    log(f"    negatives per anchor   : depends on batch size B -> {n} * (B-1) * {n} per frame pair")
    log(f"  arm x gripper: {len(arms)} arms x {len(grips)} grippers, {n} of "
        f"{len(arms) * len(grips)} combinations present")
    same_arm = sum(1 for a in arms for g1 in grips for g2 in grips
                   if g1 < g2 and (a, g1) in parsed and (a, g2) in parsed)
    same_grip = sum(1 for g in grips for a1 in arms for a2 in arms
                    if a1 < a2 and (a1, g) in parsed and (a2, g) in parsed)
    log(f"    pairs differing ONLY in gripper (same arm) : {same_arm}")
    log(f"    pairs differing ONLY in arm (same gripper) : {same_grip}")
    log(f"    pairs differing in both                    : {n * (n - 1) // 2 - same_arm - same_grip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
