#!/usr/bin/env python
"""Render the actual positive/negative structure the visual-robust contrastive loss sees.

Reproduces the trainer's own pipeline exactly -- same MultiLeRobotDataset, same
SameEpisodeBatchSampler, same `_select_visual_robust_image_keys` (imported, not reimplemented) --
and draws one figure per contrastive group (left / right).

How to read a figure: `_supervised_contrastive_loss` is called with
    labels = arange(batch_size).repeat_interleave(num_views)
so the grouping is purely positional:

  * one ROW  = one sampled frame, shown under every selected view  -> all POSITIVES of each other
  * different ROWS                                                 -> NEGATIVES of each other

With `same_episode_negatives=True` (what training uses) every row comes from the SAME episode, so
the negatives are other timesteps of that episode rather than unrelated scenes -- which is the whole
point: the model cannot separate them by scene identity alone.

No policy/model is loaded and nothing touches the GPU, so this is safe to run beside a live training
job.

Usage:
    python tools/dump_visual_robust_pairs.py \
        --root dataset_git/visual_robust_robocasa_x_lr \
        --out outputs/visual_robust_pairs
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset  # noqa: E402
from lerobot.scripts.lerobot_train_with_visual_robust import (  # noqa: E402
    SameEpisodeBatchSampler,
    _select_visual_robust_image_keys,
)

def discover_repo_ids(root: Path) -> list[str]:
    """Find the <task>/lerobot trees under root rather than hard-coding names -- the task directory
    names differ per export (PickPlaceCounterToStove... vs IIWAOmron_PnPCounterToSink...)."""
    return sorted(f"{p.parent.parent.parent.name}/lerobot" for p in root.glob("*/lerobot/meta/info.json"))


def log(msg: str) -> None:
    print(f"[dump_visual_robust_pairs] {msg}", flush=True)


def to_img(t: torch.Tensor) -> np.ndarray:
    a = t.detach().float().cpu().numpy()
    if a.ndim == 4:  # (T, C, H, W) -> last frame
        a = a[-1]
    return np.clip(a.transpose(1, 2, 0), 0, 1)


def short(key: str, prefix: str) -> str:
    return key[len(prefix) :] if key.startswith(prefix) else key


def draw(batch_items, keys, prefix, title, out_path):
    n_rows, n_cols = len(batch_items), len(keys)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.5 * n_cols, 2.75 * n_rows), squeeze=False
    )
    for r, item in enumerate(batch_items):
        for c, key in enumerate(keys):
            ax = axes[r][c]
            ax.imshow(to_img(item[key]))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(f"C{r}")
                spine.set_linewidth(3)
            if r == 0:
                ax.set_title(short(key, prefix), fontsize=8)
            if c == 0:
                ax.set_ylabel(
                    f"sample {r}\nep {int(item['episode_index'])} / f {int(item['frame_index'])}",
                    fontsize=8,
                    color=f"C{r}",
                )
    fig.suptitle(
        f"{title}\n"
        "same row (same colour) = POSITIVES  |  different rows = NEGATIVES  "
        "(same episode, different timestep)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    log(f"wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "dataset_git/visual_robust_robocasa_x_lr")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/visual_robust_pairs")
    parser.add_argument("--batch-size", type=int, default=3, help="frames per contrastive batch")
    parser.add_argument("--max-views", type=int, default=4, help="views per group per step")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prefixes",
        type=str,
        default="observation.image.",
        help="Comma-separated contrastive group prefixes, matching --dataset.visual_robust_front_prefixes",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    repo_ids = discover_repo_ids(args.root)
    if not repo_ids:
        log(f"no <task>/lerobot datasets under {args.root}")
        return 1
    log(f"repo_ids: {repo_ids}")

    ds = MultiLeRobotDataset(
        repo_ids,
        root=args.root,
        delta_timestamps={r: None for r in repo_ids},
        visual_cue_mode="vanilla",
        use_wrist_cam=False,
        use_state=True,
        cache_in_memory=False,
    )
    log(f"dataset: {ds.num_episodes} episodes, {ds.num_frames} frames")

    sampler = SameEpisodeBatchSampler(ds.meta_episodes, batch_size=args.batch_size, shuffle=True)
    indices = next(iter(sampler))
    log(f"sampled frame indices (same episode): {indices}")

    items = [ds[i] for i in indices]
    batch = {
        k: torch.stack([it[k] for it in items])
        for k in items[0]
        if torch.is_tensor(items[0][k])
    }

    for prefix in [p.strip() for p in args.prefixes.split(",") if p.strip()]:
        side = prefix.rstrip(".").rsplit(".", 1)[-1]
        available = _select_visual_robust_image_keys(batch, image_prefix=prefix)
        keys = _select_visual_robust_image_keys(
            batch, image_prefix=prefix, max_views=args.max_views, random_views=True
        )
        log(f"{side}: {len(available)} views available, {len(keys)} selected -> {[short(k, prefix) for k in keys]}")
        if len(keys) < 2:
            log(f"{side}: fewer than 2 views, contrastive loss would be skipped")
            continue
        draw(
            items,
            keys,
            prefix,
            f"visual-robust contrastive group: {side.upper()}  "
            f"({len(keys)} of {len(available)} views, batch {len(items)})",
            args.out / f"pairs_{side}.png",
        )

    # Cross-group sanity image: left and right of the SAME frame are in different groups, so they are
    # never pulled together -- this is what the per-prefix split buys.
    prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip()]
    lkeys = _select_visual_robust_image_keys(batch, image_prefix=prefixes[0], max_views=2, random_views=True) if len(prefixes) > 1 else []
    rkeys = _select_visual_robust_image_keys(batch, image_prefix=prefixes[-1], max_views=2, random_views=True) if len(prefixes) > 1 else []
    if lkeys and rkeys:
        fig, axes = plt.subplots(len(items), 4, figsize=(10, 2.75 * len(items)), squeeze=False)
        cols = [(k, "left") for k in lkeys[:2]] + [(k, "right") for k in rkeys[:2]]
        for r, item in enumerate(items):
            for c, (key, side) in enumerate(cols):
                ax = axes[r][c]
                ax.imshow(to_img(item[key]))
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor("tab:blue" if side == "left" else "tab:red")
                    spine.set_linewidth(3)
                if r == 0:
                    ax.set_title(f"[{side}] {key.split('.', 3)[-1]}", fontsize=8)
        fig.suptitle(
            "left group (blue) vs right group (red)\n"
            "these are the SAME frames, but the two groups are contrasted SEPARATELY --\n"
            "a left view is never a positive of a right view",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        fig.savefig(args.out / "groups_left_vs_right.png", dpi=110)
        plt.close(fig)
        log(f"wrote {args.out / 'groups_left_vs_right.png'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
