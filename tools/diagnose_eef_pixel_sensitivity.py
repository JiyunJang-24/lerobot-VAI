#!/usr/bin/env python
"""Which axis breaks the EEF-pixel head -- background, furniture, camera view, or framing?

"It is sensitive to the background" is the obvious reading of the policy-corpus failure, but the
head was TRAINED on 12 backgrounds and on furniture recolours, so that reading needs testing rather
than assuming. eef_pairs carries ground truth and varies each axis independently, so every number
below is measured against real labels instead of eyeballed.

Four axes, three of them real variation already in the data and one synthetic:

    background_index    12 backgrounds, all seen during pre-training
    subset (_furniture) same pose, recoloured furniture -- also seen
    camera_view_index   4 canonical angles -- also seen
    framing (synthetic) zoom out so the robot occupies less of the frame, which is the ONE thing
                        the policy corpus does that eef_pairs never does

All rows are embodiments held out of pre-training, so nothing here is scored on training data.

    python tools/diagnose_eef_pixel_sensitivity.py --checkpoint outputs/siglip_pretrain/all4_n42_all
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

EEF_ROOT = Path("/dataset/jiyun/dataset_git/eef_pairs")
SUBSETS = ["56combo_48_bg12_closed", "56combo_48_bg12_closed_furniture"]
IMAGE_KEY = "observation.images.agentview_right"
W, H = 320, 180


def log(msg: str) -> None:
    print(f"[sensitivity] {msg}", flush=True)


def subset_table(subset: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(EEF_ROOT / subset / "data" / "**" / "*.parquet"), recursive=True))
    table = pd.concat([
        pd.read_parquet(f, columns=[
            "episode_index", "frame_index", "observation.embodiment_index",
            "observation.background_index", "observation.camera_view_index",
            "observation.eef_pixel",
        ]) for f in files
    ], ignore_index=True)
    lengths = table.groupby("episode_index").size().sort_index()
    starts = lengths.cumsum().shift(fill_value=0)
    table["row"] = table.episode_index.map(starts) + table.frame_index
    return table.rename(columns={
        "observation.embodiment_index": "embodiment",
        "observation.background_index": "background",
        "observation.camera_view_index": "view",
    })


def load_frames(subset: str, rows: np.ndarray) -> torch.Tensor:
    dataset = LeRobotDataset(f"eef_pairs/{subset}", root=EEF_ROOT / subset)
    out = torch.empty(len(rows), 3, H, W, dtype=torch.uint8)
    for i, row in enumerate(rows):
        frame = dataset[int(row)][IMAGE_KEY]
        frame = frame[-1] if frame.ndim == 4 else frame
        out[i] = (frame * 255).round().clamp(0, 255).to(torch.uint8)
    return out


def zoom_out(images: torch.Tensor, uv: np.ndarray, scale: float):
    """Shrink the content about the image centre, padding with edge colour.

    This is the axis eef_pairs never varies: its robot always fills the frame, while in the policy
    corpus the arm is small and often half outside it. The label moves with the content, so the task
    is unchanged -- only the apparent size is.
    """
    small_h, small_w = int(round(H * scale)), int(round(W * scale))
    resized = torch.nn.functional.interpolate(
        images.float(), size=(small_h, small_w), mode="bilinear", align_corners=False)
    canvas = images.float().median(dim=3, keepdim=True).values.median(dim=2, keepdim=True).values
    canvas = canvas.expand(-1, -1, H, W).clone()
    top, left = (H - small_h) // 2, (W - small_w) // 2
    canvas[:, :, top:top + small_h, left:left + small_w] = resized
    moved = np.stack([uv[:, 0] * scale + left, uv[:, 1] * scale + top], axis=1)
    return canvas.round().clamp(0, 255).to(torch.uint8), moved


def zoom_in(images: torch.Tensor, uv: np.ndarray, scale: float):
    """Crop the centre and resize back to full frame -- apparent size UP, and no padded border.

    This is the control for zoom_out. That one pads, which is itself a distribution shift, so on its
    own it cannot say whether the head minds the apparent size or the artificial frame around it.
    Cropping introduces no border at all, so if this degrades too, apparent size is the axis.
    """
    crop_h, crop_w = int(round(H * scale)), int(round(W * scale))
    top, left = (H - crop_h) // 2, (W - crop_w) // 2
    cropped = images[:, :, top:top + crop_h, left:left + crop_w].float()
    grown = torch.nn.functional.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)
    moved = np.stack([(uv[:, 0] - left) / scale, (uv[:, 1] - top) / scale], axis=1)
    inside = (moved[:, 0] >= 0) & (moved[:, 0] < W) & (moved[:, 1] >= 0) & (moved[:, 1] < H)
    return grown.round().clamp(0, 255).to(torch.uint8)[inside], moved[inside]


def translate(images: torch.Tensor, uv: np.ndarray, dx: int, dy: int):
    """Shift the content, edge-padding what opens up. Apparent size unchanged."""
    shifted = torch.roll(images, shifts=(dy, dx), dims=(2, 3))
    if dy > 0:
        shifted[:, :, :dy] = shifted[:, :, dy:dy + 1]
    elif dy < 0:
        shifted[:, :, dy:] = shifted[:, :, dy - 1:dy]
    if dx > 0:
        shifted[:, :, :, :dx] = shifted[:, :, :, dx:dx + 1]
    elif dx < 0:
        shifted[:, :, :, dx:] = shifted[:, :, :, dx - 1:dx]
    moved = np.stack([uv[:, 0] + dx, uv[:, 1] + dy], axis=1)
    inside = (moved[:, 0] >= 0) & (moved[:, 0] < W) & (moved[:, 1] >= 0) & (moved[:, 1] < H)
    return shifted[inside], moved[inside]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO_ROOT / "outputs/siglip_pretrain/all4_n42_all")
    ap.add_argument("--per-group", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/eef_pixel_sensitivity.png")
    args = ap.parse_args()

    from tools.predict_eef_pixel import load_tower_and_head, predict  # noqa: PLC0415

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = json.loads((args.checkpoint / "pretrain_info.json").read_text())
    holdout = info["holdout_embodiments"]
    tower, head = load_tower_and_head(args.checkpoint, device)
    log(f"{args.checkpoint.name}, scoring only held-out embodiments {holdout}")

    rng = np.random.default_rng(args.seed)
    results = {}

    plain = subset_table(SUBSETS[0])
    plain = plain[plain.embodiment.isin(holdout)]

    def score(table, subset, n, tag):
        picks = table.sample(n=min(n, len(table)), random_state=int(rng.integers(1 << 30)))
        images = load_frames(subset, picks["row"].to_numpy())
        truth = np.stack(picks["observation.eef_pixel"].to_numpy()).astype(np.float64)
        pred = predict(tower, head, images, device)
        err = np.linalg.norm(pred - truth, axis=1)
        log(f"  {tag:<34} {err.mean():6.2f} px   (n={len(err)})")
        return err, images, truth

    log("per background (all 12 were seen during pre-training):")
    for bg in sorted(plain.background.unique()):
        err, _, _ = score(plain[plain.background == bg], SUBSETS[0], args.per_group, f"background {bg}")
        results[f"bg {bg}"] = err.mean()

    log("per camera view (all 4 were seen):")
    view_err = {}
    for view in sorted(plain.view.unique()):
        err, _, _ = score(plain[plain.view == view], SUBSETS[0], args.per_group, f"view {view}")
        view_err[f"view {view}"] = err.mean()

    log("furniture recolour, same poses (seen):")
    furn = subset_table(SUBSETS[1])
    furn = furn[furn.embodiment.isin(holdout)]
    subset_err = {}
    base_err, base_img, base_truth = score(plain, SUBSETS[0], args.per_group * 4, "plain")
    subset_err["plain"] = base_err.mean()
    f_err, _, _ = score(furn, SUBSETS[1], args.per_group * 4, "furniture")
    subset_err["furniture"] = f_err.mean()

    log("framing -- the one axis eef_pairs never varies:")
    frame_err = {"1.00 native": base_err.mean()}
    for scale in (0.75, 0.55, 0.40, 0.30):
        images, truth = zoom_out(base_img, base_truth, scale)
        pred = predict(tower, head, images, device)
        err = np.linalg.norm(pred - truth, axis=1)
        log(f"  {f'zoom OUT {scale:.2f} (padded)':<34} {err.mean():6.2f} px")
        frame_err[f"out {scale:.2f}"] = err.mean()
    for scale in (0.75, 0.55):
        images, truth = zoom_in(base_img, base_truth, scale)
        pred = predict(tower, head, images, device)
        err = np.linalg.norm(pred - truth, axis=1)
        log(f"  {f'zoom IN {scale:.2f} (no border)':<34} {err.mean():6.2f} px   (n={len(err)})")
        frame_err[f"in {scale:.2f}"] = err.mean()
    for dx, dy in ((40, 0), (0, 30)):
        images, truth = translate(base_img, base_truth, dx, dy)
        pred = predict(tower, head, images, device)
        err = np.linalg.norm(pred - truth, axis=1)
        log(f"  {f'shift dx={dx} dy={dy} (size kept)':<34} {err.mean():6.2f} px   (n={len(err)})")
        frame_err[f"shift {dx},{dy}"] = err.mean()

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.0))
    panels = [
        (results, "12 backgrounds", "seen in pre-training", "#3d5573"),
        (view_err, "4 camera views", "seen in pre-training", "#3d5573"),
        (subset_err, "furniture recolour", "seen in pre-training", "#3d5573"),
        (frame_err, "image geometry", "only 4 fixed camera poses seen", "#b3153b"),
    ]
    for ax, (data, title, sub, colour) in zip(axes, panels, strict=True):
        keys = list(data)
        ax.bar(range(len(keys)), [data[k] for k in keys], color=colour)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{title}\n{sub}", fontsize=10)
        ax.set_ylabel("pixel error" if ax is axes[0] else "")
        ax.grid(alpha=0.3, axis="y")
        ax.axhline(base_err.mean(), ls="--", c="gray", lw=1)
    top = max(max(d.values()) for d, *_ in panels)
    for ax in axes:
        ax.set_ylim(0, top * 1.1)
    fig.suptitle(
        f"What actually breaks the EEF-pixel head? ({args.checkpoint.name}, held-out embodiments only)\n"
        "appearance is free; ANY camera geometry outside the 4 it was trained on is not   "
        "(dashed line = unperturbed error)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(args.out, dpi=115, bbox_inches="tight")
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
