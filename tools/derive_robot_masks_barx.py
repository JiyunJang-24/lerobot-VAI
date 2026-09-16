#!/usr/bin/env python
"""Derive robot masks for the paired barx renders, and write a video to judge them by.

No barx dataset ships a segmentation mask, but visual_robust_new_barx_ur5e renders the SAME frame
from six embodiments, pixel-aligned: same kitchen, same instant, same camera, only the arm differs.
So the background can be recovered from the renders themselves and whatever deviates from it is
that embodiment's robot.

The plain per-frame median used in the selfws experiment does not survive the move to barx. There
it had 30 embodiments covering ~19% of the frame; here there are 6 covering ~30%, they all share a
base position and reach for the same target, so their coverage is strongly correlated and at many
pixels a MAJORITY is robot -- the median then returns a robot colour and those pixels never get
masked. That is the hole you see in the middle of the base column.

Three changes fix most of it:

  CONSENSUS INSTEAD OF MEDIAN. Across embodiments the background is not merely similar, it is the
  same render -- identical up to codec noise. So instead of asking for the middle value, find the
  largest group of embodiments that agree with each other at that pixel. Two agreeing embodiments
  are enough to pin the background; the median needs four of six.

  TEMPORAL FALLBACK. Where every embodiment disagrees (all six occlude the pixel with a different
  arm) no cross-embodiment estimate exists at all. But the kitchen is static, so that pixel was
  observable at other times in the episode. The per-pixel median of the confident estimates over
  time fills those holes -- which is exactly the base-column case.

  MORPHOLOGY. Close, then fill interior holes, then drop components below a minimum area. Masks
  derived by differencing are speckled by construction; this removes the speckle without moving
  the boundary much.

What the result still is, and this does not change: robot UNION the scene the robot occludes. The
paper-towel roll gets masked whenever an arm is in front of where it would otherwise be. Only a
simulator segmentation pass removes that.

    python tools/derive_robot_masks_barx.py --episode 0 --out outputs/barx_masks
"""

import argparse
import glob
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path("dataset_git/visual_robust_new_barx_ur5e/new_barx/UR5eOmron_PnPSinkToCounter/lerobot")
VIEW = "robot0_agentview_right"


def log(msg: str) -> None:
    print(f"[masks] {msg}", flush=True)


def embodiment_keys(root: Path) -> list[str]:
    info = json.loads((root / "meta" / "info.json").read_text())
    return [k for k in info["features"] if k.endswith(VIEW) and k.startswith("observation.images.")]


def read_episode(root: Path, key: str, ep: int, fps: int = 20) -> np.ndarray:
    meta = pd.concat([pd.read_parquet(p) for p in glob.glob(str(root / "meta/episodes/*/*.parquet"))])
    row = meta.set_index("episode_index").loc[ep]
    k = f"videos/{key}"
    vp = (root / k / f"chunk-{int(row[k + '/chunk_index']):03d}"
          / f"file-{int(row[k + '/file_index']):03d}.mp4")
    start = int(round(float(row[k + "/from_timestamp"]) * fps))
    length = int(row["length"])
    out = []
    with av.open(str(vp)) as c:
        for i, frame in enumerate(c.decode(video=0)):
            if i < start:
                continue
            if i >= start + length:
                break
            out.append(frame.to_ndarray(format="rgb24"))
    return np.stack(out)


def consensus_background(stack: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """stack (E,H,W,3) at one timestep -> (background, confidence) where confidence is the size of
    the largest agreeing group of embodiments."""
    d = np.abs(stack[:, None] - stack[None, :]).mean(-1)   # (E,E,H,W)
    votes = (d < tol).sum(1)                                # (E,H,W)
    best = votes.max(0)                                     # (H,W)
    member = votes == best[None]
    bg = (stack * member[..., None]).sum(0) / member.sum(0)[..., None]
    return bg, best


def build(stack: np.ndarray, tol: float, thresh: float, min_area: int,
          min_confidence: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """stack (T,E,H,W,3) -> masks (T,E,H,W) bool and backgrounds (T,H,W,3)."""
    t_count = stack.shape[0]
    bgs = np.zeros((t_count, *stack.shape[2:]), dtype=np.float32)
    conf = np.zeros((t_count, *stack.shape[2:4]), dtype=np.int16)
    for t in range(t_count):
        bgs[t], conf[t] = consensus_background(stack[t].astype(np.float32), tol)

    # temporal fallback: a pixel with no agreeing pair at time t is filled from the times where it
    # did have one. The kitchen is static, so this is a legitimate substitution for the scenery --
    # and it is the only way to recover the pixels every arm covers at once.
    trusted = conf >= min_confidence
    filled = int((~trusted).sum())
    if filled:
        stat = np.zeros(stack.shape[2:], dtype=np.float32)
        for c in range(3):
            chan = np.where(trusted, bgs[..., c], np.nan)
            with np.errstate(all="ignore"):
                stat[..., c] = np.nanmedian(chan, axis=0)
        stat = np.nan_to_num(stat, nan=0.0)
        bgs = np.where(trusted[..., None], bgs, stat[None])
    log(f"temporal fallback applied to {filled / conf.size * 100:.1f}% of (t,pixel) cells")

    masks = np.zeros(stack.shape[:4], dtype=bool)
    for t in range(t_count):
        for e in range(stack.shape[1]):
            raw = np.abs(stack[t, e].astype(np.float32) - bgs[t]).mean(-1) > thresh
            m = ndimage.binary_closing(raw, np.ones((5, 5)))
            m = ndimage.binary_fill_holes(m)
            lab, n = ndimage.label(m)
            if n:
                areas = ndimage.sum(m, lab, range(1, n + 1))
                keep = np.isin(lab, 1 + np.flatnonzero(areas >= min_area))
                m = keep
            masks[t, e] = m
    return masks, bgs


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    h, w = frames[0].shape[:2]
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.options = {"crf": "20"}
        for f in frames:
            c.mux(s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")))
        c.mux(s.encode())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--tol", type=float, default=8.0, help="codec tolerance for 'these agree'")
    ap.add_argument("--thresh", type=float, default=18.0, help="deviation that counts as robot")
    ap.add_argument("--min-area", type=int, default=40)
    ap.add_argument("--show", type=int, default=3, help="embodiments to put in the video")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("outputs/barx_masks"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    keys = embodiment_keys(args.root)
    names = [k.split(".")[2] for k in keys]
    log(f"episode {args.episode}: {len(keys)} embodiments {names}")
    stack = np.stack([read_episode(args.root, k, args.episode) for k in keys], axis=1)
    log(f"loaded {stack.shape[0]} frames, {stack.nbytes / 1e9:.2f} GB")

    masks, bgs = build(stack, args.tol, args.thresh, args.min_area)
    log(f"mean masked fraction per embodiment: "
        f"{[round(float(masks[:, e].mean()), 3) for e in range(len(keys))]}")

    show = list(range(min(args.show, len(keys))))
    frames = []
    for t in range(stack.shape[0]):
        top, bot = [], []
        for e in show:
            img = stack[t, e]
            ov = img.astype(np.float32).copy()
            m = masks[t, e]
            ov[m] = ov[m] * 0.35 + np.array([236, 72, 84]) * 0.65
            top.append(img)
            bot.append(ov.clip(0, 255).astype(np.uint8))
        top.append(bgs[t].clip(0, 255).astype(np.uint8))
        bot.append(np.repeat((masks[t, show[0]] * 255)[..., None], 3, -1).astype(np.uint8))
        frames.append(np.concatenate([np.concatenate(top, 1), np.concatenate(bot, 1)], 0))

    vid = args.out / f"ep{args.episode:03d}_masks.mp4"
    write_video(vid, frames, args.fps)
    log(f"wrote {vid}  ({len(frames)} frames, layout: "
        f"{'|'.join(names[e] for e in show)}|background  /  overlays|binary)")
    np.savez_compressed(args.out / f"ep{args.episode:03d}_masks.npz",
                        masks=np.packbits(masks, axis=-1), embodiments=np.array(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
