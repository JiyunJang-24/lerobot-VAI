#!/usr/bin/env python
"""Turn selfws_v2 into a frame cache for Experiment 1, with derived robot masks.

selfws_v2 stores one underlying state per ROW and renders it with every embodiment, each
embodiment in its own mp4. Reading a row through LeRobotDataset therefore means seeking 30
separate video streams, which is far too slow to train on. This decodes each stream ONCE,
sequentially, keeps a strided subset of frames, and writes one npz per episode.

The masks are the reason this script exists rather than a generic cache. No dataset in the
repository ships a robot segmentation mask, but the 30 renders of a row are pixel-aligned -- same
scene, same camera, same instant, only the arm differs -- so the per-pixel MEDIAN across
embodiments is a render of the scene with no robot in it, and

    mask_e = |I_e - median_E I| > tau

is that embodiment's robot region. Cheap, exact enough, and it needs no external model. What it
actually covers is the robot UNION the scene it occludes UNION any object this embodiment has
displaced differently from the others; the third term is contamination and is measured by
--report-divergence.

    python tools/exp1_build_cache.py --tree kitchen --stride 5 --workers 8
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import numpy as np
import pandas as pd

ROOT = Path("/dataset/jiyun/dataset_git/selfws_v2")
OUT = Path("/dataset/jiyun/exp1_cache")
VIEW = "robot0_agentview_right"


def log(msg: str) -> None:
    print(f"[cache] {msg}", flush=True)


def embodiments(root: Path) -> list[str]:
    info = json.loads((root / "meta" / "info.json").read_text())
    keys = [k for k in info["features"] if k.endswith(VIEW) and k.startswith("observation.images.")]
    return sorted(k.split(".")[2] for k in keys)


def decode(path: Path, wanted: dict[int, int], n_out: int, hw: tuple[int, int]) -> np.ndarray:
    """wanted maps absolute frame number -> output slot."""
    out = np.zeros((n_out, hw[0], hw[1], 3), dtype=np.uint8)
    hi = max(wanted)
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            slot = wanted.get(i)
            if slot is not None:
                out[slot] = frame.to_ndarray(format="rgb24")
            if i >= hi:
                break
    return out


def build_episode(args_tuple) -> str:
    tree, ep, stride, tau, out_dir = args_tuple
    root = ROOT / tree
    embs = embodiments(root)
    meta = pd.concat(
        [pd.read_parquet(p) for p in sorted((root / "meta" / "episodes").glob("*/*.parquet"))]
    ).set_index("episode_index")
    row = meta.loc[ep]
    fps = json.loads((root / "meta" / "info.json").read_text())["fps"]

    table = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").glob("*/*.parquet"))])
    table = table[table.episode_index == ep].sort_values("frame_index")
    frames = table.frame_index.to_numpy()[::stride]
    if len(frames) == 0:
        return f"ep{ep:03d} EMPTY"
    sel = table[table.frame_index.isin(frames)].sort_values("frame_index")

    images = None
    for e_i, emb in enumerate(embs):
        key = f"videos/observation.images.{emb}.{VIEW}"
        vp = (root / f"observation.images.{emb}.{VIEW}"
              / f"chunk-{int(row[key + '/chunk_index']):03d}"
              / f"file-{int(row[key + '/file_index']):03d}.mp4")
        if not vp.exists():  # v3.0 keeps videos under videos/<key>/...
            vp = (root / "videos" / f"observation.images.{emb}.{VIEW}"
                  / f"chunk-{int(row[key + '/chunk_index']):03d}"
                  / f"file-{int(row[key + '/file_index']):03d}.mp4")
        off = int(round(float(row[key + "/from_timestamp"]) * fps))
        wanted = {off + int(f): i for i, f in enumerate(frames)}
        with av.open(str(vp)) as c:
            s = c.streams.video[0]
            hw = (s.codec_context.height, s.codec_context.width)
        block = decode(vp, wanted, len(frames), hw)
        if images is None:
            images = np.zeros((len(frames), len(embs), *hw, 3), dtype=np.uint8)
        images[:, e_i] = block

    # derived masks -- median across the embodiment axis is the robot-free scene
    med = np.median(images.astype(np.float32), axis=1, keepdims=True)
    diff = np.abs(images.astype(np.float32) - med).mean(-1)
    masks = np.packbits(diff > tau, axis=-1)

    eef = np.stack([np.stack(sel[f"eef_state.{e}"].to_numpy()) for e in embs], 1).astype(np.float32)
    reach = np.stack([sel[f"reachable.{e}"].to_numpy() for e in embs], 1).astype(np.float32)
    perr = np.stack([sel[f"pos_err_m.{e}"].to_numpy() for e in embs], 1).astype(np.float32)
    rerr = np.stack([sel[f"rot_err_deg.{e}"].to_numpy() for e in embs], 1).astype(np.float32)

    path = out_dir / f"ep{ep:03d}.npz"
    np.savez(path, images=images, masks=masks, eef=eef, reachable=reach, pos_err=perr,
             rot_err=rerr, frame_index=frames.astype(np.int32),
             robot_frac=(diff > tau).mean((2, 3)).astype(np.float32))
    mb = path.stat().st_size / 1e6
    return f"ep{ep:03d} T={len(frames)} E={len(embs)} {mb:.0f}MB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", default="kitchen", choices=["kitchen", "no_kitchen"])
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--tau", type=float, default=25.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--episodes", type=str, default="")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    root = ROOT / args.tree
    out_dir = args.out or (OUT / f"selfws_{args.tree}")
    out_dir.mkdir(parents=True, exist_ok=True)
    embs = embodiments(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    eps = (sorted(int(x) for x in args.episodes.split(","))
           if args.episodes else list(range(info["total_episodes"])))
    log(f"{args.tree}: {len(eps)} episodes x {len(embs)} embodiments, stride {args.stride}")

    (out_dir / "embodiments.json").write_text(json.dumps(embs, indent=1))
    todo = [(args.tree, e, args.stride, args.tau, out_dir) for e in eps
            if not (out_dir / f"ep{e:03d}.npz").exists()]
    log(f"{len(eps) - len(todo)} already cached, {len(todo)} to build")

    t0 = time.time()
    if args.workers <= 1:
        for t in todo:
            log(build_episode(t))
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            for i, msg in enumerate(pool.map(build_episode, todo), 1):
                log(f"[{i}/{len(todo)}] {msg}  ({time.time() - t0:.0f}s)")
    log(f"done in {time.time() - t0:.0f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
