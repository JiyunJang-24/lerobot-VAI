#!/usr/bin/env python
"""Generate meta/episodes_stats.jsonl (and stats.json) for an export that shipped without them.

The v2.1 -> v3.0 converter needs per-episode min/max/mean/std/count for every feature. This export
omits both files entirely, so conversion dies on a missing path before touching any data.

Video features get stats too, because aggregate_stats iterates over every declared feature. They
are computed from a handful of decoded frames per episode rather than the whole stream: these are
only used for normalisation bookkeeping, nothing in this project consumes image statistics, and
decoding 448k frames to fill in a field nobody reads would cost hours.

    python tools/generate_episode_stats.py <dataset_root> --frames-per-episode 8
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def log(msg: str) -> None:
    print(f"[stats] {msg}", flush=True)


def column_stats(values: np.ndarray) -> dict:
    """min/max/mean/std per dimension, as lists -- the shape lerobot expects."""
    flat = values.reshape(len(values), -1).astype(np.float64)
    return {
        "min": flat.min(0).tolist(),
        "max": flat.max(0).tolist(),
        "mean": flat.mean(0).tolist(),
        "std": (flat.std(0) + 0.0).tolist(),
        "count": [len(flat)],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--frames-per-episode", type=int, default=8)
    args = ap.parse_args()

    info = json.loads((args.root / "meta" / "info.json").read_text())
    features = info["features"]
    video_keys = [k for k, v in features.items() if v["dtype"] == "video"]
    files = sorted(glob.glob(str(args.root / "data" / "**" / "*.parquet"), recursive=True))
    log(f"{len(files)} parquet, {len(video_keys)} video features")

    # Decode straight from the mp4 with torchcodec: LeRobotDataset cannot open this tree yet,
    # since it is still v2.1 and that is exactly what we are preparing it for.
    image_stats = {}
    if video_keys:
        from torchcodec.decoders import VideoDecoder

        log(f"sampling {args.frames_per_episode} frames per stream for image statistics ...")
        for key in video_keys:
            clips = sorted(glob.glob(str(args.root / "videos" / "**" / key / "*.mp4"),
                                     recursive=True))
            samples = []
            for clip in clips[:: max(1, len(clips) // 8)]:
                decoder = VideoDecoder(clip)
                idx = np.linspace(0, len(decoder) - 1, args.frames_per_episode).astype(int)
                frames = decoder.get_frames_at(sorted(set(int(i) for i in idx))).data
                samples.append(frames.float().div(255.0).mean(dim=(2, 3)).numpy())
            arr = np.concatenate(samples)
            image_stats[key] = {
                "min": arr.min(0).reshape(3, 1, 1).tolist(),
                "max": arr.max(0).reshape(3, 1, 1).tolist(),
                "mean": arr.mean(0).reshape(3, 1, 1).tolist(),
                "std": (arr.std(0) + 1e-8).reshape(3, 1, 1).tolist(),
                "count": [int(len(arr))],
            }
            log(f"  {key}: mean {np.round(arr.mean(0), 3).tolist()} over {len(arr)} frames")

    rows = []
    for path in files:
        frame = pd.read_parquet(path)
        for episode, group in frame.groupby("episode_index"):
            stats = {}
            for key in features:
                if key in video_keys:
                    stats[key] = image_stats[key]
                elif key in group.columns:
                    values = group[key].to_numpy()
                    values = (np.stack(values) if isinstance(values[0], np.ndarray)
                              else values.reshape(-1, 1))
                    stats[key] = column_stats(values)
            rows.append({"episode_index": int(episode), "stats": stats})
    rows.sort(key=lambda r: r["episode_index"])

    out = args.root / "meta" / "episodes_stats.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    log(f"wrote {out} ({len(rows)} episodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
