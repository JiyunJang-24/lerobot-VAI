#!/usr/bin/env python
"""Local-only conversion of a LeRobot dataset directory from codebase_version "v2.0" to "v2.1".

`src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py` (the converter this project already uses
via `convert_v21_to_v30_multiple.sh`) hard-requires its input to be "v2.1": it reads
`meta/episodes_stats.jsonl`, which "v2.0" datasets never had (they only have a single, global
`meta/stats.json`). Upstream lerobot used to ship a `convert_dataset_v20_to_v21.py` for exactly this
gap, but it was removed from the codebase once v3.0 landed, and it only worked against datasets
already pushed to the HF Hub.

This script is a local-filesystem-only reimplementation of that missing step:
  - Fixes any feature declared with dtype "object" in `meta/info.json` (not a valid arrow/HF dtype --
    seen on some RoboCasa MimicGen exports for `observation.state`/`action`) by reading the real
    dtype straight off the episode's own parquet column.
  - Computes real per-episode statistics directly from the raw `data/*.parquet` (and, for video
    features, sampled decoded frames from `videos/*.mp4`) -- without instantiating `LeRobotDataset`,
    since that class refuses to load anything below codebase_version "v3.0".
  - Writes `meta/episodes_stats.jsonl`.
  - Removes the deprecated `meta/stats.json`.
  - Bumps `codebase_version` to "v2.1" in `meta/info.json`.

Video per-episode stats are sampled coarsely (see --video-sample-frames): this project trains with
`dataset.use_imagenet_stats=true` by default, which overrides camera-key stats with fixed ImageNet
constants at load time (see `lerobot/datasets/factory.py::make_dataset`), so the *video* stats
written here are not used for normalization -- only state/action/scalar features are, and those are
computed exactly (no sampling).

Safe to re-run: datasets that are already >= v2.1 are left untouched.

Usage:
    python tools/convert_v20_to_v21_local.py --root /path/to/dataset_dir
"""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import jsonlines
import numpy as np
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import get_feature_stats, sample_indices
from lerobot.datasets.utils import load_info, serialize_dict, write_info
from lerobot.datasets.video_utils import decode_video_frames

V20 = "v2.0"
V21 = "v2.1"
EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
STATS_PATH = "meta/stats.json"

_ARROW_TO_DTYPE = {
    "double": "float64",
    "float": "float32",
    "int64": "int64",
    "int32": "int32",
    "bool": "bool",
}


def _fix_invalid_dtype_labels(root: Path, info: dict) -> bool:
    """Fix features declared with dtype "object" (not a valid arrow/HF dtype) by reading the real
    dtype off episode 0's own parquet column. Only touches genuinely invalid labels -- a real
    difference like float32 vs float64 between two otherwise-valid datasets is a data
    characteristic, not a bug, and is left alone here."""
    invalid_keys = [k for k, ft in info["features"].items() if ft["dtype"] == "object"]
    if not invalid_keys:
        return False

    data_path = root / info["data_path"].format(episode_chunk=0, episode_index=0)
    schema = pq.read_schema(data_path)
    changed = False
    for key in invalid_keys:
        arrow_type = schema.field(key).type
        elem_type = arrow_type.value_type if hasattr(arrow_type, "value_type") else arrow_type
        real_dtype = _ARROW_TO_DTYPE.get(str(elem_type))
        if real_dtype is None:
            raise ValueError(
                f"{root}: feature '{key}' has invalid dtype label 'object' and its real arrow type "
                f"'{elem_type}' isn't in the known mapping -- add it to _ARROW_TO_DTYPE."
            )
        print(f"[fix] {root}: feature '{key}' dtype 'object' -> '{real_dtype}' (from parquet schema)")
        info["features"][key]["dtype"] = real_dtype
        changed = True
    return changed


def _compute_one_episode_stats(
    root: Path, info: dict, ep_idx: int, video_sample_frames: int, video_backend: str
) -> dict:
    chunk_idx = ep_idx // info["chunks_size"]
    data_path = root / info["data_path"].format(episode_chunk=chunk_idx, episode_index=ep_idx)
    df = pq.read_table(data_path).to_pandas()
    ep_len = len(df)
    fps = info["fps"]

    ep_stats = {}
    for key, ft in info["features"].items():
        dtype = ft["dtype"]
        if dtype == "string":
            continue

        if dtype in ("image", "video"):
            video_path = root / info["video_path"].format(
                episode_chunk=chunk_idx, video_key=key, episode_index=ep_idx
            )
            num_samples = min(video_sample_frames, ep_len)
            sampled = np.round(np.linspace(0, ep_len - 1, num_samples)).astype(int).tolist()
            timestamps = [idx / fps for idx in sampled]
            frames = decode_video_frames(video_path, timestamps, tolerance_s=1.0 / fps, backend=video_backend)
            ep_ft_array = frames.numpy().astype(np.float32) / 255.0
            axes_to_reduce = (0, 2, 3)
            keepdims = True
        else:
            ep_ft_array = np.array(df[key].tolist())
            axes_to_reduce = 0
            keepdims = ep_ft_array.ndim == 1

        stats = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)
        if dtype in ("image", "video"):
            stats = {k: v if k == "count" else np.squeeze(v, axis=0) for k, v in stats.items()}
        ep_stats[key] = stats

    return ep_stats


def convert(
    root: Path, num_workers: int = 8, video_sample_frames: int = 8, video_backend: str = "pyav"
) -> None:
    root = Path(root)
    info = load_info(root)
    version = info.get("codebase_version", "unknown")
    if version != V20:
        print(f"[skip] {root}: codebase_version={version!r} (expected {V20!r}); nothing to do")
        return

    _fix_invalid_dtype_labels(root, info)

    total_episodes = info["total_episodes"]
    print(f"Computing per-episode stats for {total_episodes} episodes in {root} ({num_workers} workers)")

    results = [None] * total_episodes
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _compute_one_episode_stats, root, info, ep_idx, video_sample_frames, video_backend
            ): ep_idx
            for ep_idx in range(total_episodes)
        }
        done = 0
        for future in as_completed(futures):
            ep_idx = futures[future]
            results[ep_idx] = future.result()
            done += 1
            if done % 200 == 0 or done == total_episodes:
                print(f"  {done}/{total_episodes}")

    stats_path = root / EPISODES_STATS_PATH
    if stats_path.exists():
        stats_path.unlink()
    with jsonlines.open(stats_path, mode="w") as writer:
        for ep_idx, ep_stats in enumerate(results):
            writer.write({"episode_index": ep_idx, "stats": serialize_dict(ep_stats)})

    info["codebase_version"] = V21
    write_info(info, root)

    old_stats = root / STATS_PATH
    if old_stats.exists():
        old_stats.unlink()

    print(f"[done] {root}: v2.0 -> v2.1")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Local dataset directory (contains meta/, data/, videos/)")
    parser.add_argument("--num-workers", type=int, default=8, help="Thread pool size for per-episode stats computation")
    parser.add_argument(
        "--video-sample-frames",
        type=int,
        default=8,
        help="Number of frames to sample per episode per video key when computing stats (kept low: unused for "
        "training normalization when dataset.use_imagenet_stats=true, the project default)",
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="pyav",
        help="Video decoding backend for stats sampling (default: pyav, since torchcodec needs a "
        "matching system FFmpeg that may not be installed)",
    )
    args = parser.parse_args()
    convert(
        args.root,
        num_workers=args.num_workers,
        video_sample_frames=args.video_sample_frames,
        video_backend=args.video_backend,
    )
