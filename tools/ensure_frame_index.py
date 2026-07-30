"""Idempotent fix for a v3.0 LeRobot dataset missing a `frame_index` column (position of a frame
within its episode) -- seen so far in RoboCasa "mg" (MimicGen) exports, unlike "human" exports which
have it. `LeRobotDatasetMetadata.create()` (used internally by `lerobot.datasets.dataset_tools`, e.g.
`split_dataset`/`remove_feature`) always declares `frame_index` as a feature via its DEFAULT_FEATURES
merge -- regardless of what the source dataset actually has. That mismatch between declared and
actual columns makes `datasets.Dataset.from_dict(..., features=...)` raise `ValueError: Keys
mismatch` the moment any dataset_tools function touches such a dataset.

Fix: add a real `frame_index` column (0..episode_length-1 per episode, computed directly from each
data file's already-sorted `episode_index` column) to every data/*.parquet file, and declare it in
meta/info.json + meta/stats.json to match. Purely additive -- no existing column, row, or video is
touched. Skips entirely if `frame_index` is already declared (idempotent), so it's safe to call on
every dataset unconditionally (e.g. from convert_robocasa_to_v30.sh, after ensuring v3.0).

Usage:
    python tools/ensure_frame_index.py --root /path/to/some/v3.0/lerobot/dataset
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def log(msg: str) -> None:
    print(f"[ensure_frame_index] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    root: Path = args.root
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())

    if "frame_index" in info["features"]:
        log(f"{root}: frame_index already declared, nothing to do")
        return

    data_files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not data_files:
        raise SystemExit(f"No data parquet files found under {root}/data")

    all_frame_indices = []
    for path in data_files:
        table = pq.read_table(path)
        episode_index = table.column("episode_index").to_numpy()
        if np.any(np.diff(episode_index) < 0):
            raise SystemExit(f"{path}: episode_index is not sorted; frame_index computation assumes sorted rows")

        # position within each contiguous run of the same episode_index
        boundaries = np.flatnonzero(np.diff(episode_index)) + 1
        run_starts = np.concatenate(([0], boundaries))
        starts_per_row = np.repeat(run_starts, np.diff(np.concatenate((run_starts, [len(episode_index)]))))
        frame_index = (np.arange(len(episode_index)) - starts_per_row).astype(np.int64)

        table = table.append_column("frame_index", pa.array(frame_index, type=pa.int64()))
        tmp_path = path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp_path)
        tmp_path.replace(path)
        all_frame_indices.append(frame_index)
        log(f"{path}: wrote frame_index for {len(frame_index)} rows")

    combined = np.concatenate(all_frame_indices)
    info["features"]["frame_index"] = {"dtype": "int64", "shape": [1], "names": None, "fps": info.get("fps")}
    info_path.write_text(json.dumps(info, indent=4))

    stats_path = root / "meta/stats.json"
    stats = json.loads(stats_path.read_text())
    stats["frame_index"] = {
        "min": [int(combined.min())],
        "max": [int(combined.max())],
        "mean": [float(combined.mean())],
        "std": [float(combined.std())],
        "count": [int(combined.shape[0])],
    }
    stats_path.write_text(json.dumps(stats, indent=4))

    log(f"{root}: done, frame_index added to {len(data_files)} data file(s) + info.json + stats.json")


if __name__ == "__main__":
    main()
