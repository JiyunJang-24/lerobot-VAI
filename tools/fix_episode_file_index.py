#!/usr/bin/env python
"""Idempotent fix for a v3.0 LeRobot dataset where an episode's declared data file location
(meta/episodes' `data/chunk_index` / `data/file_index`) doesn't match where its rows are actually
stored. Seen in RoboCasa "mg" (MimicGen) exports converted via this project's
convert_dataset_v21_to_v30.py: at every size-based data-file boundary, the first episode physically
written to the new file is mis-declared as still belonging to the previous file. `dataset_tools`
(split_dataset/remove_feature) trusts the declared location to decide which source file to read an
episode from, so a mismatched episode ends up with zero rows in the file that gets read for it,
which surfaces as `ValueError: cannot convert float NaN to integer` in `_copy_and_reindex_data`.

Fix: for every episode, scan actual data/*.parquet files for where its episode_index rows really
live, and correct meta/episodes' data/chunk_index + data/file_index to match wherever they disagree.
Doesn't touch any data row or video. Skips (no-op) episodes that are already correct, so safe to
call on every dataset unconditionally.

Usage:
    python tools/fix_episode_file_index.py --root /path/to/some/v3.0/lerobot/dataset
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


def log(msg: str) -> None:
    print(f"[fix_episode_file_index] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    root: Path = args.root
    ds = LeRobotDataset(repo_id="fix_episode_file_index_src", root=root)

    actual_location: dict[int, tuple[int, int]] = {}
    for data_path in sorted(root.glob("data/chunk-*/file-*.parquet")):
        rel = data_path.relative_to(root)
        chunk_idx = int(rel.parts[1].split("-")[1])
        file_idx = int(rel.parts[2].split("-")[1].split(".")[0])
        episode_indices = pq.read_table(data_path, columns=["episode_index"]).column("episode_index").to_pylist()
        for ep_idx in set(episode_indices):
            actual_location[ep_idx] = (chunk_idx, file_idx)

    mismatches: dict[int, tuple[int, int]] = {}
    for ep_idx in range(ds.meta.total_episodes):
        declared = (ds.meta.episodes[ep_idx]["data/chunk_index"], ds.meta.episodes[ep_idx]["data/file_index"])
        real = actual_location.get(ep_idx)
        if real is not None and declared != real:
            mismatches[ep_idx] = real

    if not mismatches:
        log(f"{root}: no data/chunk_index or data/file_index mismatches found, nothing to do")
        return

    log(f"{root}: fixing {len(mismatches)} mismatched episode(s): {sorted(mismatches.keys())}")

    for episodes_path in sorted(root.glob("meta/episodes/chunk-*/file-*.parquet")):
        df = pd.read_parquet(episodes_path)
        touched = df["episode_index"].isin(mismatches.keys())
        if not touched.any():
            continue
        for row_idx in df.index[touched]:
            ep_idx = int(df.at[row_idx, "episode_index"])
            new_chunk, new_file = mismatches[ep_idx]
            df.at[row_idx, "data/chunk_index"] = new_chunk
            df.at[row_idx, "data/file_index"] = new_file
        tmp_path = episodes_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp_path)
        tmp_path.replace(episodes_path)
        log(f"{episodes_path}: corrected {int(touched.sum())} row(s)")

    log(f"{root}: done, fixed {len(mismatches)} episode(s)")


if __name__ == "__main__":
    main()
