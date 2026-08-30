#!/usr/bin/env python
"""Declare the per-embodiment reachability columns that selfws_v2 stores but does not declare.

The selfws_v2 export writes four extra columns per embodiment into every parquet --

    eef_state.<tag>     float32[8]  xyz(3) + quat_xyzw(4) + gripper_opening_normalized(1)
    reachable.<tag>     bool        the arm held the commanded eef pose within tolerance
    pos_err_m.<tag>     float32
    rot_err_deg.<tag>   float32

-- and documents them in meta/embodiment_coverage.json, but meta/info.json's `features` block
never lists them. Nothing that opens the dataset through LeRobotDataset can survive that: the
parquet carries 129 columns while info.json declares 69, and `Dataset.from_parquet` fails its cast
with "column names don't match" before any of the v3.0 conversion steps get a chance to run.

This adds the missing declarations, derived from the parquet itself rather than assumed, so the
file describes the data that is actually there. Idempotent: running it twice is a no-op.

    python tools/declare_selfws_extra_features.py /dataset/jiyun/dataset_git/selfws_v2/kitchen
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd


def log(msg: str) -> None:
    print(f"[declare_extra_features] {msg}", flush=True)


def infer_feature(series: pd.Series) -> dict:
    """Map a pandas column to a LeRobot feature declaration."""
    sample = series.iloc[0]
    if hasattr(sample, "__len__") and not isinstance(sample, (str, bytes)):
        return {"dtype": "float32", "shape": [len(sample)], "names": None}
    if pd.api.types.is_bool_dtype(series):
        return {"dtype": "bool", "shape": [1], "names": None}
    if pd.api.types.is_integer_dtype(series):
        return {"dtype": "int64", "shape": [1], "names": None}
    return {"dtype": "float32", "shape": [1], "names": None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="dataset root holding meta/info.json and data/")
    args = ap.parse_args()

    info_path = args.root / "meta" / "info.json"
    if not info_path.exists():
        log(f"no {info_path}")
        return 1

    parquets = sorted(glob.glob(str(args.root / "data" / "**" / "*.parquet"), recursive=True))
    if not parquets:
        log(f"no parquet files under {args.root / 'data'}")
        return 1

    info = json.loads(info_path.read_text())
    declared = set(info["features"])
    frame = pd.read_parquet(parquets[0])

    added = {}
    for column in frame.columns:
        if column in declared:
            continue
        added[column] = infer_feature(frame[column])

    if not added:
        log(f"{args.root.name}: nothing to add, all {len(frame.columns)} columns already declared")
        return 0

    info["features"].update(added)
    info_path.write_text(json.dumps(info, indent=4))
    log(f"{args.root.name}: declared {len(added)} columns "
        f"({len(declared)} -> {len(info['features'])})")
    for column in list(added)[:4]:
        log(f"    {column}  {added[column]['dtype']}{added[column]['shape']}")
    if len(added) > 4:
        log(f"    ... and {len(added) - 4} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
