#!/usr/bin/env python
"""Turn null eef_pixel list ELEMENTS into real NaN floats.

Out-of-frame rows carry the pixel as arrow nulls inside the list rather than as NaN. pandas hides
this -- it shows array([nan, nan]) -- but the datasets library yields [None, None], and
hf_transform_to_torch then dies with "Could not infer dtype of NoneType" on the first out-of-frame
frame it touches. It is invisible until a specific row is read, so a script that happens to sample
only in-frame rows passes and one that does not fails.

Writing genuine float32 NaN keeps the same meaning (the EEF is not in this frame -- eef_in_frame
already says so) and is loadable.

    python tools/normalise_eef_pixel_nulls.py --write
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TREES = [
    "visual_robust_new_barx_ur5e/new_barx/UR5eOmron_PnPSinkToCounter/lerobot",
    "barx_panda_ur5e_iiwa/IIWAOmron/pretrain/PnPSinkToCounter/lerobot",
    "barx_panda_ur5e_iiwa/PandaOmron/pretrain/PnPSinkToCounter/lerobot",
    "barx_panda_ur5e_iiwa/UR5eOmron/pretrain/PnPSinkToCounter/lerobot",
]
ROOT = Path("/dataset/jiyun/dataset_git")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    for tree in TREES:
        for path in sorted(glob.glob(str(ROOT / tree / "data" / "**" / "*.parquet"), recursive=True)):
            frame = pd.read_parquet(path)
            if "eef_pixel" not in frame.columns:
                continue
            fixed = np.full((len(frame), 2), np.nan, dtype=np.float32)
            for i, value in enumerate(frame["eef_pixel"].to_numpy()):
                if value is None:
                    continue
                arr = np.asarray(value, dtype=object)
                fixed[i] = [np.nan if v is None else float(v) for v in arr]
            nans = int(np.isnan(fixed).any(axis=1).sum())
            # NaN must be built through pyarrow from a numpy buffer. Going via pandas objects --
            # frame["eef_pixel"] = list(fixed); frame.to_parquet(...) -- silently turns every NaN
            # back into an arrow null, which is the exact thing being fixed. A numpy array carries
            # no null mask, so pa.array() keeps NaN as a value.
            flat = pa.array(fixed.reshape(-1), type=pa.float32())
            offsets = pa.array(np.arange(len(fixed) + 1, dtype=np.int32) * 2, type=pa.int32())
            pixel = pa.ListArray.from_arrays(offsets, flat)
            frame["eef_in_frame"] = frame["eef_in_frame"].astype(bool)
            table = pa.Table.from_pandas(frame.drop(columns=["eef_pixel"]), preserve_index=False)
            table = table.append_column("eef_pixel", pixel)
            if args.write:
                pq.write_table(table, path)
            print(f"[nulls] {Path(path).parent.parent.parent.name}/{Path(path).name}: "
                  f"{len(frame)} rows, {nans} without a pixel"
                  + ("  WROTE" if args.write else "  (dry run)"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
