#!/usr/bin/env python
"""Declare parquet columns that info.json omits, for the LIBERO exp08 trees.

`observation.eef_base_rel` and `robot_base_pos` are present in the data and are exactly what the
base-relative EEF-state VQA needs, but they are missing from info.json's `features`. The datasets
library then refuses the whole tree with "An error occurred while generating the dataset", naming
no column, so the cause is invisible from the error.

The same trap appeared on selfws_v2 earlier in this project. Idempotent.

    python tools/declare_libero_extra_columns.py <tree> [<tree> ...]
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DTYPE = {"double": "float64", "float": "float32", "int64": "int64", "int32": "int32",
         "bool": "bool", "string": "string"}


def log(msg: str) -> None:
    print(f"[declare] {msg}", flush=True)


def describe(column) -> dict:
    """dtype and shape inferred from the data, since info.json has neither."""
    arrow = column.type
    if hasattr(arrow, "value_type"):  # list<...>
        inner = str(arrow.value_type)
        length = len(column[0].as_py()) if len(column) else 1
        return {"dtype": DTYPE.get(inner, "float32"), "shape": [int(length)], "names": None}
    return {"dtype": DTYPE.get(str(arrow), "float32"), "shape": [1], "names": None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trees", nargs="+", type=Path)
    args = ap.parse_args()

    for tree in args.trees:
        info_path = tree / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        files = sorted(glob.glob(str(tree / "data" / "**" / "*.parquet"), recursive=True))
        if not files:
            log(f"{tree.name}: no parquet")
            continue
        table = pq.read_table(files[0])
        missing = [c for c in table.column_names if c not in info["features"]]
        if not missing:
            log(f"{tree.name}: nothing to declare")
            continue
        for name in missing:
            info["features"][name] = describe(table[name])
            log(f"{tree.name}: declared {name} {info['features'][name]['dtype']} "
                f"{info['features'][name]['shape']}")
        info_path.write_text(json.dumps(info, indent=4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
