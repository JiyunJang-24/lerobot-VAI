#!/usr/bin/env python
"""Unwrap length-1 list columns that a dataset's info.json declares as scalars.

LeRobot builds the HF `features` schema straight from `meta/info.json`, so a feature declared with
`shape: [1]` (e.g. `next.reward`, `next.done`, `annotation.human.*`) becomes a scalar
`Value("float32"/"bool"/"int64")`. Loading then casts the parquet to that schema.

Some RoboCasa exports (the IIWA/UR5e trees in ChiefJang/robocasa_x_atomic_ur5e_iiwa) instead store
those columns as length-1 *lists*, and `convert_dataset_v21_to_v30.py` copies the data through
unchanged. The mismatch only surfaces when something actually opens the dataset:

    TypeError: Couldn't cast array of type list<element: int64> to int64
    datasets.exceptions.DatasetGenerationError: An error occurred while generating the dataset

which is what makes `tools/fix_episode_file_index.py` (and later the training dataloader) fail on an
otherwise correctly converted v3.0 tree. Datasets that already store these as scalars -- the Panda
trees in the same repo, and everything produced by the older conversion path -- are untouched.

Only columns that are BOTH declared `shape: [1]` in info.json AND physically stored as a list are
rewritten, and each is verified to contain no list longer than 1 element before being flattened, so
no real per-frame data can be silently dropped.

Usage:
    python tools/flatten_singleton_columns.py --root path/to/dataset [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def log(msg: str) -> None:
    print(f"[flatten_singleton_columns] {msg}", flush=True)


def singleton_features(info: dict) -> set[str]:
    return {k for k, ft in info.get("features", {}).items() if ft.get("shape") == [1]}


def flatten_file(path: Path, targets: set[str], dry_run: bool) -> list[str]:
    table = pq.read_table(path)
    changed = []

    for name in table.schema.names:
        if name not in targets:
            continue
        col = table.column(name)
        if not pa.types.is_list(col.type) and not pa.types.is_large_list(col.type):
            continue

        # Refuse to touch anything that is not uniformly a 1-element list -- flattening those would
        # change the row count and silently corrupt the dataset.
        lengths = pc.list_value_length(col)
        distinct = pc.unique(lengths.combine_chunks()).to_pylist()
        distinct = [d for d in distinct if d is not None]
        if distinct != [1]:
            raise SystemExit(
                f"{path}: column {name} has list lengths {sorted(distinct)}, expected all 1 -- "
                f"refusing to flatten."
            )

        changed.append(name)
        if not dry_run:
            flat = pc.list_flatten(col.combine_chunks())
            idx = table.schema.get_field_index(name)
            table = table.set_column(idx, pa.field(name, flat.type), flat)

    if changed and not dry_run:
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp)
        tmp.replace(path)

    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Dataset root (the dir holding meta/)")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing")
    args = parser.parse_args()

    root = args.root.resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        log(f"no meta/info.json under {root}")
        return 1

    targets = singleton_features(json.loads(info_path.read_text()))
    if not targets:
        log(f"{root}: no shape-[1] features declared; nothing to do")
        return 0

    paths = sorted((root / "data").glob("*/*.parquet"))
    if not paths:
        log(f"{root}: no parquet files under data/")
        return 1

    total = 0
    cols_seen: set[str] = set()
    for path in paths:
        changed = flatten_file(path, targets, args.dry_run)
        if changed:
            total += 1
            cols_seen.update(changed)

    verb = "would flatten" if args.dry_run else "flattened"
    if total:
        log(f"{root}: {verb} {sorted(cols_seen)} in {total}/{len(paths)} parquet file(s)")
    else:
        log(f"{root}: all shape-[1] columns already scalar in {len(paths)} parquet file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
