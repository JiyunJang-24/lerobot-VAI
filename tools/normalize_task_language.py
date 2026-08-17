#!/usr/bin/env python
"""Rewrite a dataset's task strings into one consistent surface style.

The RoboCasa PnP exports phrase their instructions differently per embodiment:

    panda : "Pick the onion from the plate and place it in the pan."   (capitalised, full stop)
    iiwa  : "pick the can from the counter and place it in the sink"   (lower-case, no full stop)
    ur5e  : "pick the cucumber from the sink and place it on the plate located on the counter"

Since each robot also performs a *different* task, that stylistic split is perfectly correlated with
the embodiment -- so the language encoder can tell the robots apart from capitalisation alone rather
than from what the sentence actually asks for. This script removes that shortcut by putting every
instruction in the same style ("panda": leading capital + trailing full stop).

Only the surface form changes; wording is untouched. Bare task identifiers (single tokens with no
spaces, e.g. "PickPlaceCounterToStove") are left alone -- they are not natural-language
instructions.

What gets rewritten:
  - meta/tasks.parquet          -- the index, which is what LeRobotDataset returns as item["task"]
                                  (`self.meta.tasks.iloc[task_idx].name`) and therefore the only
                                  copy the policy actually sees.
  - meta/episodes/**/*.parquet  -- the per-episode `tasks` list column, kept in sync so the dataset
                                  stays self-consistent for anything else that reads it.

Idempotent: running it twice is a no-op.

Usage:
    python tools/normalize_task_language.py --root path/to/dataset [--dry-run]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def log(msg: str) -> None:
    print(f"[normalize_task_language] {msg}", flush=True)


def to_panda_style(text: str) -> str:
    """Leading capital + trailing full stop, for natural-language instructions only."""
    if " " not in text.strip():
        return text  # bare identifier like "PickPlaceCounterToStove"
    out = text.strip()
    out = out[0].upper() + out[1:]
    if not out.endswith("."):
        out += "."
    return out


def normalize(root: Path, dry_run: bool) -> int:
    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        log(f"{root}: no meta/tasks.parquet, skipping")
        return 0

    tasks = pd.read_parquet(tasks_path)
    old = list(tasks.index)
    new = [to_panda_style(t) for t in old]

    if len(set(new)) != len(new):
        dupes = {t for t in new if new.count(t) > 1}
        log(f"{root}: ERROR -- restyling collapses distinct tasks into {dupes}; refusing to write")
        return 1

    changed = [(o, n) for o, n in zip(old, new) if o != n]
    if not changed:
        log(f"{root}: all {len(old)} task strings already in panda style")
    else:
        log(f"{root}: restyling {len(changed)}/{len(old)} task strings")
        for o, n in changed[:3]:
            log(f"    {o!r}\n      -> {n!r}")
        if len(changed) > 3:
            log(f"    ... and {len(changed) - 3} more")

    if dry_run:
        return 0

    # Episode metadata first, and restyled directly rather than through a diff of tasks.parquet, so
    # that an interrupted or partially-applied earlier run still converges on a rerun (a diff-based
    # remap would see an already-restyled tasks.parquet, report "nothing to do", and leave the
    # per-episode copy stale forever).
    def restyle_list(lst):
        if lst is None:
            return lst
        vals = [to_panda_style(t) for t in lst]
        return np.array(vals, dtype=object) if isinstance(lst, np.ndarray) else type(lst)(vals)

    ep_files = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    touched = 0
    for path in ep_files:
        df = pd.read_parquet(path)
        if "tasks" not in df.columns:
            continue
        before = [list(x) for x in df["tasks"] if x is not None]
        df["tasks"] = df["tasks"].map(restyle_list)
        after = [list(x) for x in df["tasks"] if x is not None]
        if before != after:
            df.to_parquet(path)
            touched += 1

    if changed:
        tasks.index = pd.Index(new, name=tasks.index.name)
        tasks.to_parquet(tasks_path)

    log(f"{root}: rewrote {len(changed)} task string(s) and {touched} episode metadata file(s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Dataset root (the dir holding meta/)")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    args = parser.parse_args()

    if not args.root.is_dir():
        log(f"root not found: {args.root}")
        return 1
    return normalize(args.root.resolve(), args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
