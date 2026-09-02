#!/usr/bin/env python
"""Add the missing per-feature `count` to meta/episodes_stats.jsonl.

The eef_pairs exports write min/max/mean/std for every feature but omit `count`, which
lerobot.datasets.compute_stats.aggregate_stats needs -- it weights each episode's mean and variance
by its sample count when it pools them. Without the field, converting to v3.0 dies with
`KeyError: 'count'` before any data is touched.

`count` is not something to guess: it is the number of samples that produced that episode's
statistics, i.e. the episode's length, so it is read back from the parquet rather than assumed.
For image/video features lerobot counts one sample per frame as well, so the same length applies.

Idempotent -- a file that already has `count` everywhere is left alone.

    python tools/add_episode_stats_count.py /dataset/jiyun/dataset_git/eef_pairs/56combo_48_bg12_closed
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd


def log(msg: str) -> None:
    print(f"[add_count] {msg}", flush=True)


def episode_lengths(root: Path) -> dict[int, int]:
    lengths: dict[int, int] = {}
    for path in sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True)):
        frame = pd.read_parquet(path, columns=["episode_index"])
        for episode, count in frame.episode_index.value_counts().items():
            lengths[int(episode)] = lengths.get(int(episode), 0) + int(count)
    return lengths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    args = ap.parse_args()

    stats_path = args.root / "meta" / "episodes_stats.jsonl"
    if not stats_path.exists():
        log(f"no {stats_path}")
        return 1

    lengths = episode_lengths(args.root)
    if not lengths:
        log(f"no parquet under {args.root / 'data'}")
        return 1

    rows, added, already = [], 0, 0
    for line in stats_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        episode = int(record["episode_index"])
        length = lengths.get(episode)
        if length is None:
            log(f"episode {episode} has stats but no rows in the parquet -- leaving it alone")
            rows.append(record)
            continue
        for feature, stats in record["stats"].items():
            if "count" in stats:
                already += 1
                continue
            # lerobot stores count as a list so it broadcasts against the per-dimension stats.
            stats["count"] = [length]
            added += 1
        rows.append(record)

    if added == 0:
        log(f"{args.root.name}: every feature already has count ({already}) -- nothing to do")
        return 0

    stats_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    log(f"{args.root.name}: added count to {added} feature entries across {len(rows)} episodes "
        f"(episode lengths {min(lengths.values())}..{max(lengths.values())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
