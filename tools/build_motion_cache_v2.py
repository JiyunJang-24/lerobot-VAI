#!/usr/bin/env python
"""Decode the redesigned export into /dev/shm: one tensor of (2, C, H, W) pairs.

Both frames of a pair live in the same slot, so a training step is one index rather than two
lookups that have to be kept consistent.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.scripts.motion_data_v2 import DEFAULT, HIGH, LOW, ROOT, build_table, cache_path  # noqa: E402


def log(msg: str) -> None:
    print(f"[cache_v2] {msg}", flush=True)


def worker(args):
    shard, subset, rows, keys, shape, path = args
    if Path(path).exists():
        return path, None
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(f"eef_pairs/{subset}", root=ROOT / subset)
    out = torch.empty(len(rows), 2, 3, *shape, dtype=torch.uint8)
    for i, row in enumerate(rows):
        sample = dataset[int(row)]
        for k, key in enumerate(keys):
            img = sample[key]
            img = img[-1] if img.ndim == 4 else img
            out[i, k] = (img * 255).round().clamp(0, 255).to(torch.uint8)
        if shard == 0 and (i + 1) % 2000 == 0:
            print(f"[cache_v2] shard 0: {i + 1}/{len(rows)}", flush=True)
    torch.save(out, path)
    return path, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", default=DEFAULT)
    ap.add_argument("--high", action="store_true", help="cache the 512x910 renders instead")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit-per-embodiment", type=int, default=0,
                    help="cache only N pairs per embodiment. The high-res stream is ~8x the bytes, "
                         "so it is normally cached on a subset.")
    args = ap.parse_args()

    target = cache_path(args.subset, args.high)
    if target.exists():
        log(f"already built: {target}")
        return 0
    keys = HIGH if args.high else LOW
    shape = (512, 910) if args.high else (180, 320)

    table = build_table(args.subset)
    if args.limit_per_embodiment:
        table = (table.groupby("embodiment", group_keys=False)
                 .apply(lambda g: g.head(args.limit_per_embodiment)).reset_index(drop=True))
        log(f"limited to {args.limit_per_embodiment}/embodiment -> {len(table)} pairs")
    rows = table["row"].to_numpy()
    gb = len(rows) * 2 * 3 * shape[0] * shape[1] / 1e9
    log(f"{len(rows)} pairs x 2 frames at {shape} -> {gb:.0f} GB into {target}")
    (table[["row", "cache_pos", "embodiment", "camera"]]
     .to_parquet(target.with_suffix(".index.parquet")))

    shards = np.array_split(np.arange(len(rows)), args.workers)
    tmp = Path("/dev/shm") / f".shards_{target.stem}"
    tmp.mkdir(exist_ok=True)
    jobs = [(i, args.subset, rows[sh], keys, shape, str(tmp / f"s{i:03d}.pt"))
            for i, sh in enumerate(shards) if len(sh)]

    import multiprocessing as mp

    with mp.get_context("spawn").Pool(args.workers) as pool:
        done = pool.map(worker, jobs)
    log("assembling ...")
    out = torch.cat([torch.load(p) for p, _ in done])
    torch.save(out, target)
    for p, _ in done:
        os.unlink(p)
    tmp.rmdir()
    log(f"wrote {target} ({target.stat().st_size / 1e9:.0f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
