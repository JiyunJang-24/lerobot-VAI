#!/usr/bin/env python
"""Decode an eef_pairs subset combination once into /dev/shm, in parallel.

Decoding inside the training loop drove load average past 500 with the GPUs at 0% (CLAUDE.md 9.3),
so the cache is not optional. A single process decodes ~11 frames/s, which is 6 hours for the
144-pose subset, so this shards the work.

Each worker writes its own shard and the parent concatenates, rather than sharing one tensor:
shards are restartable, and a worker dying leaves the finished ones on disk.

    python tools/build_motion_cache.py --subsets 56combo_144_bg12_closed --workers 12
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.scripts.motion_data import EEF_ROOT, build_table, cache_path  # noqa: E402

IMAGE_KEY = "observation.images.agentview_right"


def log(msg: str) -> None:
    print(f"[cache] {msg}", flush=True)


def worker(args):
    """One subset, one contiguous slice of it.

    Opening the per-subset LeRobotDataset directly rather than MultiLeRobotDataset: the multi
    wrapper drops keys that are not present in EVERY member, and returns None for them, which
    surfaces only as 'NoneType is not subscriptable' inside a pool worker.
    """
    shard_id, subset, local_rows, positions, shard_path = args
    if Path(shard_path).exists():
        return shard_path, positions
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(f"eef_pairs/{subset}", root=EEF_ROOT / subset)
    out = torch.empty(len(local_rows), 3, 180, 320, dtype=torch.uint8)
    for i, row in enumerate(local_rows):
        image = dataset[int(row)][IMAGE_KEY]
        image = image[-1] if image.ndim == 4 else image
        out[i] = (image * 255).round().clamp(0, 255).to(torch.uint8)
        if shard_id == 0 and (i + 1) % 2000 == 0:
            print(f"[cache] shard 0: {i + 1}/{len(local_rows)}", flush=True)
    torch.save(out, shard_path)
    return shard_path, positions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subsets", required=True, help="comma-separated")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    subsets = args.subsets.split(",")
    target = cache_path(subsets)
    if target.exists():
        log(f"already built: {target}")
        return 0

    table = build_table(subsets)
    rows = table["row"].to_numpy()
    log(f"{len(rows)} frames across {len(subsets)} subset(s) -> {target}")

    # build_table's `row` is a global index across the concatenated subsets, so recover the
    # per-subset local index by subtracting that subset's offset.
    tmp = Path("/dev/shm") / f".shards_{target.stem}"
    tmp.mkdir(exist_ok=True)
    jobs, shard_id = [], 0
    for subset in subsets:
        mask = (table["subset"] == subset).to_numpy()
        positions = np.flatnonzero(mask)
        local = rows[positions] - rows[positions].min()
        per = max(1, len(positions) // max(1, args.workers // len(subsets) or 1))
        for start in range(0, len(positions), per):
            sl = slice(start, start + per)
            jobs.append((shard_id, subset, local[sl], positions[sl],
                         str(tmp / f"shard_{shard_id:03d}.pt")))
            shard_id += 1
    log(f"{len(jobs)} shards")

    import multiprocessing as mp

    with mp.get_context("spawn").Pool(args.workers) as pool:
        done = pool.map(worker, jobs)
    log("shards done, assembling ...")
    out = torch.empty(len(rows), 3, 180, 320, dtype=torch.uint8)
    for path, positions in done:
        out[positions] = torch.load(path)
    torch.save(out, target)
    for path, _ in done:
        os.unlink(path)
    tmp.rmdir()
    log(f"wrote {target} ({target.stat().st_size / 1e9:.0f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
