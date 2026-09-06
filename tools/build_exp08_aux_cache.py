#!/usr/bin/env python
"""Decode an exp08 auxiliary domain into /dev/shm, reading the parquet directly.

Going through LeRobotDataset managed ~500 frames/minute here, which is roughly two hours for the
two domains, because every frame pays the datasets-library formatting cost. The images are PNG
bytes in a parquet column, so decoding them straight with PIL skips all of it.

One cache per DOMAIN: the state and pixel settings read the same frames and differ only in which
answer column they use.

    python tools/build_exp08_aux_cache.py --domain real
"""

import argparse
import glob
import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.scripts.train_libero_exp08 import ACTION_TREES, LIBERO, SYNTH_TREES  # noqa: E402

COLUMNS = ["observation.image", "observation.eef_base_rel", "eef_pixel", "episode_index"]


def log(msg: str) -> None:
    print(f"[aux_cache] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True, choices=["real", "synth"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = Path("/dev/shm") / f"exp08_aux_{args.domain}.pt"
    if out.exists():
        log(f"already built: {out}")
        return 0
    root = LIBERO / ("action" if args.domain == "real" else "synthetic_randik")
    trees = ACTION_TREES if args.domain == "real" else SYNTH_TREES

    files = []
    for tree in trees:
        files += sorted(glob.glob(str(root / tree / "data" / "**" / "*.parquet"), recursive=True))
    log(f"{args.domain}: {len(trees)} trees, {len(files)} parquet")

    from PIL import Image

    images, base_rel, pixels, episodes = [], [], [], []
    offset = 0
    for path in files:
        table = pq.read_table(path, columns=COLUMNS)
        raw = table["observation.image"].to_pylist()
        for cell in raw:
            arr = np.array(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
            images.append(torch.from_numpy(arr).permute(2, 0, 1))
        base_rel += table["observation.eef_base_rel"].to_pylist()
        pixels += table["eef_pixel"].to_pylist()
        # episode_index restarts per tree, so offset it to keep pairs from crossing trees
        eps = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        episodes.append(eps + offset)
        if path.endswith(sorted(glob.glob(str(Path(path).parent / "*.parquet")))[-1]):
            offset += int(eps.max()) + 1
        if len(images) % 10000 < 500:
            log(f"  {len(images)}")
    blob = {
        "images": torch.stack(images),
        "base_rel": np.asarray(base_rel, dtype=np.float64),
        "pixels": np.asarray(pixels, dtype=np.float64),
        "episodes": np.concatenate(episodes),
    }
    torch.save(blob, out)
    log(f"wrote {out} ({out.stat().st_size / 1e9:.1f} GB), {len(images)} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
