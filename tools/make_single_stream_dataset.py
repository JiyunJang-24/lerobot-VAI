#!/usr/bin/env python
"""Derive a single-image dataset from barx3_segsplit by keeping exactly one of its two streams.

Used for the one-image ablations:

    scene_only   observation.images.scene   the kitchen with the robot blacked out
    robot_only   observation.images.robot   the robot on black

No video is re-encoded -- the mp4 is hard-linked, and only the metadata is rewritten. That keeps
the pixels bit-identical to what the two-image runs trained on, so any difference between the runs
is the missing stream and nothing else.

The unwanted stream is REMOVED FROM THE DATASET rather than filtered at run time, for the same
reason the wrist camera had to be: --dataset.use_wrist_cam drops `observation.wrist_image` and keys
containing "wrist", so any other key sails through it and trains anyway.

    python tools/make_single_stream_dataset.py --keep observation.images.scene \
        --out dataset_git/barx3_scene_only/raw
"""

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import pandas as pd

SRC = Path("dataset_git/barx3_segsplit/raw")
SUBSETS = ["panda_mg", "iiwa", "ur5e"]


def log(m: str) -> None:
    print(f"[single] {m}", flush=True)


def build(name: str, keep: str, out: Path) -> None:
    src, dst = SRC / name, out / name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    shutil.copytree(src / "data", dst / "data")
    shutil.copytree(src / "meta", dst / "meta")

    info = json.loads((src / "meta" / "info.json").read_text())
    drop = [k for k in info["features"] if "images" in k and k != keep]
    if keep not in info["features"]:
        raise SystemExit(f"{name}: {keep} not in {list(info['features'])}")

    # hard-link the kept video; the bytes are shared with the two-image dataset
    for f in glob.glob(str(src / "videos" / keep / "**" / "*.mp4"), recursive=True):
        rel = Path(f).relative_to(src)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        os.link(f, dst / rel)

    for k in drop:
        info["features"].pop(k)
    info["total_videos"] = int(info["total_episodes"])
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    for f in glob.glob(str(dst / "meta/episodes/*/*.parquet")):
        t = pd.read_parquet(f)
        t = t[[c for c in t.columns
               if not any(c.startswith(f"{pre}/{d}/") for d in drop for pre in ("videos", "stats"))]]
        t.to_parquet(f, index=False)

    sp = dst / "meta" / "stats.json"
    if sp.exists():
        st = json.loads(sp.read_text())
        for k in drop:
            st.pop(k, None)
        sp.write_text(json.dumps(st, indent=4))
    log(f"{name}: kept {keep}, dropped {drop}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for s in SUBSETS:
        build(s, args.keep, args.out)
    log(f"done -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
