#!/usr/bin/env python
"""Rename a v2.x LeRobot dataset's per-camera video directories to the full feature key.

`convert_dataset_v21_to_v30.py` finds a camera's mp4s with

    (root / "videos").glob(f"*/{video_key}/*.mp4")

where `video_key` is the full feature name, e.g. `observation.images.robot0_agentview_right`. Some
RoboCasa exports (the IIWA/UR5e trees in ChiefJang/robocasa_x_atomic_ur5e_iiwa, for instance) name
those directories with the bare camera name instead -- `videos/chunk-000/robot0_agentview_right/` --
so the glob matches nothing, every camera reports 0 episodes, and the conversion dies late with

    ValueError: Number of episodes is not the same ({1000, 0}).

after having already written a partial `<name>_v30` tree. This script renames such directories to
the prefixed form so the converter can see them. It only moves directories -- no video is read,
re-encoded, or rewritten.

Idempotent and safe to run on already-correct datasets (including v3.0 ones, which are skipped
outright since their layout is `videos/<video_key>/chunk-XXX/file-YYY.mp4` and needs no fixing).

Usage:
    python tools/normalize_video_dirs.py --root path/to/dataset [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path


def log(msg: str) -> None:
    print(f"[normalize_video_dirs] {msg}", flush=True)


def video_keys(info: dict) -> list[str]:
    return [k for k, ft in info.get("features", {}).items() if ft.get("dtype") == "video"]


def normalize(root: Path, dry_run: bool) -> int:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        log(f"no meta/info.json under {root}; nothing to do")
        return 0

    info = json.loads(info_path.read_text())
    version = info.get("codebase_version", "unknown")
    if version == "v3.0":
        log(f"{root} is already v3.0 (packed video layout); skipping")
        return 0

    videos_dir = root / "videos"
    if not videos_dir.is_dir():
        log(f"{root} has no videos/ dir; skipping")
        return 0

    keys = video_keys(info)
    if not keys:
        log(f"{root} declares no video features; skipping")
        return 0

    renamed = 0
    for key in keys:
        if "." not in key:
            continue
        # Exports abbreviate the dir name in more than one way: RoboCasa's cross-embodiment trees
        # drop the whole "observation.images." prefix and keep one component
        # ("robot0_agentview_right"), while the visual-robust trees keep the embodiment too
        # ("IIWAOmron.robot0_agentview_right"). Try prefix-stripping first, then the last component.
        candidates = []
        for prefix in ("observation.images.", "observation.image.", "observation.wrist_image."):
            if key.startswith(prefix):
                candidates.append(key[len(prefix) :])
        candidates.append(key.rsplit(".", 1)[-1])

        for chunk_dir in sorted(videos_dir.glob("*")):
            if not chunk_dir.is_dir():
                continue
            dst = chunk_dir / key
            if dst.is_dir():
                continue  # already in the expected form
            src = next((chunk_dir / c for c in candidates if (chunk_dir / c).is_dir()), None)
            if src is None:
                continue
            if dry_run:
                log(f"would rename {src} -> {dst.name}")
            else:
                src.rename(dst)
                log(f"renamed {src.relative_to(root)} -> {dst.relative_to(root)}")
            renamed += 1

    if renamed == 0:
        log(f"{root}: video dirs already use full feature keys, nothing renamed")
    else:
        log(f"{root}: renamed {renamed} directory/ies")

    # Report what the converter will actually see, so a silent zero-match can't slip through again.
    for key in keys:
        n = len(list(videos_dir.glob(f"*/{key}/*.mp4")))
        log(f"  {key}: {n} mp4(s) visible to the converter")
        if n == 0:
            log(f"  WARNING: {key} still matches no mp4 under {videos_dir}/*/{key}/")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Dataset root (the dir holding meta/)")
    parser.add_argument("--dry-run", action="store_true", help="Report renames without doing them")
    args = parser.parse_args()

    if not args.root.is_dir():
        log(f"root not found: {args.root}")
        return 1
    return normalize(args.root.resolve(), args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
