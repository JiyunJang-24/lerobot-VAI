#!/usr/bin/env python
"""Offline readiness check for the visual-robust auxiliary dataset.

Exits 0 only if every `<task>/lerobot` tree is in the exact shape
`lerobot_train_with_visual_robust.py` needs, so a caller can skip the download and setup steps
entirely rather than re-running them "just in case".

That distinction matters: re-running the download over an already-converted tree restores the
original per-episode files into it (and reverts meta/info.json to v2.1), leaving data/ holding both
the packed `file-000.parquet` and 108 `episode_*.parquet`. LeRobot globs `data/*/*.parquet`, so it
would load both and train on a corrupted dataset without raising.

Checks, per task:
  - codebase_version == v3.0
  - >= 2 `observation.image.*` features (fewer and the contrastive loss silently returns None)
  - a matching `videos/<key>/` dir holding at least one mp4 for each of those features
  - no leftover `episode_*.parquet` next to the packed data files
  - no leftover v2.1 metadata (meta/episodes.jsonl, meta/tasks.jsonl)

Touches only the local filesystem -- no network, so it cannot be rate limited.

Usage:
    python tools/check_visual_robust_ready.py [--root dataset_git/visual_robust_robocasa_x]
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "dataset_git" / "visual_robust_robocasa_x"
MIN_FRONT_VIEWS = 2


def log(msg: str) -> None:
    print(f"[check_visual_robust_ready] {msg}", flush=True)


def problems_for(root: Path) -> list[str]:
    bad = []
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return ["no meta/info.json"]

    info = json.loads(info_path.read_text())
    version = info.get("codebase_version")
    if version != "v3.0":
        bad.append(f"codebase_version={version!r}, want 'v3.0'")

    front = sorted(
        k for k, ft in info.get("features", {}).items()
        if ft.get("dtype") == "video" and k.startswith("observation.image.")
    )
    if len(front) < MIN_FRONT_VIEWS:
        bad.append(f"{len(front)} observation.image.* feature(s), need >= {MIN_FRONT_VIEWS}")

    for key in front:
        vids = list((root / "videos" / key).glob("**/*.mp4"))
        if not vids:
            bad.append(f"no mp4 under videos/{key}/")

    # These two only mean "contaminated" on a tree that already claims v3.0. On a freshly downloaded
    # v2.1 tree, per-episode parquet and episodes.jsonl/tasks.jsonl are simply the normal layout --
    # reporting them as damage there would send someone chasing a corruption that isn't real.
    if version == "v3.0":
        strays = list((root / "data").glob("**/episode_*.parquet"))
        if strays:
            bad.append(f"{len(strays)} stray episode_*.parquet in data/ (re-download contaminated it)")

        for legacy in ("episodes.jsonl", "tasks.jsonl"):
            if (root / "meta" / legacy).is_file():
                bad.append(f"leftover v2.1 meta/{legacy}")

    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--min-front-views",
        type=int,
        default=2,
        help="Fail unless each task has at least this many observation.image.* features. Below 2 the "
        "contrastive loss silently returns None; set it to the embodiment count (4 with Jaco) to "
        "also catch a tree that was prepared from an older, smaller export.",
    )
    args = parser.parse_args()
    global MIN_FRONT_VIEWS
    MIN_FRONT_VIEWS = args.min_front_views

    # Discover the trees instead of hard-coding names: the task directories differ per export
    # (PickPlaceCounterToSink... in the background-variation trees, IIWAOmron_PnPCounterToSink... in
    # new_barx), and a fixed list reports a perfectly good dataset as MISSING.
    roots = sorted(p.parent.parent for p in args.root.glob("*/lerobot/meta/info.json"))
    if not roots:
        log(f"no <task>/lerobot datasets found under {args.root}")
        return 1

    ok = True
    for root in roots:
        task = root.parent.name
        bad = problems_for(root)
        if bad:
            log(f"{task}: NOT READY -- {'; '.join(bad)}")
            ok = False
        else:
            info = json.loads((root / "meta" / "info.json").read_text())
            n_front = sum(
                1 for k, ft in info["features"].items()
                if ft.get("dtype") == "video" and k.startswith("observation.image.")
            )
            log(f"{task}: ready (v3.0, {info['total_episodes']} eps, {n_front} front views)")

    log("ALL READY" if ok else "NOT READY")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
