#!/usr/bin/env python
"""Fetch the ChiefJang/visual_robust_robocasa_x `<task>/lerobot` trees.

Only the `lerobot/` subtrees are pulled (~0.24 GB); the sibling `examples/` dirs hold the raw
per-embodiment hdf5/mp4 renders and are not used for training.

The repo is ~3.2k small files, which trips the Hub's "1000 api requests per 5 minutes" limit at any
real concurrency, so this keeps `max_workers` low and retries the whole snapshot on HTTP 429 with a
backoff. `snapshot_download` skips files already on disk, so each retry resumes rather than restarts.

Usage:
    python tools/download_visual_robust_x.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError

REPO = "ChiefJang/visual_robust_robocasa_x"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEST = REPO_ROOT / "dataset_git" / "visual_robust_robocasa_x"
MAX_ATTEMPTS = 12
BACKOFF_S = 90


TASKS = ("PickPlaceCounterToSink", "PickPlaceCounterToStove", "PickPlaceSinkToCounter")
# Each task tree is 108 episodes x N cameras of video plus 108 parquet shards. N was 9 (Panda /
# UR5e / IIWA x 3 views) and became 12 when JacoOmron was added upstream, so it is a parameter.
EXPECTED_CAMERAS = 12
EXPECTED_PARQUET = 108


def incomplete() -> list[str]:
    bad = []
    for t in TASKS:
        root = DEST / t / "lerobot"
        info_path = root / "meta" / "info.json"
        if not info_path.is_file():
            bad.append(f"{t}(no meta)")
            continue

        # Once setup_visual_robust_x_dataset.sh has run, the tree is v3.0: per-episode mp4/parquet
        # have been packed into a handful of files, so the raw-file counts below can never be met
        # again. Re-downloading then would refill the converted tree with the original
        # episode_*.parquet, which LeRobot's data/*/*.parquet glob would happily load alongside the
        # packed file-000.parquet -- silently corrupting the dataset. Treat v3.0 as done.
        if json.loads(info_path.read_text()).get("codebase_version") == "v3.0":
            continue

        expected_mp4 = 108 * EXPECTED_CAMERAS
        n_mp4 = len(list((root / "videos").glob("**/*.mp4")))
        n_pq = len(list((root / "data").glob("**/*.parquet")))
        if n_mp4 < expected_mp4 or n_pq < EXPECTED_PARQUET:
            bad.append(f"{t}({n_mp4}/{expected_mp4} mp4, {n_pq}/{EXPECTED_PARQUET} pq)")
    return bad


def main() -> int:
    global DEST, EXPECTED_CAMERAS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=DEST, help="Where to download the <task>/lerobot trees")
    parser.add_argument("--cameras", type=int, default=EXPECTED_CAMERAS, help="Cameras per episode (9 pre-Jaco, 12 with Jaco)")
    args = parser.parse_args()
    DEST = args.dest.resolve()
    EXPECTED_CAMERAS = args.cameras

    DEST.mkdir(parents=True, exist_ok=True)
    missing = incomplete()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if not missing:
            break
        try:
            snapshot_download(
                REPO,
                repo_type="dataset",
                local_dir=str(DEST),
                allow_patterns=["*/lerobot/*"],
                max_workers=2,
            )
        except HfHubHTTPError as exc:
            if "429" not in str(exc) or attempt == MAX_ATTEMPTS:
                print(f"DOWNLOAD_FAILED: {exc}")
                return 1

        # snapshot_download swallows a rate-limited repo-info call -- it logs the 429 and just
        # returns the existing local_dir rather than raising -- so a clean return is not proof the
        # fetch finished. Re-check what is actually on disk and retry until it is all there.
        missing = incomplete()
        if missing and attempt < MAX_ATTEMPTS:
            print(
                f"incomplete after attempt {attempt}/{MAX_ATTEMPTS}: {', '.join(missing)}; "
                f"sleeping {BACKOFF_S}s and resuming",
                flush=True,
            )
            time.sleep(BACKOFF_S)

    if missing:
        print(f"DOWNLOAD_INCOMPLETE: {', '.join(missing)}")
        return 1

    for t in sorted(p.parent.parent for p in DEST.glob("*/lerobot/meta/info.json")):
        n_mp4 = len(list((t / "videos").glob("**/*.mp4")))
        n_pq = len(list((t / "data").glob("**/*.parquet")))
        print(f"  {t.parent.name:26s} {n_mp4:5d} mp4  {n_pq:5d} parquet")
    print("DOWNLOAD_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
