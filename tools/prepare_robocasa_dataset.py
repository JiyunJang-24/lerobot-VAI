#!/usr/bin/env python
"""Build two small, training-ready v3.0 LeRobot datasets out of the (already v3.0) RoboCasa "human"
+ "mg" (MimicGen) demonstrations for one atomic task, ready to be trained on together via this
project's normal multi-dataset training path (`--dataset.repo_id=[human,mg]`), the same way
`train_smolVLA_scaling.sh` consumes datasets produced by `convert_v21_to_v30_multiple.sh`.

Why this exists
----------------
`convert_robocasa_to_v30.sh` handles getting RoboCasa's raw exports to codebase_version "v3.0"
in place (v2.0/v2.1 -> v3.0, no cameras dropped, no episode cap). This script is the next step:
given two already-v3.0 sources, pick only the episodes/cameras actually needed for a training run
and copy that (smaller) subset into this repo's `dataset_git/`, without touching the sources.

What it does (idempotent -- safe to re-run; skipped per-repo if its output already exists with the
requested episode count)
------------------------------------------------------------------------------------------------
1. Requires both --source-human and --source-mg to already be codebase_version "v3.0" -- errors out
   with a pointer to `./convert_robocasa_to_v30.sh` otherwise (this script does no format
   conversion).
2. "human" (all its episodes always count toward --total-episodes) and "mg" (only the first N
   episodes needed to reach --total-episodes) are each, in this repo's `dataset_git/`:
     a. subset to the needed episode range via `lerobot.datasets.dataset_tools.split_dataset`
        (skipped if the whole source is already within the episode cap -- e.g. human is usually
        smaller than --total-episodes, so no video re-encoding happens for it at all)
     b. stripped down to --cameras via `dataset_tools.remove_feature` (default keeps
        robot0_agentview_right + robot0_eye_in_hand, drops robot0_agentview_left)
   Both dataset_tools calls operate on a single source dataset at a time (no cross-dataset video
   concatenation), which past debugging in this project found to be the safe path -- see this
   script's git history / claude.txt for the aggregate_datasets video-corruption bug that motivated
   avoiding any physical merge step.

No merging/aggregation step: `MultiLeRobotDataset` (used by the bracketed `--dataset.repo_id=[human,mg]`
CLI syntax) only requires overlapping feature keys, not identical schemas -- it silently drops
non-common keys rather than erroring.

Usage:
    python tools/prepare_robocasa_dataset.py \\
        --source-human /root/Desktop/workspace/jiyun/robocasa/datasets/v1.0/pretrain/atomic/TurnOnSinkFaucet/20250819/lerobot \\
        --source-mg /root/Desktop/workspace/jiyun/robocasa/datasets/v1.0/pretrain/atomic/TurnOnSinkFaucet/20250819/mg/demo/2025-08-21-12-24-03/lerobot \\
        --output-root dataset_git/robocasa_turnonsinkfaucet \\
        --total-episodes 3000
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.datasets.dataset_tools import remove_feature, split_dataset  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.utils import load_info  # noqa: E402

V30 = "v3.0"
DEFAULT_CAMERAS = [
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
]


def log(msg: str) -> None:
    print(f"[prepare_robocasa_dataset] {msg}", flush=True)


def require_v30(path: Path, label: str) -> None:
    version = load_info(path).get("codebase_version", "unknown")
    if version != V30:
        raise SystemExit(
            f"{label} at {path} is codebase_version={version!r}, not {V30!r}. "
            f"Run ./convert_robocasa_to_v30.sh first (it converts in place)."
        )


def build_subset(source: Path, dest: Path, tag: str, cameras_to_keep: list[str], max_episodes: int | None, force: bool) -> int:
    """Copy `source` (already v3.0) into `dest`, capped to `max_episodes` and stripped down to
    `cameras_to_keep`. Returns the number of episodes in the result. Skips entirely if `dest`
    already exists with the requested episode count (unless force)."""
    if dest.exists():
        existing_n = json.loads((dest / "meta/info.json").read_text())["total_episodes"]
        source_n = json.loads((source / "meta/info.json").read_text())["total_episodes"]
        wanted_n = min(max_episodes, source_n) if max_episodes is not None else source_n
        if not force:
            if existing_n != wanted_n:
                raise SystemExit(
                    f"{dest} already exists with {existing_n} episodes, but {wanted_n} are needed "
                    f"for this run. Use --force to rebuild it, or pick a different --output-root."
                )
            log(f"{dest} already has the requested {existing_n} episodes, skipping (use --force to redo)")
            return existing_n
        shutil.rmtree(dest)

    src_ds = LeRobotDataset(repo_id=f"robocasa_{tag}_src", root=source)
    n_total = src_ds.meta.total_episodes
    n_keep = min(max_episodes, n_total) if max_episodes is not None else n_total

    work_ds = src_ds
    split_tmp = dest.parent / f".{tag}_split_tmp"
    if n_keep < n_total:
        log(f"{tag}: selecting first {n_keep}/{n_total} episodes")
        if split_tmp.exists():
            shutil.rmtree(split_tmp)
        work_ds = split_dataset(src_ds, {"kept": list(range(n_keep))}, output_dir=split_tmp)["kept"]

    drop_cameras = [c for c in work_ds.meta.video_keys if c not in cameras_to_keep]
    log(f"{tag}: keeping {n_keep} episodes, dropping cameras {drop_cameras or '(none)'}")
    if drop_cameras:
        remove_feature(work_ds, feature_names=drop_cameras, repo_id=f"robocasa_{tag}", output_dir=dest)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(work_ds.root, dest, dirs_exist_ok=True)

    if split_tmp.exists():
        shutil.rmtree(split_tmp)

    log(f"{tag}: done -> {dest} ({n_keep} episodes)")
    return n_keep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-human", type=Path, required=True)
    parser.add_argument("--source-mg", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--total-episodes", type=int, default=3000)
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=DEFAULT_CAMERAS,
        help="Camera feature keys to KEEP (all others dropped from both datasets)",
    )
    parser.add_argument("--force", action="store_true", help="Redo the subset step even if it already exists")
    args = parser.parse_args()

    require_v30(args.source_human, "--source-human")
    require_v30(args.source_mg, "--source-mg")

    output_root = args.output_root.resolve()
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    human_episodes = build_subset(
        args.source_human, raw_dir / "human", "human", args.cameras, max_episodes=None, force=args.force
    )

    mg_needed = args.total_episodes - human_episodes
    if mg_needed <= 0:
        raise SystemExit(
            f"--total-episodes={args.total_episodes} is <= human's own episode count "
            f"({human_episodes}); nothing would be taken from mg. Raise --total-episodes."
        )
    mg_source_total = json.loads((args.source_mg / "meta/info.json").read_text())["total_episodes"]
    if mg_needed > mg_source_total:
        raise SystemExit(
            f"Need {mg_needed} mg episodes ({args.total_episodes} total - {human_episodes} human), "
            f"but mg source only has {mg_source_total}."
        )

    mg_episodes = build_subset(
        args.source_mg, raw_dir / "mg", "mg", args.cameras, max_episodes=mg_needed, force=args.force
    )

    log(
        f"done: human={human_episodes} + mg={mg_episodes} = {human_episodes + mg_episodes} episodes "
        f"ready under {raw_dir} (repo_ids: human, mg)"
    )


if __name__ == "__main__":
    main()
