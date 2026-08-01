#!/usr/bin/env python
"""Build a training-ready set of v3.0 LeRobot dataset subsets that combine multiple RoboCasa
embodiments for one smolVLA run (`train_smolVLA_robocasa_x.sh`):

  - panda_human: ALL episodes of the atomic-task "human" teleop demos (Panda arm).
  - panda_mg:    just enough "mg" (MimicGen, Panda arm) episodes so that
                 panda_human + panda_mg == --panda-total-episodes.
  - iiwa:        ALL episodes of the IIWA cross-embodiment dataset.
  - ur5e:        ALL episodes of the UR5e cross-embodiment dataset.

This is the same idea as `tools/prepare_robocasa_dataset.py` (which only handles the
human+mg Panda pair), generalized to also pull in the IIWA/UR5e cross-embodiment sources
untouched (full episode count) alongside the capped Panda pair. All sources must already be
codebase_version "v3.0" (run `./convert_robocasa_to_v30.sh` first) -- this script only *subsets*
(episode cap + camera selection) via `lerobot.datasets.dataset_tools`, same convention as the
other train_*.sh scripts in this repo (each dataset_tools call touches a single source dataset,
never merging/concatenating across sources, which past debugging in this project found to be the
safe path -- see claude.txt for the aggregate_datasets video-corruption bug that motivated this).

Usage:
    python tools/prepare_robocasa_x_dataset.py \\
        --output-root dataset_git/robocasa_x \\
        --panda-total-episodes 1000
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

ROBOCASA_ROOT = Path("/root/Desktop/workspace/jiyun/robocasa/datasets")
DEFAULT_PANDA_HUMAN = ROBOCASA_ROOT / "v1.0/pretrain/atomic/TurnOnSinkFaucet/20250819/lerobot"
DEFAULT_PANDA_MG = (
    ROBOCASA_ROOT / "v1.0/pretrain/atomic/TurnOnSinkFaucet/20250819/mg/demo/2025-08-21-12-24-03/lerobot"
)
DEFAULT_IIWA = (
    ROBOCASA_ROOT
    / "cross_embodiment/lerobot_datasets/IIWAOmron_Robotiq85Gripper/TurnOnSinkFaucet/2026-07-24/lerobot"
)
DEFAULT_UR5E = (
    ROBOCASA_ROOT
    / "cross_embodiment/lerobot_datasets/UR5eOmron_Robotiq85Gripper/TurnOnSinkFaucet/2026-07-24/lerobot"
)


def log(msg: str) -> None:
    print(f"[prepare_robocasa_x_dataset] {msg}", flush=True)


def require_v30(path: Path, label: str) -> None:
    version = load_info(path).get("codebase_version", "unknown")
    if version != V30:
        raise SystemExit(
            f"{label} at {path} is codebase_version={version!r}, not {V30!r}. "
            f"Run ./convert_robocasa_to_v30.sh first (it converts in place)."
        )


def build_subset(
    source: Path, dest: Path, tag: str, cameras_to_keep: list[str], max_episodes: int | None, force: bool
) -> int:
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

    src_ds = LeRobotDataset(repo_id=f"robocasa_x_{tag}_src", root=source)
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
        remove_feature(work_ds, feature_names=drop_cameras, repo_id=f"robocasa_x_{tag}", output_dir=dest)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(work_ds.root, dest, dirs_exist_ok=True)

    if split_tmp.exists():
        shutil.rmtree(split_tmp)

    log(f"{tag}: done -> {dest} ({n_keep} episodes)")
    return n_keep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-panda-human", type=Path, default=DEFAULT_PANDA_HUMAN)
    parser.add_argument("--source-panda-mg", type=Path, default=DEFAULT_PANDA_MG)
    parser.add_argument("--source-iiwa", type=Path, default=DEFAULT_IIWA)
    parser.add_argument("--source-ur5e", type=Path, default=DEFAULT_UR5E)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--panda-total-episodes",
        type=int,
        default=1000,
        help="panda_human (all) + panda_mg (however many needed) should sum to this.",
    )
    parser.add_argument(
        "--iiwa-episodes",
        type=int,
        default=None,
        help="Keep only the first N iiwa episodes (default: all of them).",
    )
    parser.add_argument(
        "--ur5e-episodes",
        type=int,
        default=None,
        help="Keep only the first N ur5e episodes (default: all of them).",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=DEFAULT_CAMERAS,
        help="Camera feature keys to KEEP (all others dropped from every source)",
    )
    parser.add_argument("--force", action="store_true", help="Redo the subset step even if it already exists")
    args = parser.parse_args()

    require_v30(args.source_panda_human, "--source-panda-human")
    require_v30(args.source_panda_mg, "--source-panda-mg")
    require_v30(args.source_iiwa, "--source-iiwa")
    require_v30(args.source_ur5e, "--source-ur5e")

    output_root = args.output_root.resolve()
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    panda_human_episodes = build_subset(
        args.source_panda_human,
        raw_dir / "panda_human",
        "panda_human",
        args.cameras,
        max_episodes=None,
        force=args.force,
    )

    mg_needed = args.panda_total_episodes - panda_human_episodes
    if mg_needed <= 0:
        raise SystemExit(
            f"--panda-total-episodes={args.panda_total_episodes} is <= panda_human's own episode "
            f"count ({panda_human_episodes}); nothing would be taken from panda_mg. Raise "
            f"--panda-total-episodes."
        )
    mg_source_total = json.loads((args.source_panda_mg / "meta/info.json").read_text())["total_episodes"]
    if mg_needed > mg_source_total:
        raise SystemExit(
            f"Need {mg_needed} panda_mg episodes ({args.panda_total_episodes} total - "
            f"{panda_human_episodes} panda_human), but panda_mg source only has {mg_source_total}."
        )

    panda_mg_episodes = build_subset(
        args.source_panda_mg,
        raw_dir / "panda_mg",
        "panda_mg",
        args.cameras,
        max_episodes=mg_needed,
        force=args.force,
    )

    for label, requested, source in (
        ("--iiwa-episodes", args.iiwa_episodes, args.source_iiwa),
        ("--ur5e-episodes", args.ur5e_episodes, args.source_ur5e),
    ):
        if requested is None:
            continue
        available = json.loads((source / "meta/info.json").read_text())["total_episodes"]
        if requested > available:
            # Clamp rather than fail: these sources sit just under a round 1000 (a handful of
            # episodes drop out during conversion), and refusing to run over a ~1% shortfall would
            # just force the caller to look up exact per-robot counts. build_subset() takes the
            # min() itself, so this is only about saying so out loud.
            log(f"WARNING: {label}={requested} but {source} only has {available}; using all {available}.")

    iiwa_episodes = build_subset(
        args.source_iiwa,
        raw_dir / "iiwa",
        "iiwa",
        args.cameras,
        max_episodes=args.iiwa_episodes,
        force=args.force,
    )
    ur5e_episodes = build_subset(
        args.source_ur5e,
        raw_dir / "ur5e",
        "ur5e",
        args.cameras,
        max_episodes=args.ur5e_episodes,
        force=args.force,
    )

    total = panda_human_episodes + panda_mg_episodes + iiwa_episodes + ur5e_episodes
    log(
        f"done: panda_human={panda_human_episodes} + panda_mg={panda_mg_episodes} + iiwa={iiwa_episodes} "
        f"+ ur5e={ur5e_episodes} = {total} episodes ready under {raw_dir} "
        f"(repo_ids: panda_human, panda_mg, iiwa, ur5e)"
    )


if __name__ == "__main__":
    main()
