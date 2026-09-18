#!/usr/bin/env python
"""Build training-ready camera subsets of the robocasa_v10_crossembodiment "_traj" datasets for
train_smolVLA_robocasa_v10_traj.sh: same idea as tools/prepare_robocasa_x_dataset.py's
build_subset (drop unwanted cameras via lerobot.datasets.dataset_tools.remove_feature), but generic
over an arbitrary list of sources instead of the hardcoded panda/iiwa/ur5e mix, and with no episode
capping (these datasets are already small: ~100-110 episodes each).

Every subset is also always passed through tools/resize_dataset_videos.py's fix_videos() after the
camera drop: it force-sets a short keyframe interval (critical for training throughput -- see that
module's docstring for the ~50x dataloading slowdown this fixes) and, only if a source's native
camera resolution differs from --target-width/--target-height, resizes it too (needed here because
OpenDrawer/PreheatOven are natively 180x320 while PickPlaceCounterToSink is 256x256, and
MultiLeRobotDataset silently drops any camera feature whose shape isn't identical across all
sub-datasets it combines).

All sources must already be codebase_version "v3.0" (run ./convert_robocasa_to_v30.sh first).

Usage:
    python tools/prepare_robocasa_v10_traj_dataset.py \\
        --source opendrawer:dataset_git/robocasa_v10_crossembodiment/OpenDrawer_UR5eOmron_traj \\
        --source pickplace:dataset_git/robocasa_v10_crossembodiment/PickPlaceCounterToSink_PandaOmron_traj \\
        --source preheatoven:dataset_git/robocasa_v10_crossembodiment/PreheatOven_IIWAOmron_traj \\
        --output-root dataset_git/robocasa_v10_traj_lr \\
        --cameras observation.images.robot0_agentview_left observation.images.robot0_agentview_right
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.datasets.dataset_tools import remove_feature  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.utils import load_info  # noqa: E402

from resize_dataset_videos import fix_videos  # noqa: E402

V30 = "v3.0"


def log(msg: str) -> None:
    print(f"[prepare_robocasa_v10_traj_dataset] {msg}", flush=True)


def require_v30(path: Path, label: str) -> None:
    version = load_info(path).get("codebase_version", "unknown")
    if version != V30:
        raise SystemExit(
            f"{label} at {path} is codebase_version={version!r}, not {V30!r}. "
            f"Run ./convert_robocasa_to_v30.sh first (it converts in place)."
        )


def build_subset(
    source: Path,
    dest: Path,
    tag: str,
    cameras_to_keep: list[str],
    force: bool,
    target_width: int,
    target_height: int,
    gop_size: int,
) -> int:
    """Copy `source` (already v3.0) into `dest`, stripped down to `cameras_to_keep`, then always
    re-encode those cameras via fix_videos() (short keyframe interval, plus a resize if the
    source's native resolution differs from target_width/target_height). Returns the number of
    episodes. Skips entirely if `dest` already exists (unless force)."""
    if dest.exists():
        if not force:
            n = json.loads((dest / "meta/info.json").read_text())["total_episodes"]
            log(f"{dest} already exists ({n} episodes), skipping (use --force to redo)")
            return n
        shutil.rmtree(dest)

    src_ds = LeRobotDataset(repo_id=f"robocasa_v10_traj_{tag}_src", root=source)
    n_total = src_ds.meta.total_episodes

    drop_cameras = [c for c in src_ds.meta.video_keys if c not in cameras_to_keep]
    log(f"{tag}: keeping {n_total} episodes, dropping cameras {drop_cameras or '(none)'}")
    if drop_cameras:
        remove_feature(src_ds, feature_names=drop_cameras, repo_id=f"robocasa_v10_traj_{tag}", output_dir=dest)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_ds.root, dest, dirs_exist_ok=True)

    fix_videos(dest, cameras_to_keep, gop_size=gop_size, width=target_width, height=target_height)

    log(f"{tag}: done -> {dest} ({n_total} episodes)")
    return n_total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        dest="sources",
        metavar="TAG:PATH",
        help="Repeatable. A tag (used as the output repo_id) and source dataset path, joined by ':'.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        required=True,
        help="Camera feature keys to KEEP (all others dropped from every source)",
    )
    parser.add_argument("--force", action="store_true", help="Redo the subset step even if it already exists")
    parser.add_argument(
        "--target-width", type=int, default=256, help="Common camera width all sources are resized to if needed"
    )
    parser.add_argument(
        "--target-height", type=int, default=256, help="Common camera height all sources are resized to if needed"
    )
    parser.add_argument(
        "--gop-size",
        type=int,
        default=1,
        help="Max frames between keyframes, always applied (default 1) -- see resize_dataset_videos.py",
    )
    args = parser.parse_args()

    sources: list[tuple[str, Path]] = []
    for spec in args.sources:
        tag, _, path_str = spec.partition(":")
        if not tag or not path_str:
            raise SystemExit(f"--source must be TAG:PATH, got {spec!r}")
        sources.append((tag, Path(path_str)))

    for tag, path in sources:
        require_v30(path, f"--source {tag}")

    output_root = args.output_root.resolve()
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for tag, path in sources:
        counts[tag] = build_subset(
            path,
            raw_dir / tag,
            tag,
            args.cameras,
            force=args.force,
            target_width=args.target_width,
            target_height=args.target_height,
            gop_size=args.gop_size,
        )

    total = sum(counts.values())
    log(
        f"done: {' + '.join(f'{t}={n}' for t, n in counts.items())} = {total} episodes ready under "
        f"{raw_dir} (repo_ids: {', '.join(t for t, _ in sources)})"
    )


if __name__ == "__main__":
    main()
