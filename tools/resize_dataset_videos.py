#!/usr/bin/env python
"""Re-encode every video file for the given camera keys of a v3.0 LeRobot dataset with a short
keyframe interval, optionally also resizing (letterbox pad, not stretch) to a common WxH and
patching meta/info.json's declared shape/video_info/info to match.

Two independent problems this fixes, both discovered building robocasa_v10_traj_lr:

1. Keyframe density (the main reason this tool exists -- always applied, regardless of --width/
   --height). Video encoded with a sparse (or absent) keyframe interval makes every *random* frame
   read decode from the nearest preceding keyframe forward -- for a from-scratch re-encode with no
   explicit -g, ffmpeg/libopenh264 emitted a single keyframe for an entire ~18-minute, ~22K-frame
   file. Training samples frames in random order, so this made every dataloader fetch potentially
   decode thousands of frames to serve one: measured as `dataloading_s`/`fetch_s` in
   train_smolVLA_*.sh's logging staying pinned around 1s/step no matter how many DataLoader workers
   were added (more workers doesn't help a decode that's fundamentally O(episode length) per
   sample). Forcing -g via --gop-size (default 1, i.e. every frame independently decodable) dropped
   measured `data_s` from ~1.03s/step to ~0.02s/step on this dataset with no change to worker count.
   This is unrelated to lerobot.datasets.dataset_tools.remove_feature's own video handling: that
   copies kept-camera video files verbatim (_copy_videos, a plain shutil.copy) and never
   re-encodes them, so it does not introduce this problem -- it is only a risk for tools (this one
   included) that explicitly re-encode a video without setting a keyframe interval.

2. Shape mismatch (only when --width/--height differ from the source). MultiLeRobotDataset disables
   any feature whose decoded tensor shape differs across its sub-datasets
   (lerobot_dataset.py's `_compute_pad_specs`), and unlike the pre-normalized robocasa_x
   cross-embodiment sources, robocasa_v10_traj_lr's OpenDrawer/PreheatOven cameras are natively
   180x320 while PickPlaceCounterToSink is 256x256 -- so without resizing, both camera features get
   silently dropped from every batch (see "All image features are missing" crash).

Runs ffmpeg directly (not this repo's usual PyAV read path) because this box's ffmpeg has no GPL
h264 encoder (libx264) -- libopenh264 is used instead, which the vera conda env's ffmpeg build has.

Always re-encodes every requested camera (there is no cheap way to check an existing file's
keyframe interval from meta/info.json alone), even when the shape already matches the target.

Usage:
    python tools/resize_dataset_videos.py \\
        --root dataset_git/robocasa_v10_traj_lr/raw/opendrawer \\
        --cameras observation.images.robot0_agentview_left observation.images.robot0_agentview_right \\
        --width 256 --height 256
"""

import argparse
import json
import subprocess
from pathlib import Path

FFMPEG = "/opt/conda/envs/vera/bin/ffmpeg"


def reencode_one(path: Path, gop_size: int, width: int | None, height: int | None) -> None:
    tmp = path.with_suffix(".reencode_tmp.mp4")
    cmd = [FFMPEG, "-y", "-i", str(path)]
    if width is not None and height is not None:
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
        )
        cmd += ["-vf", vf]
    cmd += ["-c:v", "libopenh264", "-pix_fmt", "yuv420p", "-b:v", "2M", "-g", str(gop_size), str(tmp)]
    subprocess.run(cmd, check=True, capture_output=True)
    tmp.replace(path)


def fix_videos(
    root: Path,
    cameras: list[str],
    gop_size: int = 1,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """Re-encode `cameras`' video files under `root` with a short keyframe interval (always), and
    resize them to width x height if given and different from the source's declared shape. Patches
    meta/info.json to match. See module docstring for why the keyframe fix is unconditional."""
    if (width is None) != (height is None):
        raise ValueError("width and height must be given together")

    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())

    for camera in cameras:
        feature = info["features"].get(camera)
        if feature is None:
            raise ValueError(f"{camera} not found in {info_path}")

        target_width, target_height = None, None
        if width is not None and (feature["shape"][1] != width or feature["shape"][0] != height):
            target_width, target_height = width, height

        video_files = sorted((root / "videos" / camera).rglob("*.mp4"))
        if not video_files:
            raise ValueError(f"No video files found for {camera} under {root}")
        for video_file in video_files:
            resize_note = f", resizing to {target_width}x{target_height}" if target_width is not None else ""
            print(f"re-encoding {video_file} (gop={gop_size}{resize_note})")
            reencode_one(video_file, gop_size, target_width, target_height)

        if target_width is not None:
            feature["shape"] = [target_height, target_width, 3]
            for info_block in ("video_info", "info"):
                block = feature.get(info_block)
                if block is not None:
                    if "video.height" in block:
                        block["video.height"] = target_height
                    if "video.width" in block:
                        block["video.width"] = target_width

    info_path.write_text(json.dumps(info, indent=4) + "\n")
    print(f"updated {info_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cameras", type=str, nargs="+", required=True)
    parser.add_argument("--width", type=int, default=None, help="Resize target width (omit to only fix keyframes)")
    parser.add_argument("--height", type=int, default=None, help="Resize target height (omit to only fix keyframes)")
    parser.add_argument(
        "--gop-size",
        type=int,
        default=1,
        help="Max frames between keyframes (default 1 = every frame independently decodable -- see module docstring)",
    )
    args = parser.parse_args()
    fix_videos(args.root, args.cameras, args.gop_size, args.width, args.height)


if __name__ == "__main__":
    main()
