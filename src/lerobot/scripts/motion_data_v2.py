"""Loader for the redesigned eef_pairs export (`56combo_8000_local_motion_27cam_kitchen_*`).

The layout changed completely and for the better, so this is a separate module rather than a
contortion of motion_data.py:

    old   one image per row; a training pair had to be ASSEMBLED by choosing two rows that shared
          an embodiment, a background, a view and a colour, and differed in pose. Every axis was a
          directory or an index to be lined up, and getting it wrong was silent.
    new   ONE PAIR PER ROW. `observation.images.current` and `.next` are two video streams, and the
          label is already in the row. Nothing has to be paired.

Also new, and directly useful:
    delta_eef_cam    the displacement expressed in the CAMERA frame. With 27 cameras a world-frame
                     label would be ambiguous from images alone -- the model would have to identify
                     the camera first. The camera-frame label removes that, which is the whole
                     reason varying the camera is safe here.
    *_high           512x910 renders alongside the 180x320 ones. At 180x320 a 5 degree rotation
                     moved the gripper's extremities 0.87 px, about a ninth of a patch, which is
                     why rotation was never learnable. This makes that testable rather than assumed.
    embodiments.json the index -> {arm, gripper} mapping that was missing, so "unseen ARM" and
                     "unseen GRIPPER" can finally be separated.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/dataset/jiyun/dataset_git/eef_pairs")
DEFAULT = "56combo_8000_local_motion_27cam_kitchen_l8s0"
LOW = ("observation.images.current", "observation.images.next")
HIGH = ("observation.images.current_high", "observation.images.next_high")


def log(msg: str) -> None:
    print(f"[motion_v2] {msg}", flush=True)


def cache_path(subset: str, high: bool) -> Path:
    return Path("/dev/shm") / f"motion_v2_{subset}{'_high' if high else ''}.pt"


def build_table(subset: str = DEFAULT) -> pd.DataFrame:
    """One row per motion pair, with the label already attached."""
    files = sorted(glob.glob(str(ROOT / subset / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {ROOT / subset / 'data'}")
    keep = ["episode_index", "frame_index", "observation.state", "observation.state_next",
            "observation.delta_eef", "observation.delta_eef_cam",
            "observation.translation_magnitude_m", "observation.rotation_magnitude_degrees",
            "observation.gripper_delta", "observation.embodiment_index",
            "observation.camera_index", "observation.eef_in_frame",
            "observation.eef_in_frame_next"]
    frames = []
    for f in files:
        available = pd.read_parquet(f, columns=None).columns if False else None  # noqa: F841
        frames.append(pd.read_parquet(f, columns=keep))
    table = pd.concat(frames, ignore_index=True)
    table = table.rename(columns={"observation.embodiment_index": "embodiment",
                                  "observation.camera_index": "camera"})
    lengths = table.groupby("episode_index").size().sort_index()
    starts = lengths.cumsum().shift(fill_value=0)
    table["row"] = table.episode_index.map(starts) + table.frame_index
    table["cache_pos"] = np.arange(len(table))
    log(f"{len(table)} pairs, {table.embodiment.nunique()} embodiments, "
        f"{table.camera.nunique()} cameras")
    return table


def embodiment_names(subset: str = DEFAULT) -> dict:
    """index -> {arm, gripper}, if the export shipped it."""
    path = ROOT / subset / "meta" / "embodiments.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def deltas(table: pd.DataFrame, frame: str = "cam"):
    """(translation, rotation quaternion xyzw, gripper delta).

    frame="cam" is the default on purpose: with many cameras a world-frame label cannot be read off
    the images without first identifying the camera.
    """
    key = "observation.delta_eef_cam" if frame == "cam" else "observation.delta_eef"
    d = np.stack(table[key].to_numpy()).astype(np.float64)
    grip = np.stack(table["observation.gripper_delta"].to_numpy()).astype(np.float64).ravel()
    return d[:, :3], d[:, 3:7], grip


def usable(table: pd.DataFrame) -> pd.DataFrame:
    """Pairs where the gripper is visible in BOTH frames.

    A pair with the EEF out of frame has no visual evidence for the label, so it is noise for this
    task -- not a hard example.
    """
    mask = (table["observation.eef_in_frame"].to_numpy().astype(bool)
            & table["observation.eef_in_frame_next"].to_numpy().astype(bool))
    log(f"{int(mask.sum())}/{len(table)} pairs have the gripper visible in both frames "
        f"({100 * mask.mean():.0f}%)")
    return table[mask].reset_index(drop=True)


def split_embodiments(table: pd.DataFrame, n_heldout: int, seed: int = 0):
    """Random held-out split, plus the named split if the mapping exists."""
    embs = np.sort(table.embodiment.unique())
    rng = np.random.default_rng(seed)
    heldout = sorted(int(e) for e in rng.choice(embs, size=n_heldout, replace=False))
    train = sorted(int(e) for e in embs if e not in set(heldout))
    assert not (set(train) & set(heldout)), "held-out embodiments leaked into training"
    log(f"{len(train)} train / {len(heldout)} held-out embodiments, no overlap")
    return train, heldout


def load_cache(subset: str = DEFAULT, high: bool = False):
    path = cache_path(subset, high)
    if not path.exists():
        raise FileNotFoundError(f"{path} -- build it with tools/build_motion_cache_v2.py")
    log(f"mapping {path} ({path.stat().st_size / 1e9:.0f} GB)")
    return torch.load(path, mmap=True)
