"""Shared machinery for the cross-embodiment motion-prediction experiments.

Builds (image_t, image_t+h) pairs with an exact Cartesian EEF displacement between them, out of the
eef_pairs renders that already exist. No new rendering is needed, because those renders already have
the property the hypothesis requires: all 56 embodiments are posed at the SAME 48 canonical EEF
states, so a motion defined as "pose i -> pose j" is literally identical across embodiments.

That is stronger than sampling a shared motion *distribution* per robot. Here the pool of pose pairs
is drawn once and reused for every embodiment, so embodiment identity carries exactly zero
information about the label -- the shortcut the hypothesis is worried about is impossible by
construction rather than by sampling luck. assert_no_motion_shortcut() checks it.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EEF_ROOT = Path("/dataset/jiyun/dataset_git/eef_pairs")
SUBSETS = [
    "56combo_48_bg12_closed",
    "56combo_48_bg12_open",
    "56combo_48_bg12_closed_furniture",
    "56combo_48_bg12_open_furniture",
]
CACHE = Path("/dev/shm") / f"eefpairs_cache_{'_'.join(SUBSETS)}.pt"


def log(msg: str) -> None:
    print(f"[motion_data] {msg}", flush=True)


# --------------------------------------------------------------------------------------------
# quaternions.  eef_pairs stores xyzw; the barx policy corpus stores wxyz (verified, section 9).
# Everything below is xyzw, and to_wxyz/from_wxyz are the only places the other convention appears.
# --------------------------------------------------------------------------------------------
def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.concatenate([-q[..., :3], q[..., 3:]], axis=-1)


def quat_angle_deg(q: np.ndarray) -> np.ndarray:
    """Rotation magnitude, sign-invariant: q and -q are one rotation."""
    w = np.clip(np.abs(q[..., 3]), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(w))


def geodesic_deg(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = pred / np.linalg.norm(pred, axis=-1, keepdims=True).clip(1e-9)
    target = target / np.linalg.norm(target, axis=-1, keepdims=True).clip(1e-9)
    dot = np.clip(np.abs((pred * target).sum(-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


# --------------------------------------------------------------------------------------------
def build_table() -> pd.DataFrame:
    """One row per rendered frame, in EXACTLY the order the /dev/shm image cache was written in.

    The cache is indexed positionally against pretrain_siglip_eefpairs.build_row_table, so this
    reproduces that function's ordering (sorted glob, concat in subset order) and then asserts the
    result lines up. Getting this silently wrong would pair every image with another image's label.
    """
    tables, offset = [], 0
    for subset in SUBSETS:
        files = sorted(glob.glob(str(EEF_ROOT / subset / "data" / "**" / "*.parquet"), recursive=True))
        table = pd.concat([
            pd.read_parquet(f, columns=[
                "episode_index", "frame_index", "observation.embodiment_index",
                "observation.background_index", "observation.camera_view_index",
                "observation.state",
            ]) for f in files
        ], ignore_index=True)
        table = table.rename(columns={
            "observation.embodiment_index": "embodiment",
            "observation.background_index": "background",
            "observation.camera_view_index": "view",
        })
        lengths = table.groupby("episode_index").size().sort_index()
        starts = lengths.cumsum().shift(fill_value=0)
        table["row"] = table.episode_index.map(starts) + table.frame_index + offset
        table["subset"] = subset
        table["gripper"] = 0 if "closed" in subset else 1
        table["pose"] = table.episode_index
        tables.append(table)
        offset += int(lengths.sum())
    out = pd.concat(tables, ignore_index=True)
    out["cache_pos"] = np.arange(len(out))

    from lerobot.scripts.pretrain_siglip_eefpairs import build_row_table

    reference = build_row_table(EEF_ROOT, SUBSETS, share_poses=True).reset_index(drop=True)
    assert len(reference) == len(out), f"cache ordering: {len(reference)} vs {len(out)} rows"
    assert (reference["row"].to_numpy() == out["row"].to_numpy()).all(), "cache ordering diverged"
    return out


def pose_states(table: pd.DataFrame) -> np.ndarray:
    """(48, 7) xyz + quat_xyzw. One value per pose -- verified std 3e-5 within a pose."""
    first = table.drop_duplicates("pose").sort_values("pose")
    return np.stack(first["observation.state"].to_numpy()).astype(np.float64)[:, :7]


def sample_pose_pairs(states: np.ndarray, n_pairs: int, max_delta: float,
                      rng: np.random.Generator) -> np.ndarray:
    """One pool of (i, j) pose pairs, reused for EVERY embodiment.

    max_delta keeps the motions local. The 48 poses span 0.009-0.307 m pairwise; the real corpus
    moves ~0.0025 m per step, so 0.15 m is about a 60-step horizon rather than a teleport.
    """
    xyz = states[:, :3]
    dist = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)
    i, j = np.where((dist > 0) & (dist <= max_delta))
    keep = rng.choice(len(i), size=min(n_pairs, len(i)), replace=False)
    pairs = np.stack([i[keep], j[keep]], axis=1)
    log(f"pose-pair pool: {len(pairs)} of {len(i)} pairs within {max_delta} m "
        f"(|d| mean {dist[i[keep], j[keep]].mean():.3f} m)")
    return pairs


def motion_labels(states: np.ndarray, pairs: np.ndarray):
    """delta_position (metres) and delta_rotation (quaternion xyzw), both base-frame.

    The convention is fixed here and nowhere else: d_pos = p_j - p_i, and q_rel = q_j * conj(q_i)
    so that q_rel applied on the LEFT of q_i gives q_j. verify_reconstruction() checks exactly that,
    which is the sanity test the experiment brief asks for.
    """
    d_pos = states[pairs[:, 1], :3] - states[pairs[:, 0], :3]
    q_rel = quat_mul(states[pairs[:, 1], 3:7], quat_conj(states[pairs[:, 0], 3:7]))
    return d_pos, q_rel


def verify_reconstruction(states: np.ndarray, pairs: np.ndarray) -> None:
    """s_t + delta must reconstruct s_t+h, or every label below is meaningless."""
    d_pos, q_rel = motion_labels(states, pairs)
    pos_back = states[pairs[:, 0], :3] + d_pos
    pos_err = np.abs(pos_back - states[pairs[:, 1], :3]).max()
    quat_back = quat_mul(q_rel, states[pairs[:, 0], 3:7])
    # sign-invariant: q and -q are the same rotation
    sign = np.sign((quat_back * states[pairs[:, 1], 3:7]).sum(-1, keepdims=True))
    quat_err = np.abs(quat_back * sign - states[pairs[:, 1], 3:7]).max()
    log(f"reconstruction check: max |pos| err {pos_err:.2e} m, max |quat| err {quat_err:.2e}")
    assert pos_err < 1e-9, f"position delta does not reconstruct the target: {pos_err}"
    assert quat_err < 1e-6, f"rotation delta does not reconstruct the target: {quat_err}"


def assert_no_motion_shortcut(pairs_per_embodiment: dict[int, np.ndarray]) -> None:
    """Every embodiment must see the identical motion pool, or identity predicts motion."""
    reference = None
    for emb, pairs in pairs_per_embodiment.items():
        key = np.sort(pairs.view([("i", pairs.dtype), ("j", pairs.dtype)]).ravel())
        if reference is None:
            reference = key
        elif not np.array_equal(reference, key):
            raise AssertionError(f"embodiment {emb} has a different motion pool -- shortcut possible")
    log(f"motion-pool check: all {len(pairs_per_embodiment)} embodiments share one pool")


def assert_no_leakage(train: list[int], heldout: list[int]) -> None:
    overlap = sorted(set(train) & set(heldout))
    if overlap:
        raise AssertionError(f"held-out embodiments appear in training: {overlap}")
    log(f"leakage check: {len(train)} train / {len(heldout)} held-out embodiments, no overlap")


def load_cache():
    if not CACHE.exists():
        raise FileNotFoundError(
            f"{CACHE} missing -- run pretrain_siglip_eefpairs.py once to build the image cache.")
    log(f"loading image cache {CACHE} (53 GB, takes a minute) ...")
    return torch.load(CACHE)


def load_images(cache, positions, device):
    from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad

    images = cache[positions].to(device=device, dtype=torch.float32) / 255.0
    return resize_with_pad(images, 512, 512, pad_value=0) * 2.0 - 1.0
