"""LAP-style English descriptions of the motion between two EEF states.

The ground truth for the two-image motion task, phrased as a sentence instead of a vector:

    "move forward moderately, move right slightly, turn left slightly, close gripper"

Why this and not the regression head already in train_motion_prediction.py: the regression head
transfers DIRECTION to a novel arm but not MAGNITUDE (CLAUDE.md 10.7b -- direction cosine
0.712 -> 0.756 with the seed ranges separated, while MAE 2.82 -> 2.90 overlaps). Pixel displacement
scale depends on how big the arm looks and how it is framed, and that is exactly what a new
morphology changes. Language quantises magnitude into a few buckets, which is a coarser claim that
may survive the appearance change when centimetres do not. That is the thing worth testing.

FRAME. Direction words are only meaningful in a frame the model can see:

  * eef_pairs renders every embodiment at the same pose values under 4 canonical cameras, so its
    coordinates are already one fixed frame. Use them directly.
  * The real trajectories store WORLD xyz with the base parked differently per episode -- measured
    1.52 m of cross-episode spread in x, and net motion that flips sign between episodes. "Right"
    in world coordinates is not "right" in the image, so world-frame words would be noise.
    frame="eef" rotates the displacement into the gripper's own frame at time t, which is
    episode-independent and visible in the image.

Thresholds are quantiles of the actual displacement distribution rather than fixed constants, so
the three magnitude buckets are populated evenly. A label set dominated by one phrase is
memorisable, and CLAUDE.md 8 records exactly that failure: the LAP objective hit 1.000 content
accuracy by 15k steps and then stopped teaching the VLM anything.
"""

from __future__ import annotations

import numpy as np

from lerobot.scripts.motion_data import quat_conj, quat_mul

# +x forward, +y left, +z up is the robotics convention; the words follow it.
AXIS_WORDS = (("forward", "backward"), ("left", "right"), ("up", "down"))
ROT_WORDS = (("roll left", "roll right"), ("tilt up", "tilt down"), ("turn left", "turn right"))
MAGNITUDES = ("slightly", "moderately", "far")


def rotate_into(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Express v in the frame that q defines, i.e. apply q^-1 to v."""
    padded = np.concatenate([v, np.zeros((len(v), 1))], axis=1)
    return quat_mul(quat_mul(quat_conj(q), padded), q)[:, :3]


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """xyzw quaternion -> roll, pitch, yaw in radians, sign-consistent.

    Canonicalised to the w >= 0 hemisphere FIRST: q and -q are one rotation but give euler angles
    that differ by pi, which would put a single physical motion into opposite word buckets.
    """
    q = np.where(q[:, 3:4] < 0, -q, q)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=1)


def motion_deltas(state_t: np.ndarray, state_h: np.ndarray, frame: str = "fixed"):
    """(d_translation, d_euler, d_gripper) between two [>=7] EEF states, xyz + quat_xyzw."""
    d_pos = state_h[:, :3] - state_t[:, :3]
    if frame == "eef":
        d_pos = rotate_into(state_t[:, 3:7], d_pos)
    elif frame != "fixed":
        raise ValueError(f"frame must be 'fixed' or 'eef', got {frame!r}")
    d_euler = quat_to_euler(quat_mul(state_h[:, 3:7], quat_conj(state_t[:, 3:7])))
    d_grip = (state_h[:, 7] - state_t[:, 7]) if state_t.shape[1] > 7 else np.zeros(len(state_t))
    return d_pos, d_euler, d_grip


def fit_thresholds(values: np.ndarray, idle_quantile=0.35, mid_quantile=0.75) -> tuple:
    """(idle, slight/moderate, moderate/far) cut points from |values|.

    Quantiles of the data, not constants: the same word should mean roughly the same share of the
    distribution in both corpora, whose scales differ.
    """
    magnitude = np.abs(values).ravel()
    return (
        float(np.quantile(magnitude, idle_quantile)),
        float(np.quantile(magnitude, mid_quantile)),
        float(np.quantile(magnitude, 0.93)),
    )


class MotionDescriber:
    """Turns a pair of EEF states into one LAP-style sentence."""

    def __init__(self, trans_thresholds, rot_thresholds, grip_threshold: float,
                 include_rotation: bool = True, frame: str = "fixed"):
        self.trans = trans_thresholds
        self.rot = rot_thresholds
        self.grip = float(grip_threshold)
        self.include_rotation = include_rotation
        self.frame = frame

    @staticmethod
    def _phrase(value: float, thresholds, words, verb: str) -> str | None:
        if abs(value) < thresholds[0]:
            return None
        word = words[0] if value > 0 else words[1]
        size = MAGNITUDES[0] if abs(value) < thresholds[1] else (
            MAGNITUDES[1] if abs(value) < thresholds[2] else MAGNITUDES[2])
        return f"{verb}{word} {size}" if verb else f"{word} {size}"

    def describe(self, state_t: np.ndarray, state_h: np.ndarray) -> list[str]:
        d_pos, d_euler, d_grip = motion_deltas(state_t, state_h, self.frame)
        out = []
        for i in range(len(state_t)):
            parts = []
            for axis in range(3):
                phrase = self._phrase(d_pos[i, axis], self.trans, AXIS_WORDS[axis], "move ")
                if phrase:
                    parts.append(phrase)
            if self.include_rotation:
                for axis in range(3):
                    phrase = self._phrase(d_euler[i, axis], self.rot, ROT_WORDS[axis], "")
                    if phrase:
                        parts.append(phrase)
            if abs(d_grip[i]) >= self.grip:
                parts.append("close gripper" if d_grip[i] > 0 else "open gripper")
            out.append(", ".join(parts) if parts else "stay still")
        return out


def content_words(sentence: str) -> set:
    """The words a scorer should check -- direction, magnitude, gripper; not punctuation or 'move'."""
    flat = {w for pair in AXIS_WORDS + ROT_WORDS for w in " ".join(pair).split()}
    flat |= set(MAGNITUDES) | {"open", "close", "still"}
    return {w for w in sentence.replace(",", " ").split() if w in flat}
