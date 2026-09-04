#!/usr/bin/env python
"""Generate the LAP-style motion labels and check they are worth training on.

Three things decide whether a label set teaches anything, and all three are printed here:

  balance    if one phrase covers most of the data, the model learns that phrase and stops
  entropy    bits per label. CLAUDE.md 8 recorded the LAP objective reaching 1.000 content
             accuracy by 15k steps on 7.72 bits, after which the VLM received no gradient at all
  agreement  the same physical motion must get the same words in both corpora, or a model
             pre-trained on synthetic labels is being asked a different question on real data

    python tools/preview_motion_language.py
"""

import argparse
import glob
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.scripts.motion_data import motion_labels, pose_states, sample_pose_pairs  # noqa: E402
from lerobot.scripts.motion_language import (  # noqa: E402
    MotionDescriber, content_words, fit_thresholds, motion_deltas,
)

SYNTH = "/dataset/jiyun/dataset_git/eef_pairs/56combo_144_bg12_closed"
REAL = ("/dataset/jiyun/dataset_git/visual_robust_new_barx_ur5e/new_barx/"
        "UR5eOmron_PnPSinkToCounter/lerobot")


def log(msg: str) -> None:
    print(msg, flush=True)


def synthetic_pairs(n: int, seed: int):
    frames = pd.concat([pd.read_parquet(f, columns=["episode_index", "observation.state"])
                        for f in sorted(glob.glob(SYNTH + "/data/**/*.parquet", recursive=True))])
    states = np.stack(frames.groupby("episode_index")["observation.state"].first().to_numpy()
                      ).astype(np.float64)
    poses = states[:, :7]
    rng = np.random.default_rng(seed)
    pairs = sample_pose_pairs(poses, n, 0.15, rng)
    # eef_pairs has no gripper column in the state, so pad one; the closed subset never opens it.
    padded = np.concatenate([poses, np.zeros((len(poses), 1))], axis=1)
    return padded[pairs[:, 0]], padded[pairs[:, 1]]


def real_pairs(horizon: int, stride: int):
    frames = pd.concat([pd.read_parquet(f, columns=["episode_index", "observation.state"])
                        for f in sorted(glob.glob(REAL + "/data/**/*.parquet", recursive=True))])
    states = np.stack(frames["observation.state"].to_numpy()).astype(np.float64)
    episodes = frames.episode_index.to_numpy()
    a, b = [], []
    for ep in np.unique(episodes):
        rows = np.flatnonzero(episodes == ep)
        usable = rows[: len(rows) - horizon][::stride]
        a.append(usable)
        b.append(usable + horizon)
    a, b = np.concatenate(a), np.concatenate(b)
    return states[a], states[b]


def report(name: str, sentences: list[str]) -> None:
    counts = Counter(sentences)
    probs = np.array(list(counts.values()), dtype=float)
    probs /= probs.sum()
    entropy = float(-(probs * np.log2(probs)).sum())
    log(f"\n  {name}: {len(sentences)} labels, {len(counts)} distinct, {entropy:.2f} bits")
    log(f"    top-1 covers {100 * max(counts.values()) / len(sentences):.1f}%")
    for sentence, count in counts.most_common(6):
        log(f"      {100 * count / len(sentences):5.1f}%  {sentence}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=int, default=6000)
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    log("=" * 78)
    log("SYNTHETIC  eef_pairs 56combo_144_bg12_closed   (one fixed frame)")
    s_t, s_h = synthetic_pairs(args.pairs, args.seed)
    d_pos, d_euler, _ = motion_deltas(s_t, s_h, "fixed")
    st, sr = fit_thresholds(d_pos), fit_thresholds(d_euler)
    log(f"  translation thresholds (m):   idle<{st[0]:.3f}  slight<{st[1]:.3f}  moderate<{st[2]:.3f}")
    log(f"  rotation thresholds (deg):    idle<{np.degrees(sr[0]):.1f}  "
        f"slight<{np.degrees(sr[1]):.1f}  moderate<{np.degrees(sr[2]):.1f}")
    synth = MotionDescriber(st, sr, grip_threshold=0.5, frame="fixed")
    s_sentences = synth.describe(s_t, s_h)
    report("synthetic", s_sentences)
    log("\n  examples:")
    for i in range(5):
        log(f"    d_pos {np.round(d_pos[i], 3)}  rot {np.round(np.degrees(d_euler[i]), 1)} deg")
        log(f"      -> \"{s_sentences[i]}\"")

    log("\n" + "=" * 78)
    log("REAL  visual_robust trajectories   (world xyz -> rotated into the EEF frame)")
    r_t, r_h = real_pairs(args.horizon, args.stride)
    rd_pos, rd_euler, rd_grip = motion_deltas(r_t, r_h, "eef")
    rt, rr = fit_thresholds(rd_pos), fit_thresholds(rd_euler)
    log(f"  translation thresholds (m):   idle<{rt[0]:.3f}  slight<{rt[1]:.3f}  moderate<{rt[2]:.3f}")
    log(f"  rotation thresholds (deg):    idle<{np.degrees(rr[0]):.1f}  "
        f"slight<{np.degrees(rr[1]):.1f}  moderate<{np.degrees(rr[2]):.1f}")
    gt = float(np.quantile(np.abs(rd_grip), 0.9)) or 0.5
    real = MotionDescriber(rt, rr, grip_threshold=gt, frame="eef")
    r_sentences = real.describe(r_t, r_h)
    report("real", r_sentences)
    log("\n  examples:")
    for i in range(0, 5):
        log(f"    d_pos {np.round(rd_pos[i], 3)}  rot {np.round(np.degrees(rd_euler[i]), 1)} deg")
        log(f"      -> \"{r_sentences[i]}\"")

    log("\n" + "=" * 78)
    log("VOCABULARY AGREEMENT  (a synthetic-pretrained VLM must meet the same words on real data)")
    sv = set().union(*(content_words(s) for s in s_sentences))
    rv = set().union(*(content_words(s) for s in r_sentences))
    log(f"  synthetic uses {len(sv)} content words, real uses {len(rv)}")
    log(f"  only in synthetic: {sorted(sv - rv) or 'none'}")
    log(f"  only in real:      {sorted(rv - sv) or 'none'}")
    log(f"  shared:            {len(sv & rv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
