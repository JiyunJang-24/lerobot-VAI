#!/usr/bin/env python
"""Collect the VLM motion runs into one table and figure.

Scores are per-category Jaccard over the claims in the generated sentence. Reported separately
because the categories are not comparable: synthetic and real agree on what translation words mean
(bucket thresholds within 7%) but are 7x apart on rotation, so a single blended number would hide
the one thing worth knowing.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS = ["zeroshot", "rot_on", "rot_on_s1", "rot_off", "novary"]
LABEL = {"zeroshot": "no training\n(stock VLM)", "rot_on": "trained", "rot_on_s1": "trained\n(seed 1)",
         "rot_off": "trained\nno rotation words", "novary": "trained\nno bg/furniture variation"}
JACO = ["JacoOmron", "JacoOmronPandaGripper"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/vlm_motion_results.png")
    args = ap.parse_args()

    data = {}
    for tag in RUNS:
        path = REPO_ROOT / f"outputs/vlm_eval_{tag}.json"
        if path.exists():
            data[tag] = json.loads(path.read_text())
    if not data:
        print("no results yet")
        return 1

    def real_mean(d, keys, metric):
        vals = [d["real"][e][metric] for e in keys if e in d.get("real", {})]
        return float(np.mean(vals)) if vals else float("nan")

    rows = []
    for tag, d in data.items():
        syn = d.get("synthetic", {})
        seen_real = [e for e in d.get("real", {}) if e not in JACO]
        rows.append({
            "tag": tag,
            "syn_seen": syn.get("seen", {}).get("all", float("nan")),
            "syn_held": syn.get("heldout", {}).get("all", float("nan")),
            "syn_held_tr": syn.get("heldout", {}).get("translation", float("nan")),
            "real_seen_tr": real_mean(d, seen_real, "translation"),
            "real_jaco_tr": real_mean(d, JACO, "translation"),
            "real_seen_all": real_mean(d, seen_real, "all"),
            "real_jaco_all": real_mean(d, JACO, "all"),
        })

    hdr = (f"{'run':<12}{'SYN seen':>10}{'SYN held':>10}{'SYN held tr':>13}"
           f"{'REAL seen tr':>14}{'REAL Jaco tr':>14}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['tag']:<12}{r['syn_seen']:>10.3f}{r['syn_held']:>10.3f}{r['syn_held_tr']:>13.3f}"
              f"{r['real_seen_tr']:>14.3f}{r['real_jaco_tr']:>14.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    tags = [r["tag"] for r in rows]
    x = np.arange(len(tags))
    colours = ["#8b95a3" if t == "zeroshot" else "#3d5573" for t in tags]

    for ax, (keys, names, title) in zip(axes, [
        (["syn_seen", "syn_held"], ["seen embodiments", "HELD-OUT embodiments"],
         "SYNTHETIC — all claims"),
        (["real_seen_tr", "real_jaco_tr"], ["seen robots", "HELD-OUT Jaco"],
         "REAL trajectories — translation claims only"),
    ], strict=True):
        width = 0.38
        for i, (k, nm) in enumerate(zip(keys, names, strict=True)):
            vals = [r[k] for r in rows]
            ax.bar(x + (i - 0.5) * width, vals, width,
                   color=[c if i == 0 else "#b3153b" if t != "zeroshot" else "#c9b1b8"
                          for c, t in zip(colours, tags, strict=True)],
                   label=nm, edgecolor="white")
            for xi, v in zip(x + (i - 0.5) * width, vals, strict=True):
                if v == v:
                    ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 3),
                                ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL.get(t, t) for t in tags], fontsize=8)
        ax.set_ylabel("claim Jaccard (1.0 = exactly right)")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8.5)
        ax.set_ylim(0, 1.0)
    fig.suptitle("Can the VLM answer 'how did the gripper move?' from two images?\n"
                 "trained only on synthetic renders; the real column is zero-shot transfer",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(args.out, dpi=115, bbox_inches="tight")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
