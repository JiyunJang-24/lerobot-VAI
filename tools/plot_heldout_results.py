#!/usr/bin/env python
"""Plot the held-out embodiment results: the scaling ladder and the objective comparison.

Reads the JSON that tools/eval_heldout_embodiment.py writes. Two figures:

  heldout_scaling.png     metric vs number of embodiments seen during pre-training
  heldout_objectives.png  the four objectives at n=42, on one shared held-out set

The ladder points do NOT share a held-out set -- each run is scored on whatever it did not train
on, which is the honest comparison but means the x axis also changes the test set (n=4 is scored
on 52 embodiments, n=32 on 24). That is stated on the figure rather than hidden, because it is the
main caveat when reading the trend.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LADDER = [(4, "all4_n4"), (8, "all4_n8"), (16, "all4_n16"), (32, "all4_n32"), (42, "all4_n42")]
OBJECTIVES = [
    ("all4_n42", "contrastive"),
    ("all4_n42_eefstate", "EEF state\n(xyz+quat)"),
    ("all4_n42_eefpixel", "EEF pixel\n(u,v)"),
    ("all4_n42_all", "all three"),
]


def load(path: Path, key: str):
    data = json.loads(path.read_text())
    return data["results"][key], len(data["holdout"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs")
    args = ap.parse_args()

    # ---- ladder ---------------------------------------------------------------------------------
    xs, gaps, retr, probes, n_holdout, baseline = [], [], [], [], [], {}
    for n, tag in LADDER:
        path = args.out_dir / (f"heldout_{tag}.json" if n != 42 else "heldout_n42_objectives.json")
        row, n_ho = load(path, tag)
        xs.append(n)
        gaps.append(row["heldout"]["gap_A_minus_B"])
        retr.append(row["pose_retrieval_top1"])
        probes.append(row.get("embodiment_probe_acc", float("nan")))
        n_holdout.append(n_ho)
        if not baseline:
            base, _ = load(path, "pretrained (init)")
            baseline = {"gap": base["heldout"]["gap_A_minus_B"],
                        "retr": base["pose_retrieval_top1"],
                        "probe": base.get("embodiment_probe_acc", float("nan"))}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    panels = [
        (gaps, "held-out gap  (A − B)", baseline["gap"], "higher = same pose across unseen robots\nlooks more alike than a different pose"),
        (retr, "pose retrieval top-1", baseline["retr"], "query an UNSEEN embodiment against seen ones;\ncorrect if the neighbour shares the pose"),
        (probes, "embodiment probe acc", baseline["probe"], "can a linear probe still tell WHICH robot?\nlower = identity suppressed"),
    ]
    for ax, (ys, title, base_val, sub) in zip(axes, panels, strict=True):
        ax.plot(xs, ys, "o-", lw=2, ms=7, color="tab:blue", label="pre-trained on N embodiments")
        ax.axhline(base_val, ls="--", c="gray", label=f"stock SigLIP ({base_val:+.3f})")
        for x, y, ho in zip(xs, ys, n_holdout, strict=True):
            ax.annotate(f"{y:.3f}\n({ho} held out)", (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs); ax.set_xticklabels(xs)
        ax.set_xlabel("embodiments seen during pre-training")
        ax.set_title(title, fontsize=11)
        ax.text(0.5, -0.28, sub, transform=ax.transAxes, ha="center", fontsize=8, color="dimgray")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Scaling embodiment diversity — every point scored ONLY on embodiments that run never saw\n"
        "caveat: the held-out set shrinks as N grows (52 → 14), so the x axis moves the test set too",
        fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.90])
    scaling_path = args.out_dir / "heldout_scaling.png"
    fig.savefig(scaling_path, dpi=110, bbox_inches="tight")
    print(f"wrote {scaling_path}")

    # ---- objectives -----------------------------------------------------------------------------
    path = args.out_dir / "heldout_n42_objectives.json"
    names, o_gap, o_retr, o_probe = [], [], [], []
    for tag, label in OBJECTIVES:
        row, _ = load(path, tag)
        names.append(label)
        o_gap.append(row["heldout"]["gap_A_minus_B"])
        o_retr.append(row["pose_retrieval_top1"])
        o_probe.append(row.get("embodiment_probe_acc", float("nan")))
    base, _ = load(path, "pretrained (init)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    bars = [
        (o_gap, "held-out gap  (A − B)", base["heldout"]["gap_A_minus_B"]),
        (o_retr, "pose retrieval top-1", base["pose_retrieval_top1"]),
        (o_probe, "embodiment probe acc", base.get("embodiment_probe_acc", float("nan"))),
    ]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for ax, (ys, title, base_val) in zip(axes, bars, strict=True):
        ax.bar(range(len(names)), ys, color=colors)
        ax.axhline(base_val, ls="--", c="gray", label=f"stock SigLIP ({base_val:+.3f})")
        for i, y in enumerate(ys):
            ax.annotate(f"{y:.3f}", (i, y), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=9)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=9)
        ax.set_title(title, fontsize=11); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    fig.suptitle(
        "Which pre-training objective? — all at 42 embodiments, scored on the SAME 14 held out",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    obj_path = args.out_dir / "heldout_objectives.png"
    fig.savefig(obj_path, dpi=110, bbox_inches="tight")
    print(f"wrote {obj_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
