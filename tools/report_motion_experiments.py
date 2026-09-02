#!/usr/bin/env python
"""Collect Experiment 1 and 2 into the tables and the figure the brief asks for.

Experiment 1 is the seen-vs-held-out comparison at full embodiment count; experiment 2 is that
held-out number as a function of how many embodiments were trained on, under both sampling
protocols. Protocol A grows the total data with the embodiment count, protocol B holds it fixed --
only B can separate "morphology diversity helped" from "more images helped".

    python tools/report_motion_experiments.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
LADDER = [2, 4, 8, 16, 42]


def load(root: Path, tag: str):
    path = root / tag / "results.json"
    return json.loads(path.read_text()) if path.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "outputs/motion_prediction")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/motion_experiments.png")
    args = ap.parse_args()

    exp1 = load(args.root, "exp1_n42")
    lines = []
    if exp1:
        base = exp1["baseline"]
        lines.append("EXPERIMENT 1 — motion prediction, 42 train / 14 held-out embodiments")
        lines.append(f"  predict-the-mean baseline: {base['translation_mae_m'] * 100:.2f} cm, "
                     f"identity-rotation baseline: {base['rotation_err_deg']:.1f} deg")
        lines.append(f"  {'split':<10}{'trans MAE':>12}{'dir cos':>10}{'rot err':>10}{'grip acc':>10}")
        for row in (exp1["seen"], exp1["heldout"]):
            lines.append(f"  {row['split']:<10}{row['translation_mae_m'] * 100:>9.2f} cm"
                         f"{row['translation_dir_cos']:>10.3f}{row['rotation_err_deg']:>7.1f} deg"
                         f"{row['gripper_acc']:>10.3f}")
        lines.append("")
        lines.append("  per held-out embodiment:")
        lines.append(f"  {'emb':>5}{'split':>10}{'trans MAE':>12}{'rot err':>11}{'grip acc':>10}")
        for row in exp1["per_embodiment"]["heldout"]:
            lines.append(f"  {row['embodiment']:>5}{'heldout':>10}"
                         f"{row['translation_mae_m'] * 100:>9.2f} cm{row['rotation_err_deg']:>8.1f} deg"
                         f"{row['gripper_acc']:>10.3f}")
        for row in exp1["per_embodiment"]["seen"][:5]:
            lines.append(f"  {row['embodiment']:>5}{'seen':>10}"
                         f"{row['translation_mae_m'] * 100:>9.2f} cm{row['rotation_err_deg']:>8.1f} deg"
                         f"{row['gripper_acc']:>10.3f}")

    series = {}
    for protocol, prefix in (("A: samples/embodiment fixed", "exp2A"), ("B: total samples fixed", "exp2B")):
        pts = []
        for n in LADDER:
            data = load(args.root, f"{prefix}_n{n}")
            if data:
                pts.append((data["n_train_embodiments"], data["heldout"], data["seen"],
                            data["n_samples"]))
        if pts:
            series[protocol] = pts

    if series:
        lines.append("")
        lines.append("EXPERIMENT 2 — held-out error vs number of training embodiments")
        for protocol, pts in series.items():
            lines.append(f"  {protocol}")
            lines.append(f"  {'n_emb':>6}{'samples':>10}{'held MAE':>12}{'held rot':>11}"
                         f"{'held dircos':>13}{'seen MAE':>11}")
            for n, held, seen, n_samples in pts:
                lines.append(f"  {n:>6}{n_samples:>10}{held['translation_mae_m'] * 100:>9.2f} cm"
                             f"{held['rotation_err_deg']:>8.1f} deg{held['translation_dir_cos']:>13.3f}"
                             f"{seen['translation_mae_m'] * 100:>8.2f} cm")

    report = "\n".join(lines)
    print(report)
    (args.root / "summary.txt").write_text(report + "\n")

    if series:
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
        metrics = [
            ("translation_mae_m", "held-out translation MAE (cm)", 100.0),
            ("rotation_err_deg", "held-out rotation error (deg)", 1.0),
            ("translation_dir_cos", "held-out direction cosine", 1.0),
        ]
        colours = {"A: samples/embodiment fixed": "#3d5573", "B: total samples fixed": "#b3153b"}
        for ax, (key, label, scale) in zip(axes, metrics, strict=True):
            for protocol, pts in series.items():
                xs = [p[0] for p in pts]
                ys = [p[1][key] * scale for p in pts]
                ax.plot(xs, ys, "o-", lw=2, ms=7, label=protocol, color=colours[protocol])
                for x, y in zip(xs, ys, strict=True):
                    ax.annotate(f"{y:.2f}" if scale != 1.0 or key.endswith("cos") else f"{y:.1f}",
                                (x, y), textcoords="offset points", xytext=(0, 7), ha="center",
                                fontsize=7.5)
            if exp1 and key == "translation_mae_m":
                ax.axhline(exp1["baseline"]["translation_mae_m"] * 100, ls="--", c="gray", lw=1,
                           label="predict-the-mean")
            if exp1 and key == "rotation_err_deg":
                ax.axhline(exp1["baseline"]["rotation_err_deg"], ls="--", c="gray", lw=1,
                           label="identity rotation")
            ax.set_xscale("log", base=2)
            ax.set_xticks(LADDER)
            ax.set_xticklabels(LADDER)
            ax.set_xlabel("synthetic embodiments trained on")
            ax.set_title(label, fontsize=11)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(
            "Experiment 2 — does synthetic embodiment diversity improve motion prediction on "
            "embodiments never trained on?\nProtocol B holds the total sample count fixed, so only "
            "B separates morphology diversity from simply having more images", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        fig.savefig(args.out, dpi=115, bbox_inches="tight")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
