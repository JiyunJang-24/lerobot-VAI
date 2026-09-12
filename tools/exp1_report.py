#!/usr/bin/env python
"""Collect every Experiment 1 metrics.json into the one table the go/no-go decision reads.

Also renders the qualitative retrieval figure: for a handful of queries, the top-K neighbours
under each method side by side. A frozen backbone is expected to retrieve images that LOOK like
the query -- same robot, same colours. A working selective alignment is expected to retrieve
DIFFERENT robots at the same underlying state.

    python tools/exp1_report.py --root outputs/exp1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lerobot.scripts.exp1_data import Exp1Config, Exp1Dataset  # noqa: E402
from lerobot.scripts.exp1_eval import centred, embed  # noqa: E402
from lerobot.scripts.exp1_train import load_tower  # noqa: E402

GROUPS = ["seen_scene/seen_emb", "unseen_scene/seen_emb",
          "seen_scene/unseen_emb", "unseen_scene/unseen_emb"]
SHORT = {"seen_scene/seen_emb": "seen/seen", "unseen_scene/seen_emb": "unseenSc/seenEmb",
         "seen_scene/unseen_emb": "seenSc/unseenEmb", "unseen_scene/unseen_emb": "unseen/unseen"}
NAMES = {"A": "A frozen", "B": "B global", "C": "C selective",
         "D": "D selective+eef", "E": "E selective+eef+anchor"}


def table(root: Path) -> str:
    rows = []
    for m in "ABCDE":
        f = root / m / "metrics.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        g = r.get("groups", {})
        eef = r.get("eef_probe", {})
        ident = r.get("identity_probe", {})
        scene = r.get("scene_probe", {})
        u = "unseen_scene/unseen_emb"
        rows.append({
            "method": NAMES[m],
            **{f"R@1 {SHORT[k]}": g.get(k, {}).get("retrieval", {}).get("R@1", float("nan"))
               for k in GROUPS},
            "eef cm": eef.get(u, {}).get("trans_cm", float("nan")),
            "eef deg": eef.get(u, {}).get("rot_deg", float("nan")),
            "emb probe": ident.get(u, {}).get("embodiment_acc", float("nan")),
            "arm probe": ident.get(u, {}).get("arm_acc", float("nan")),
            "scene probe": scene.get("seen_scene/seen_emb", {}).get("acc", float("nan")),
            "pose disc": g.get(u, {}).get("scene_control", {}).get("pose_discrimination",
                                                                   float("nan")),
        })
    if not rows:
        return "(no metrics.json found)"
    cols = list(rows[0])
    width = {c: max(len(c), *(len(f"{r[c]:.3f}") if isinstance(r[c], float) else len(str(r[c]))
                              for r in rows)) for c in cols}
    out = [" | ".join(c.ljust(width[c]) for c in cols),
           "-|-".join("-" * width[c] for c in cols)]
    for r in rows:
        out.append(" | ".join(
            (f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])).ljust(width[c])
            for c in cols))
    return "\n".join(out)


def neighbours(root: Path, split: Path, group: str, n_query: int, k: int, seed: int) -> Path:
    methods = [m for m in "ABCDE" if (root / m / "vision_tower.safetensors").exists()]
    if not methods:
        raise SystemExit("no checkpoints to visualise")
    device = torch.device("cuda")
    cfg = Exp1Config(split=split, seed=seed)
    data = Exp1Dataset(cfg, group)
    rng = np.random.default_rng(seed)
    items = data.enumerate_group(4, np.random.default_rng(seed))
    queries = rng.choice(len(items), n_query, replace=False)

    fig, axes = plt.subplots(len(methods) * n_query, k + 1,
                             figsize=(1.7 * (k + 1), 1.35 * len(methods) * n_query))
    for mi, m in enumerate(methods):
        tower = load_tower(str(root / m / "vision_tower.safetensors"), device).eval()
        feats = centred(embed(tower, data, items, device))
        sim = feats @ feats.T
        emb_ids = np.array([e for _, _, e in items])
        for qi, q in enumerate(queries):
            row = mi * n_query + qi
            s = np.where(emb_ids == emb_ids[q], -np.inf, sim[q])
            top = np.argsort(-s)[:k]
            sq, tq, eq = items[q]
            ax = axes[row, 0]
            ax.imshow(data.cache.image(sq, tq, eq)); ax.set_xticks([]); ax.set_yticks([])
            ax.set_ylabel(NAMES[m].split()[0], fontsize=8)
            if qi == 0:
                ax.set_title("query", fontsize=8)
            for ki, j in enumerate(top):
                sj, tj, ej = items[j]
                ax = axes[row, ki + 1]
                ax.imshow(data.cache.image(sj, tj, ej)); ax.set_xticks([]); ax.set_yticks([])
                hit = (sj, tj) == (sq, tq)
                for spine in ax.spines.values():
                    spine.set_edgecolor("tab:green" if hit else "tab:red")
                    spine.set_linewidth(2.2)
                if qi == 0 and mi == 0:
                    ax.set_title(f"top {ki + 1}", fontsize=8)
        del tower
        torch.cuda.empty_cache()
    fig.suptitle(f"Nearest neighbours, {group}. Green = same underlying state, red = not.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    path = root / f"neighbours_{group.replace('/', '_')}.png"
    fig.savefig(path, dpi=115, bbox_inches="tight")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("outputs/exp1"))
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--group", default="unseen_scene/unseen_emb")
    ap.add_argument("--queries", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    text = table(args.root)
    print(text)
    (args.root / "results_table.txt").write_text(text + "\n")
    if not args.no_figures:
        p = neighbours(args.root, args.root / "split.json", args.group,
                       args.queries, args.k, args.seed)
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
