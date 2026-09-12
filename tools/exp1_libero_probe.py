#!/usr/bin/env python
"""Metric 4: does the alignment destroy OBJECT information? Measured on LIBERO, not on selfws.

selfws_v2 labels the manipulated object only through the task string, and the task string is 1:1
with the episode, which is 1:1 with the scene. An "object identity" probe there would be a
relabelled scene probe -- it would look fine and prove nothing. visual_robust_libero's
synthetic_replay shards carry `observation.environment_state`, real object poses, and render 24
embodiments of every row, so the same counterfactual structure is available with a genuine object
target.

The honest caveat, stated up front: the encoders were trained on RoboCasa kitchens and LIBERO is
a different domain, so this is a CROSS-DOMAIN probe. A drop here mixes "object information was
destroyed" with "the encoder moved away from LIBERO". The comparison between methods is still
meaningful because every method pays the same domain cost.

Reports, per checkpoint:
  * object-pose regression R^2 from frozen features (ridge, fit and tested on disjoint rows)
  * cross-embodiment state retrieval on LIBERO, which includes kinova3 -- an arm that appears in
    no selfws corpus at all

    python tools/exp1_libero_probe.py --checkpoints outputs/exp1/A outputs/exp1/D
"""

import argparse
import glob
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.scripts.exp1_eval import apply_ridge, centred, ridge  # noqa: E402
from lerobot.scripts.exp1_train import load_tower, prepare  # noqa: E402

ROOT = Path("/dataset/jiyun/dataset_git/visual_robust_libero/synthetic_replay")


def log(msg: str) -> None:
    print(f"[libero] {msg}", flush=True)


def shards() -> list[tuple[str, Path]]:
    """(label, path). The leaf directory name is a render-config string shared across tasks, so
    the task-level directory is what identifies the shard."""
    out = []
    for info in sorted(ROOT.glob("*/*/*/meta/info.json")):
        path = info.parent.parent
        out.append((path.parents[1].name.replace("_1_lerobot", ""), path))
    return out


def load_rows(label: str, shard: Path, embodiments: list[str], per_shard: int, rng) -> tuple:
    files = sorted(glob.glob(str(shard / "data" / "*" / "*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    keep = rng.choice(len(df), min(per_shard, len(df)), replace=False)
    df = df.iloc[np.sort(keep)].reset_index(drop=True)
    images, meta = [], []
    for i, row in df.iterrows():
        for e in embodiments:
            col = f"observation.image.{e}"
            if col not in df.columns:
                continue
            images.append(np.asarray(Image.open(io.BytesIO(row[col]["bytes"])).convert("RGB")))
            meta.append((label, int(i), e))
    env = np.stack(df["observation.environment_state"].to_numpy()).astype(np.float32)
    return np.stack(images), meta, env


@torch.no_grad()
def embed(tower, images, device, batch=12) -> np.ndarray:
    out = []
    for i in range(0, len(images), batch):
        x = prepare(images[i:i + batch], device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(tower(pixel_values=x, patch_attention_mask=None).last_hidden_state
                       .float().mean(1).cpu())
    return torch.cat(out).numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    ap.add_argument("--embodiments", default=("panda_pandagripper,iiwa_robotiq85gripper,"
                                              "ur5e_robotiq140gripper,jaco_rethinkgripper,"
                                              "kinova3_pandagripper,sawyer_robotiq85gripper"))
    ap.add_argument("--per-shard", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/exp1/libero_probe.json"))
    args = ap.parse_args()

    device = torch.device("cuda")
    embs = args.embodiments.split(",")
    rng = np.random.default_rng(args.seed)

    # Per-shard, not pooled. LIBERO suites describe different object sets, so
    # environment_state has a different width AND a different meaning per suite -- 47 dims here,
    # 110 there. Pooling them into one regression (even truncated to a common prefix) would be
    # regressing incomparable quantities. Each shard gets its own ridge and its own gallery, and
    # the reported number is the mean over shards.
    per_shard = {}
    for label, shard in shards():
        imgs, meta, env = load_rows(label, shard, embs, args.per_shard, rng)
        per_shard[label] = (imgs, meta, env)
        log(f"{label}: {len(imgs)} images, env_state {env.shape}")

    results = {}
    for ckpt in args.checkpoints:
        f = ckpt / "vision_tower.safetensors"
        tower = load_tower(str(f) if f.exists() else "", device).eval()
        r2s, r1s, r5s, n_tot = [], [], [], 0
        frac_pos, alphas = [], []
        for label, (imgs, meta, env) in per_shard.items():
            feats = embed(tower, imgs, device)
            row_ids = np.array([m[1] for m in meta])
            emb_ids = np.array([embs.index(m[2]) for m in meta])
            # Drop object-state dimensions that barely move inside this shard. Their explained
            # variance denominator is ~0, so a tiny prediction error becomes an R^2 of -10^3 and
            # the mean over dimensions is then reporting numerical noise, not object information.
            live = env.std(0) > 1e-3
            env = env[:, live]
            target = ((env - env.mean(0)) / env.std(0))[row_ids]

            # split by ROW, never by image: the same row rendered with another arm is the same
            # object configuration, and letting it straddle the split would leak the answer
            rows = np.unique(row_ids)
            order = np.random.default_rng(args.seed).permutation(len(rows))
            cut = max(1, int(0.7 * len(rows)))
            tr_rows = set(rows[order[:cut]].tolist())
            tr = np.array([i for i, r in enumerate(row_ids) if r in tr_rows])
            te = np.array([i for i, r in enumerate(row_ids) if r not in tr_rows])
            if len(te) == 0 or len(tr) == 0:
                continue
            # alpha picked on a split of the TRAINING rows -- 768 features against a few hundred
            # rows overfits badly at the alpha that suited the selfws probe
            inner = max(1, int(0.75 * len(tr)))
            best, best_alpha = -np.inf, 1e3
            for alpha in (1e1, 1e2, 1e3, 1e4, 1e5):
                w = ridge(feats[tr[:inner]], target[tr[:inner]], alpha=alpha)
                pr = apply_ridge(w, feats[tr[inner:]])
                if len(pr) == 0:
                    continue
                score = -float(((pr - target[tr[inner:]]) ** 2).mean())
                if score > best:
                    best, best_alpha = score, alpha
            w = ridge(feats[tr], target[tr], alpha=best_alpha)
            pred = apply_ridge(w, feats[te])
            ss_res = ((pred - target[te]) ** 2).sum(0)
            ss_tot = ((target[te] - target[te].mean(0)) ** 2).sum(0)
            keep = ss_tot > 1e-6
            r2_dims = 1 - ss_res[keep] / ss_tot[keep]
            # median over dimensions, not mean: one ill-conditioned dimension should not decide
            r2s.append(float(np.median(r2_dims)))
            frac_pos.append(float((r2_dims > 0).mean()))
            alphas.append(best_alpha)

            z = centred(feats)
            sim = np.where(emb_ids[:, None] == emb_ids[None, :], -np.inf, z @ z.T)
            order2 = np.argsort(-sim, axis=1)
            hit = row_ids[order2] == row_ids[:, None]
            r1s.append(float(hit[:, 0].mean()))
            r5s.append(float(hit[:, :5].any(1).mean()))
            n_tot += len(feats)

        results[ckpt.name] = {
            "object_state_r2_median": float(np.mean(r2s)),
            "object_dims_predicted": float(np.mean(frac_pos)),
            "object_state_r2_per_shard": r2s,
            "ridge_alpha_per_shard": alphas,
            "retrieval_R@1": float(np.mean(r1s)),
            "retrieval_R@5": float(np.mean(r5s)),
            "n": n_tot, "n_shards": len(r2s),
        }
        log(f"{ckpt.name}: object R^2(med) {np.mean(r2s):+.3f}  dims>0 {np.mean(frac_pos):.3f}"
            f"  LIBERO R@1 {np.mean(r1s):.3f} R@5 {np.mean(r5s):.3f}"
            f"  ({n_tot} images over {len(r2s)} shards)")
        del tower
        torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"embodiments": embs, "results": results}, indent=2))
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
