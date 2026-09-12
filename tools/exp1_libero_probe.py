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

    all_images, all_meta, env_by_shard = [], [], {}
    for label, shard in shards():
        imgs, meta, env = load_rows(label, shard, embs, args.per_shard, rng)
        env_by_shard[label] = env
        all_images.append(imgs)
        all_meta += meta
        log(f"{label}: {len(imgs)} images, env_state {env.shape}")
    images = np.concatenate(all_images)
    shard_ids = np.array([m[0] for m in all_meta])
    row_ids = np.array([m[1] for m in all_meta])
    emb_ids = np.array([embs.index(m[2]) for m in all_meta])
    state_ids = np.array([hash((s, r)) % (2**31) for s, r in zip(shard_ids, row_ids, strict=True)])

    # object target: environment_state for that row, z-scored per shard so shards with a larger
    # world scale do not dominate the regression
    # LIBERO suites describe different numbers of objects, so environment_state has a different
    # width per shard. Truncating to the common prefix keeps the target well-defined; the
    # regression is over whatever object state all suites share.
    dim = min(v.shape[1] for v in env_by_shard.values())
    log(f"environment_state widths {[v.shape[1] for v in env_by_shard.values()]} -> using {dim}")
    target = []
    for s, r in zip(shard_ids, row_ids, strict=True):
        env = env_by_shard[s][:, :dim]
        target.append((env[r] - env.mean(0)) / (env.std(0) + 1e-6))
    target = np.stack(target)

    results = {}
    for ckpt in args.checkpoints:
        f = ckpt / "vision_tower.safetensors"
        tower = load_tower(str(f) if f.exists() else "", device).eval()
        feats = embed(tower, images, device)

        idx = np.random.default_rng(args.seed).permutation(len(feats))
        cut = int(0.7 * len(idx))
        tr, te = idx[:cut], idx[cut:]
        w = ridge(feats[tr], target[tr], alpha=10.0)
        pred = apply_ridge(w, feats[te])
        ss_res = ((pred - target[te]) ** 2).sum(0)
        ss_tot = ((target[te] - target[te].mean(0)) ** 2).sum(0) + 1e-9
        r2 = float(np.mean(1 - ss_res / ss_tot))

        z = centred(feats)
        sim = np.where(emb_ids[:, None] == emb_ids[None, :], -np.inf, z @ z.T)
        order = np.argsort(-sim, axis=1)
        hit = state_ids[order] == state_ids[:, None]
        results[ckpt.name] = {
            "object_state_r2": r2,
            "retrieval_R@1": float(hit[:, 0].mean()),
            "retrieval_R@5": float(hit[:, :5].any(1).mean()),
            "n": int(len(feats)),
        }
        log(f"{ckpt.name}: object R^2 {r2:+.3f}  LIBERO R@1 {results[ckpt.name]['retrieval_R@1']:.3f}"
            f"  R@5 {results[ckpt.name]['retrieval_R@5']:.3f}")
        del tower
        torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"embodiments": embs, "results": results}, indent=2))
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
