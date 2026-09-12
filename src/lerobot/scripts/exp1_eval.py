#!/usr/bin/env python
"""Experiment 1 evaluation: retrieval, three probes, and a scene control.

Every number is computed from FROZEN features of a finished encoder. Nothing here reuses a
training loss, and the EEF probe is refit from scratch rather than reading the auxiliary head,
so a method cannot score well on metric 2 merely by having trained a head.

  1 CROSS-EMBODIMENT STATE RETRIEVAL   query one embodiment, gallery = every OTHER embodiment.
                                       Correct = same (scene, frame). R@1, R@5, and the median
                                       EEF distance of the top-1 hit in cm, because a miss that
                                       lands 1 cm away is not the same failure as one 30 cm away.
  2 EEF PROBE                          ridge regression, features -> xyz and 6D rotation.
                                       Fit on seen/seen, applied to all four groups.
  3 EMBODIMENT PROBE                   logistic probe -> embodiment id and arm id. Diagnostic.
  4 SCENE PROBE                        logistic probe -> scene id, on the seen-scene groups.
                                       Object identity is 1:1 with scene in this corpus, so an
                                       "object" probe here would be a relabelled scene probe;
                                       the honest object metric lives in the LIBERO corpus.
  5 SCENE CONTROL                      same-scene/different-state similarity. A tower that
                                       matches on background scores high here and its retrieval
                                       number means nothing; this is what caught stock SigLIP
                                       posing as pose-aware on the real Jaco data.

All similarities are CENTRED (batch mean removed before normalising). SigLIP features are
anisotropic enough that raw cosines sit near 0.95 for unrelated images.

    python -m lerobot.scripts.exp1_eval --checkpoint outputs/exp1/C --out outputs/exp1/C/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.scripts.exp1_data import Exp1Config, Exp1Dataset  # noqa: E402
from lerobot.scripts.exp1_train import load_tower, prepare, quat_to_6d  # noqa: E402

GROUPS = ["seen_scene/seen_emb", "unseen_scene/seen_emb",
          "seen_scene/unseen_emb", "unseen_scene/unseen_emb"]


def log(msg: str) -> None:
    print(f"[eval] {msg}", flush=True)


@torch.no_grad()
def embed(tower, dataset, items, device, batch=12) -> np.ndarray:
    out = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        imgs = np.stack([dataset.cache.image(s, t, e) for s, t, e in chunk])
        x = prepare(imgs, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tok = tower(pixel_values=x, patch_attention_mask=None).last_hidden_state
        out.append(tok.float().mean(1).cpu())
    return torch.cat(out).numpy()


def centred(x: np.ndarray) -> np.ndarray:
    z = x - x.mean(0, keepdims=True)
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)


def retrieval(feats, state_ids, emb_ids, eef_xyz, ks=(1, 5)) -> dict:
    """Query each row; gallery is every row from a DIFFERENT embodiment."""
    z = centred(feats)
    sim = z @ z.T
    same_emb = emb_ids[:, None] == emb_ids[None, :]
    sim = np.where(same_emb, -np.inf, sim)
    order = np.argsort(-sim, axis=1)
    correct = state_ids[order] == state_ids[:, None]
    res = {f"R@{k}": float(correct[:, :k].any(1).mean()) for k in ks}
    top1 = order[:, 0]
    dist_cm = np.linalg.norm(eef_xyz - eef_xyz[top1], axis=1) * 100
    res["top1_eef_cm_median"] = float(np.median(dist_cm))
    # the median over ALL queries is 0 whenever R@1 > 0.5, which hides how bad the misses are
    miss = ~correct[:, 0]
    res["miss_eef_cm_median"] = float(np.median(dist_cm[miss])) if miss.any() else 0.0
    res["n_query"] = int(len(feats))
    return res


def scene_control(feats, state_ids, scene_ids, emb_ids) -> dict:
    z = centred(feats)
    sim = z @ z.T
    eye = np.eye(len(z), dtype=bool)
    same_state = (state_ids[:, None] == state_ids[None, :]) & ~eye
    diff_emb = emb_ids[:, None] != emb_ids[None, :]
    same_scene = scene_ids[:, None] == scene_ids[None, :]
    a = same_state & diff_emb
    b = same_scene & ~same_state & ~eye
    out = {
        "same_state_diff_emb": float(sim[a].mean()) if a.any() else float("nan"),
        "same_scene_diff_state": float(sim[b].mean()) if b.any() else float("nan"),
    }
    out["pose_discrimination"] = out["same_state_diff_emb"] - out["same_scene_diff_state"]
    return out


def ridge(x, y, alpha=1.0):
    x1 = np.concatenate([x, np.ones((len(x), 1))], 1)
    a = x1.T @ x1 + alpha * np.eye(x1.shape[1])
    a[-1, -1] = 0.0
    return np.linalg.solve(a, x1.T @ y)


def apply_ridge(w, x):
    return np.concatenate([x, np.ones((len(x), 1))], 1) @ w


def rot_error_deg(pred6, true6) -> np.ndarray:
    def to_mat(v):
        a, b = v[:, :3], v[:, 3:]
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b = b - (a * b).sum(1, keepdims=True) * a
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return np.stack([a, b, np.cross(a, b)], -1)
    r = np.einsum("nij,nkj->nik", to_mat(pred6), to_mat(true6))
    tr = np.clip((np.trace(r, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def logistic_probe(xtr, ytr, xte, yte, epochs=300, lr=1e-2, device="cuda") -> float:
    classes = sorted(set(ytr.tolist()) | set(yte.tolist()))
    remap = {c: i for i, c in enumerate(classes)}
    xtr_t = torch.tensor(xtr, dtype=torch.float32, device=device)
    xte_t = torch.tensor(xte, dtype=torch.float32, device=device)
    ytr_t = torch.tensor([remap[int(v)] for v in ytr], device=device)
    yte_t = torch.tensor([remap[int(v)] for v in yte], device=device)
    mu, sd = xtr_t.mean(0), xtr_t.std(0) + 1e-6
    xtr_t, xte_t = (xtr_t - mu) / sd, (xte_t - mu) / sd
    lin = torch.nn.Linear(xtr.shape[1], len(classes)).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(lin(xtr_t), ytr_t).backward()
        opt.step()
    with torch.no_grad():
        return float((lin(xte_t).argmax(1) == yte_t).float().mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", type=Path, default=Path("outputs/exp1/split.json"))
    ap.add_argument("--states-per-scene", type=int, default=0)
    ap.add_argument("--max-embodiments", type=int, default=0)
    # equalised gallery -- see Exp1Dataset.enumerate_group for why this is not optional
    ap.add_argument("--states-per-group", type=int, default=48)
    ap.add_argument("--embodiments-per-state", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    ckpt = args.checkpoint / "vision_tower.safetensors"
    tower = load_tower(str(ckpt) if ckpt.exists() else "", device).eval()
    log(f"tower: {ckpt if ckpt.exists() else 'stock'}")

    results = {"checkpoint": str(args.checkpoint), "groups": {}}
    store = {}
    for group in GROUPS:
        cfg = Exp1Config(split=args.split, seed=args.seed)
        data = Exp1Dataset(cfg, group)
        rng = np.random.default_rng(args.seed)
        embs = data.embodiments
        if args.max_embodiments and len(embs) > args.max_embodiments:
            embs = sorted(rng.choice(embs, args.max_embodiments, replace=False).tolist())
            data.embodiments = embs
        items = data.enumerate_group(args.states_per_scene, np.random.default_rng(args.seed),
                                     total_states=args.states_per_group,
                                     embodiments_per_state=args.embodiments_per_state)
        if not items:
            log(f"{group}: EMPTY, skipping")
            continue
        feats = embed(tower, data, items, device)

        scene_ids = np.array([s for s, _, _ in items])
        state_ids = np.array([s * 100000 + t for s, t, _ in items])
        emb_index = {e: i for i, e in enumerate(sorted({e for _, _, e in items}))}
        emb_ids = np.array([emb_index[e] for _, _, e in items])
        arm_index = {a: i for i, a in enumerate(sorted({e.split("_")[0] for _, _, e in items}))}
        arm_ids = np.array([arm_index[e.split("_")[0]] for _, _, e in items])
        # scene-relative: world xyz would let the probe win by recognising the kitchen
        eef = np.stack([data.cache.eef_rel(s, t, e) for s, t, e in items])

        results["groups"][group] = {
            "n_items": len(items), "n_scenes": len(set(scene_ids.tolist())),
            "n_embodiments": len(emb_index),
            "retrieval": retrieval(feats, state_ids, emb_ids, eef[:, :3]),
            "scene_control": scene_control(feats, state_ids, scene_ids, emb_ids),
        }
        store[group] = dict(feats=feats, eef=eef, scene=scene_ids, emb=emb_ids, arm=arm_ids)
        r = results["groups"][group]["retrieval"]
        n_states = len(set(state_ids.tolist()))
        results["groups"][group]["n_states"] = n_states
        log(f"{group:26s} n={len(items):5d} states={n_states:3d} "
            f"R@1 {r['R@1']:.3f} R@5 {r['R@5']:.3f} "
            f"miss {r['miss_eef_cm_median']:.1f}cm "
            f"posedisc {results['groups'][group]['scene_control']['pose_discrimination']:+.3f}")

    # ---- probes, all fit on seen/seen and applied everywhere -------------------------------
    base = store.get("seen_scene/seen_emb")
    if base is not None:
        xtr = base["feats"]
        ytr = np.concatenate([base["eef"][:, :3],
                              quat_to_6d(torch.tensor(base["eef"][:, 3:7])).numpy()], 1)
        w = ridge(xtr, ytr, alpha=10.0)
        results["eef_probe"] = {}
        for group, s in store.items():
            pred = apply_ridge(w, s["feats"])
            true = np.concatenate([s["eef"][:, :3],
                                   quat_to_6d(torch.tensor(s["eef"][:, 3:7])).numpy()], 1)
            results["eef_probe"][group] = {
                "trans_cm": float(np.linalg.norm(pred[:, :3] - true[:, :3], axis=1).mean() * 100),
                "rot_deg": float(rot_error_deg(pred[:, 3:], true[:, 3:]).mean()),
            }
            log(f"eef probe  {group:26s} {results['eef_probe'][group]['trans_cm']:.2f} cm "
                f"{results['eef_probe'][group]['rot_deg']:.1f} deg")

        # embodiment / arm probe: held out by ITEM within the group, so it measures how much
        # identity the features carry, not whether the split leaked
        results["identity_probe"] = {}
        for group, s in store.items():
            rng = np.random.default_rng(args.seed)
            idx = rng.permutation(len(s["feats"]))
            cut = int(0.7 * len(idx))
            tr, te = idx[:cut], idx[cut:]
            entry = {
                "embodiment_acc": logistic_probe(s["feats"][tr], s["emb"][tr],
                                                 s["feats"][te], s["emb"][te]),
                "embodiment_chance": 1.0 / len(set(s["emb"].tolist())),
                "arm_acc": logistic_probe(s["feats"][tr], s["arm"][tr],
                                          s["feats"][te], s["arm"][te]),
                "arm_chance": 1.0 / len(set(s["arm"].tolist())),
            }
            results["identity_probe"][group] = entry
            log(f"identity   {group:26s} emb {entry['embodiment_acc']:.3f} "
                f"(chance {entry['embodiment_chance']:.3f})  arm {entry['arm_acc']:.3f} "
                f"(chance {entry['arm_chance']:.3f})")

        # scene probe on the groups that have more than one scene
        results["scene_probe"] = {}
        for group, s in store.items():
            n_scene = len(set(s["scene"].tolist()))
            if n_scene < 2:
                continue
            rng = np.random.default_rng(args.seed)
            idx = rng.permutation(len(s["feats"]))
            cut = int(0.7 * len(idx))
            tr, te = idx[:cut], idx[cut:]
            acc = logistic_probe(s["feats"][tr], s["scene"][tr], s["feats"][te], s["scene"][te])
            results["scene_probe"][group] = {"acc": acc, "chance": 1.0 / n_scene,
                                             "n_scenes": n_scene}
            log(f"scene      {group:26s} {acc:.3f} (chance {1.0 / n_scene:.3f}, {n_scene} scenes)")

    out = args.out or (args.checkpoint / "metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
