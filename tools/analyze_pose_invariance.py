#!/usr/bin/env python
"""Does the contrastive backbone give the SAME feature for the SAME EEF pose on a robot it has
never seen?

This is the property the dual-tower Diffusion Policy experiment is betting on, so it is worth
measuring directly rather than inferring from the training loss. The tower under test is the one
those runs load, `all4_n42_all`, and the stock SigLIP is the control -- without it a cosine of
0.9 means nothing, because SigLIP features are anisotropic enough that unrelated images already
sit near 0.99.

Everything is reported CENTRED as well as raw for that reason: subtracting the batch mean removes
the shared component that makes every raw cosine look high.

Four measurements:
  1. same pose across embodiments, split by whether each side was in training
  2. the contrast it has to beat -- same embodiment, different pose
  3. pose retrieval: give it a held-out robot's image, does the nearest TRAINING image share the
     pose? Reported as an EEF distance in cm as well as a hit rate, because "wrong" says nothing
     about how wrong
  4. how similarity decays with the actual distance between two poses

    python tools/analyze_pose_invariance.py --checkpoint outputs/siglip_pretrain/all4_n42_all
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad  # noqa: E402

ROOT = Path("/dataset/jiyun/dataset_git/eef_pairs")
SUBSET = "56combo_48_bg12_closed"


def log(msg: str) -> None:
    print(f"[pose_inv] {msg}", flush=True)


def load_tower(path: str, device):
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText

    vlm = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", dtype=torch.float32)
    tower = vlm.model.vision_model
    if path:
        tower.load_state_dict(load_file(path), strict=True)
    return tower.to(device).eval()


@torch.no_grad()
def features(tower, images, device, batch=16):
    out = []
    for piece in images.split(batch):
        x = piece.to(device=device, dtype=torch.float32) / 255.0
        x = resize_with_pad(x, 512, 512, pad_value=0) * 2.0 - 1.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(tower(pixel_values=x, patch_attention_mask=None).last_hidden_state.mean(1))
    return torch.cat(out).float().cpu()


def build_table():
    files = sorted(glob.glob(str(ROOT / SUBSET / "data" / "**" / "*.parquet"), recursive=True))
    import pandas as pd

    table = pd.concat([pd.read_parquet(f, columns=[
        "episode_index", "frame_index", "observation.embodiment_index",
        "observation.background_index", "observation.camera_view_index",
        "observation.color_variant_index", "observation.state"]) for f in files], ignore_index=True)
    table = table.rename(columns={"observation.embodiment_index": "embodiment",
                                  "observation.background_index": "background",
                                  "observation.camera_view_index": "view",
                                  "observation.color_variant_index": "color"})
    lengths = table.groupby("episode_index").size().sort_index()
    starts = lengths.cumsum().shift(fill_value=0)
    table["row"] = table.episode_index.map(starts) + table.frame_index
    table["pose"] = table.episode_index
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO_ROOT / "outputs/siglip_pretrain/all4_n42_all")
    ap.add_argument("--poses", type=int, default=24)
    ap.add_argument("--per-group", type=int, default=6, help="embodiments per group per pose")
    ap.add_argument("--vary", default="",
                    help="comma-separated nuisance axes allowed to differ between the two frames: "
                         "background, view, color. Empty holds all three fixed, which isolates "
                         "pose and robot; adding them shows what the invariance survives.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/pose_invariance.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    info = json.loads((args.checkpoint / "pretrain_info.json").read_text())
    heldout = set(int(x) for x in info["args"]["holdout_embodiments"].split(","))
    train = set(info["train_embodiments"])
    log(f"{args.checkpoint.name}: {len(train)} train / {len(heldout)} held-out embodiments")

    table = build_table()
    # By default hold background, camera view and colour FIXED, so the only thing differing
    # between two rows is the pose and the robot -- otherwise a low similarity could be the
    # backdrop rather than the thing being measured. --vary relaxes them one at a time.
    vary = {v for v in args.vary.split(",") if v}
    if "background" not in vary:
        table = table[table.background == table.background.min()]
    if "color" not in vary:
        table = table[table.color == 0]
    if "view" not in vary:
        table = table[table.view == sorted(table.view.unique())[0]]
    log(f"nuisance axes allowed to differ: {sorted(vary) or 'none (all fixed)'}")
    poses = sorted(table.pose.unique())[: args.poses]
    log(f"holding background/view/colour fixed; {len(poses)} poses")

    rng = np.random.default_rng(args.seed)
    dataset = LeRobotDataset(f"eef_pairs/{SUBSET}", root=ROOT / SUBSET)

    picks, meta = [], []
    for pose in poses:
        sub = table[table.pose == pose]
        for group, members in (("train", train), ("heldout", heldout)):
            available = sorted(set(sub.embodiment.unique()) & members)
            if not available:
                continue
            chosen = rng.choice(available, size=min(args.per_group, len(available)), replace=False)
            for emb in chosen:
                rows = sub[sub.embodiment == emb]
                row = rows.iloc[int(rng.integers(len(rows)))]
                picks.append(int(row["row"]))
                meta.append((int(pose), int(emb), group,
                             np.asarray(row["observation.state"], dtype=np.float64)[:3]))
    log(f"decoding {len(picks)} frames ...")
    images = torch.empty(len(picks), 3, 180, 320, dtype=torch.uint8)
    for i, row in enumerate(picks):
        img = dataset[row]["observation.images.agentview_right"]
        img = img[-1] if img.ndim == 4 else img
        images[i] = (img * 255).round().clamp(0, 255).to(torch.uint8)

    pose_ids = np.array([m[0] for m in meta])
    groups = np.array([m[2] for m in meta])
    xyz = np.stack([m[3] for m in meta])

    results = {"checkpoint": args.checkpoint.name, "n_frames": len(picks),
               "n_poses": len(poses), "vary": sorted(vary), "towers": {}}
    for name, path in (("contrastive (all4_n42_all)", str(args.checkpoint / "vision_tower.safetensors")),
                       ("stock SigLIP (control)", "")):
        tower = load_tower(path, device)
        feats = features(tower, images, device)
        raw = F.normalize(feats, dim=-1)
        cen = F.normalize(feats - feats.mean(0, keepdim=True), dim=-1)
        sim_raw = (raw @ raw.T).numpy()
        sim = (cen @ cen.T).numpy()
        n = len(feats)
        eye = np.eye(n, dtype=bool)

        same_pose = (pose_ids[:, None] == pose_ids[None, :]) & ~eye
        diff_pose = (pose_ids[:, None] != pose_ids[None, :])
        is_h = groups == "heldout"
        pair = {}
        for label, mask in (
            ("same pose, train x train", same_pose & (~is_h[:, None]) & (~is_h[None, :])),
            ("same pose, train x HELD-OUT", same_pose & (~is_h[:, None]) & is_h[None, :]),
            ("same pose, HELD-OUT x HELD-OUT", same_pose & is_h[:, None] & is_h[None, :]),
            ("DIFFERENT pose, same embodiment", diff_pose & ~eye),
        ):
            if mask.sum():
                pair[label] = {"centred": float(sim[mask].mean()),
                               "raw": float(sim_raw[mask].mean()), "n": int(mask.sum())}

        # retrieval: each held-out frame against the TRAINING frames only
        held_idx = np.flatnonzero(is_h)
        train_idx = np.flatnonzero(~is_h)
        scores = sim[np.ix_(held_idx, train_idx)]
        best = scores.argmax(1)
        hit = pose_ids[train_idx][best] == pose_ids[held_idx]
        err_cm = np.linalg.norm(xyz[train_idx][best] - xyz[held_idx], axis=1) * 100
        top5 = np.argsort(-scores, axis=1)[:, :5]
        hit5 = (pose_ids[train_idx][top5] == pose_ids[held_idx][:, None]).any(1)

        # decay: centred similarity as a function of the real distance between two poses
        pose_dist = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1) * 100
        bins = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 100)]
        decay = {}
        for lo, hi in bins:
            m = (pose_dist >= lo) & (pose_dist < hi) & ~eye
            if m.sum():
                decay[f"{lo}-{hi} cm"] = {"centred": float(sim[m].mean()), "n": int(m.sum())}

        results["towers"][name] = {
            "pairs": pair,
            "retrieval": {"top1": float(hit.mean()), "top5": float(hit5.mean()),
                          "median_pose_error_cm": float(np.median(err_cm)),
                          "mean_pose_error_cm": float(err_cm.mean()), "n": int(len(held_idx))},
            "decay_with_pose_distance": decay,
        }
        log(f"--- {name}")
        for k, v in pair.items():
            log(f"    {k:<34} centred {v['centred']:+.3f}  raw {v['raw']:+.3f}  (n={v['n']})")
        r = results["towers"][name]["retrieval"]
        log(f"    held-out -> nearest TRAINING frame: same pose top1 {r['top1']:.3f} "
            f"top5 {r['top5']:.3f}, median EEF error {r['median_pose_error_cm']:.1f} cm")
        del tower
        torch.cuda.empty_cache()

    args.out.write_text(json.dumps(results, indent=2))
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
