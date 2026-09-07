#!/usr/bin/env python
"""What does the contrastive backbone see, and how does a held-out robot differ from a trained one?

Produces the figures behind the write-up:
  1. an image grid -- the same canonical EEF poses rendered on TRAINED robots and on robots the
     tower was never trained on, so the visual difference is on the page rather than asserted
  2. a PCA of the features, coloured by pose and marked by train/held-out. If the tower encodes
     pose and ignores identity, points cluster by COLOUR and the two marker shapes sit on top of
     each other -- that is the whole claim in one picture
  3. a per-embodiment breakdown, so "held-out works" is not hiding one robot that fails
  4. a similarity matrix over poses

Everything is measured on 56combo_48_bg12_closed, the corpus this tower trained on, with the
holdout list read from the checkpoint's own pretrain_info.json rather than re-chosen here.

    python tools/visualize_embodiment_representation.py
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad  # noqa: E402

ROOT = Path("/dataset/jiyun/dataset_git/eef_pairs")
SUBSET = "56combo_48_bg12_closed"
KEY = "observation.images.agentview_right"


def log(msg: str) -> None:
    print(f"[viz] {msg}", flush=True)


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
    ap.add_argument("--poses", type=int, default=16)
    ap.add_argument("--per-group", type=int, default=8)
    ap.add_argument("--grid-poses", type=int, default=4)
    ap.add_argument("--grid-emb", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs")
    args = ap.parse_args()

    device = torch.device("cuda")
    info = json.loads((args.checkpoint / "pretrain_info.json").read_text())
    heldout = sorted(int(x) for x in info["args"]["holdout_embodiments"].split(","))
    train = sorted(info["train_embodiments"])
    log(f"{len(train)} trained / {len(heldout)} held-out embodiments")

    table = build_table()
    # hold every nuisance axis fixed: only pose and robot vary
    table = table[(table.background == table.background.min()) & (table.color == 0)]
    table = table[table.view == sorted(table.view.unique())[0]]
    poses = sorted(table.pose.unique())[: args.poses]

    rng = np.random.default_rng(args.seed)
    dataset = LeRobotDataset(f"eef_pairs/{SUBSET}", root=ROOT / SUBSET)

    rows, meta = [], []
    for pose in poses:
        sub = table[table.pose == pose]
        for group, members in (("trained", train), ("held-out", heldout)):
            avail = sorted(set(sub.embodiment.unique()) & set(members))
            for emb in rng.choice(avail, size=min(args.per_group, len(avail)), replace=False):
                r = sub[sub.embodiment == emb].iloc[0]
                rows.append(int(r["row"]))
                meta.append((int(pose), int(emb), group,
                             np.asarray(r["observation.state"], dtype=np.float64)[:3]))
    log(f"decoding {len(rows)} frames ...")
    images = torch.empty(len(rows), 3, 180, 320, dtype=torch.uint8)
    for i, r in enumerate(rows):
        img = dataset[r][KEY]
        img = img[-1] if img.ndim == 4 else img
        images[i] = (img * 255).round().clamp(0, 255).to(torch.uint8)

    pose_ids = np.array([m[0] for m in meta])
    emb_ids = np.array([m[1] for m in meta])
    groups = np.array([m[2] for m in meta])

    # ---------- 1. the image grid -------------------------------------------------------------
    # Only ~21% of (embodiment, pose, background, view, colour) combinations are rendered, so the
    # grid must be built from embodiments that actually EXIST at the chosen poses -- picking at
    # random leaves most cells blank.
    have = {}
    for pose in poses:
        have[pose] = set(table[table.pose == pose].embodiment.unique())
    gp = sorted(poses, key=lambda p: -len(have[p]))[: args.grid_poses]
    common = set.intersection(*(have[p] for p in gp))
    picked = {}
    for group, members in (("trained", train), ("held-out", heldout)):
        avail = sorted(common & set(members))
        if len(avail) < args.grid_emb:   # relax to per-pose coverage if the intersection is thin
            avail = sorted({e for p in gp for e in have[p]} & set(members))
        picked[group] = rng.choice(avail, size=min(args.grid_emb, len(avail)), replace=False)
    log(f"grid: poses {gp}, {len(common)} embodiments common to all of them")
    ncol = args.grid_emb * 2
    fig, axes = plt.subplots(len(gp), ncol, figsize=(2.05 * ncol, 1.35 * len(gp)),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.06})
    # Decode the grid cells directly rather than hoping the random feature sample happens to
    # contain these (embodiment, pose) pairs -- it mostly does not, which leaves the grid blank.
    def frame_for(pose, emb):
        hit = table[(table.pose == pose) & (table.embodiment == emb)]
        if not len(hit):
            return None
        img = dataset[int(hit.iloc[0]["row"])][KEY]
        img = img[-1] if img.ndim == 4 else img
        return (img * 255).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()

    for r, pose in enumerate(gp):
        col = 0
        for group in ("trained", "held-out"):
            for emb in picked[group]:
                ax = axes[r, col]
                arr = frame_for(pose, int(emb))
                if arr is not None:
                    ax.imshow(arr)
                ax.set_xticks([]); ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_color("#b3153b" if group == "held-out" else "#3d5573")
                    side.set_linewidth(2.0)
                if r == 0:
                    ax.set_title(f"emb {emb}", fontsize=8.5,
                                 color="#b3153b" if group == "held-out" else "#3d5573")
                if col == 0:
                    ax.set_ylabel(f"pose {pose}", fontsize=9)
                col += 1
    fig.suptitle("Same canonical EEF poses (rows) on TRAINED robots (blue) and robots the tower "
                 "NEVER saw (red)\nEvery image in a row is the identical end-effector state",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    grid_path = args.out_dir / "embodiment_grid.png"
    fig.savefig(grid_path, dpi=120, bbox_inches="tight")
    log(f"wrote {grid_path}")

    # ---------- 2 & 3. features, PCA, per-embodiment --------------------------------------------
    results = {"checkpoint": args.checkpoint.name, "n_train": len(train), "n_heldout": len(heldout),
               "towers": {}}
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    for ax, (name, path) in zip(axes, (
            ("contrastive (all4_n42_all)", str(args.checkpoint / "vision_tower.safetensors")),
            ("stock SigLIP (control)", "")), strict=True):
        tower = load_tower(path, device)
        feats = features(tower, images, device)
        centred = F.normalize(feats - feats.mean(0, keepdim=True), dim=-1)
        sim = (centred @ centred.T).numpy()

        # PCA on the centred features
        u, s, _ = torch.pca_lowrank(centred - centred.mean(0, keepdim=True), q=2)
        xy = (u * s).numpy()
        cmap = plt.get_cmap("turbo")
        for group, marker, size in (("trained", "o", 46), ("held-out", "^", 78)):
            m = groups == group
            ax.scatter(xy[m, 0], xy[m, 1], c=[cmap(p / max(poses)) for p in pose_ids[m]],
                       marker=marker, s=size, edgecolors="black" if group == "held-out" else "none",
                       linewidths=0.6, alpha=0.9, label=f"{group} embodiment")
        ax.set_title(f"{name}\ncolour = EEF pose, triangle = robot never seen", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=9, loc="upper right")

        per_emb = {}
        for emb in sorted(set(emb_ids.tolist())):
            mine = emb_ids == emb
            others = ~mine
            same = (pose_ids[:, None] == pose_ids[None, :])
            m = np.outer(mine, others) & same
            if m.sum():
                per_emb[int(emb)] = {
                    "group": "held-out" if emb in heldout else "trained",
                    "same_pose_cross_embodiment": float(sim[m].mean()),
                }
        results["towers"][name] = {
            "per_embodiment": per_emb,
            "pca_explained": [float(x) for x in (s**2 / (s**2).sum()).tolist()],
        }
        worst = sorted(per_emb.items(), key=lambda kv: kv[1]["same_pose_cross_embodiment"])[:3]
        log(f"{name}: weakest embodiments {[(k, round(v['same_pose_cross_embodiment'], 3)) for k, v in worst]}")
        del tower
        torch.cuda.empty_cache()
    fig.suptitle("Feature space: does a held-out robot land with its POSE or with its own kind?",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pca_path = args.out_dir / "embodiment_pca.png"
    fig.savefig(pca_path, dpi=115, bbox_inches="tight")
    log(f"wrote {pca_path}")

    (args.out_dir / "embodiment_representation.json").write_text(json.dumps(results, indent=2))
    log("wrote embodiment_representation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
