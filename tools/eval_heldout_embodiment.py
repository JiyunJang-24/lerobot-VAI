#!/usr/bin/env python
"""Layer 1 of the held-out embodiment evaluation: does the VISION BACKBONE alone generalise?

Scores a SigLIP tower on embodiments it never saw during contrastive pre-training. No policy, no
VLM, no action -- this isolates the representation question from everything downstream, which is
the point: if the representation does not transfer, nothing built on top of it will.

Every number here is computed on HELD-OUT embodiments only. That is the difference from
tools/compare_siglip_all_checkpoints.py and tools/probe_embodiment_invariance.py, which score the
same embodiments the encoder was trained on and therefore cannot answer "does this generalise".

THREE METRICS, deliberately not one:

  A/B/C pair split      A = same pose, different embodiment   (should be HIGH)
                        B = different pose, same embodiment   (should be LOW)
                        C = different pose, different embodiment
                        A alone cannot distinguish invariance from collapse; B and C can.

  pose retrieval        query = a held-out embodiment's image; gallery = images of TRAINING
                        embodiments. Does the nearest neighbour share the query's canonical pose?
                        This is the honest one: its form is nothing like the contrastive objective,
                        so a high score cannot come from having memorised the training loss. Chance
                        is 1/n_poses.

  embodiment leakage    a linear probe trying to predict WHICH embodiment an image shows, from the
                        frozen features. An invariant representation should make this HARD. Reported
                        as accuracy against a chance baseline -- if the encoder is invariant, this
                        drops toward chance; if it just memorised, it stays high.

The dataset (eef_pairs/56combo_*) stores one canonical EEF pose per EPISODE and labels each frame
with observation.embodiment_index, so "held out" is a clean split on that column and "same pose"
is exactly "same episode".

    python tools/eval_heldout_embodiment.py \\
        --tower outputs/siglip_pretrain/both_cmean_eefattn_6views_b16/vision_tower.safetensors \\
        --holdout-frac 0.25
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

VLM_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEFAULT_DATA = Path("/dataset/jiyun/dataset_git/eef_pairs/56combo_48_bg12_closed")


def log(msg: str) -> None:
    print(f"[heldout_eval] {msg}", flush=True)


def load_tower(path, device):
    from transformers import AutoModelForImageTextToText

    tower = AutoModelForImageTextToText.from_pretrained(VLM_MODEL, dtype=torch.float32).model.vision_model
    if path:
        from safetensors.torch import load_file

        tower.load_state_dict(load_file(str(path)), strict=True)
    return tower.to(device).eval()


def build_index(root: Path, poses: int, per_pose: int, seed: int):
    """Sample (episode, embodiment, frame-within-episode) rows straight from the parquet.

    Reads the columns only -- the images come later, and only for the rows actually sampled, so
    this stays cheap even though the export has ~80k frames.
    """
    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    frames = []
    for path in files:
        frame = pd.read_parquet(
            path, columns=["episode_index", "frame_index", "observation.embodiment_index"]
        )
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table = table.rename(columns={"observation.embodiment_index": "embodiment"})

    rng = np.random.default_rng(seed)
    episodes = np.sort(table.episode_index.unique())[:poses]
    rows = []
    for episode in episodes:
        block = table[table.episode_index == episode]
        for embodiment, group in block.groupby("embodiment"):
            take = group.sample(n=min(per_pose, len(group)), random_state=int(rng.integers(1 << 30)))
            for _, row in take.iterrows():
                rows.append((int(episode), int(embodiment), int(row.frame_index)))
    return pd.DataFrame(rows, columns=["pose", "embodiment", "frame"])


@torch.no_grad()
def encode_rows(tower, dataset, index, device, chunk=16):
    """Mean-pooled features for each sampled row, in dataset order."""
    from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad

    key = "observation.images.agentview_right"
    # The dataset is flat: global row = episode start + frame index.
    episodes = dataset.meta_episodes
    starts = {int(e): int(s) for e, s in zip(episodes["episode_index"], episodes["dataset_from_index"],
                                             strict=False)} if "episode_index" in episodes else None

    images = []
    for _, row in index.iterrows():
        global_index = (starts[row.pose] if starts else int(episodes["dataset_from_index"][row.pose])) + row.frame
        item = dataset[int(global_index)]
        image = item[key]
        image = image[-1] if image.ndim == 4 else image
        images.append(resize_with_pad(image.unsqueeze(0), 512, 512, pad_value=0)[0] * 2.0 - 1.0)

    feats = []
    stack = torch.stack(images)
    for piece in stack.split(chunk):
        out = tower(pixel_values=piece.to(device), patch_attention_mask=None).last_hidden_state
        feats.append(out.mean(dim=1).cpu())
    return torch.cat(feats).float()


def pair_metrics(feats, poses, embodiments, center=True):
    """A / B / C means over the pair types, on whichever subset is passed in."""
    f = feats.clone()
    if center:
        f = f - f.mean(dim=0, keepdim=True)
    f = F.normalize(f, dim=-1)
    sim = f @ f.T
    same_pose = poses[:, None] == poses[None, :]
    same_emb = embodiments[:, None] == embodiments[None, :]
    eye = torch.eye(len(f), dtype=torch.bool)

    a = sim[same_pose & ~same_emb]
    b = sim[~same_pose & same_emb]
    c = sim[~same_pose & ~same_emb]
    return {
        "A_same_pose_diff_emb": a.mean().item(),
        "B_diff_pose_same_emb": b.mean().item(),
        "C_diff_pose_diff_emb": c.mean().item(),
        "gap_A_minus_B": (a.mean() - b.mean()).item(),
    }


def pose_retrieval(query_feats, query_poses, gallery_feats, gallery_poses, topk=(1, 5)):
    """Query = held-out embodiment images, gallery = training-embodiment images.

    Correct when the retrieved gallery image shares the query's canonical pose. Nothing about this
    resembles the contrastive loss's own form, so it cannot be satisfied by memorising it.
    """
    q = F.normalize(query_feats, dim=-1)
    g = F.normalize(gallery_feats, dim=-1)
    sim = q @ g.T
    order = sim.argsort(dim=1, descending=True)
    out = {}
    for k in topk:
        hit = (gallery_poses[order[:, :k]] == query_poses[:, None]).any(dim=1)
        out[f"pose_retrieval_top{k}"] = hit.float().mean().item()
    out["pose_retrieval_chance"] = 1.0 / len(torch.unique(gallery_poses))
    return out


def embodiment_probe(feats, labels, seed=0, epochs=300):
    """How well can a linear probe recover WHICH embodiment this is? Lower = more invariant.

    Plain torch rather than sklearn: sklearn is not installed in this env, and a multinomial
    logistic regression on a few hundred rows is a dozen lines here.
    """
    classes = torch.unique(labels)
    if len(classes) < 2:
        return {}
    remap = {int(c): i for i, c in enumerate(classes.tolist())}
    y = torch.tensor([remap[int(v)] for v in labels])
    x = F.normalize(feats, dim=-1)

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(x), generator=generator)
    split = int(0.7 * len(x))
    tr, te = order[:split], order[split:]
    if len(te) == 0 or len(torch.unique(y[tr])) < 2:
        return {}

    # A probe fit on a handful of rows per class reports noise, not leakage. Refuse rather than
    # emit a number that looks like "perfectly invariant" when it only means "not enough data".
    per_class = len(tr) / len(classes)
    if per_class < 5:
        return {"embodiment_probe_acc": float("nan"), "embodiment_probe_chance": 1.0 / len(classes),
                "embodiment_probe_skipped": f"only {per_class:.1f} train rows per class"}

    probe = torch.nn.Linear(x.shape[1], len(classes))
    optimiser = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(epochs):
        optimiser.zero_grad()
        loss = F.cross_entropy(probe(x[tr]), y[tr])
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        acc = (probe(x[te]).argmax(dim=1) == y[te]).float().mean().item()
    return {"embodiment_probe_acc": acc, "embodiment_probe_chance": 1.0 / len(classes)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--repo-id", default=None, help="defaults to the data dir's name")
    ap.add_argument("--tower", action="append", default=[],
                    help="vision tower .safetensors; repeatable. Pretrained SigLIP is always included.")
    ap.add_argument("--holdout-frac", type=float, default=0.25)
    ap.add_argument("--holdout-embodiments", default=None,
                    help="explicit comma-separated embodiment indices to hold out (overrides --holdout-frac)")
    ap.add_argument("--poses", type=int, default=16, help="canonical poses (= episodes) to sample")
    ap.add_argument("--per-pose", type=int, default=1, help="frames per (pose, embodiment)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/heldout_embodiment_eval.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset

    repo_id = args.repo_id or args.data.name
    dataset = MultiLeRobotDataset(
        [repo_id], root=args.data.parent, delta_timestamps={repo_id: None},
        visual_cue_mode="vanilla", use_wrist_cam=False, use_state=True, cache_in_memory=False,
    )

    index = build_index(args.data, args.poses, args.per_pose, args.seed)
    all_embodiments = np.sort(index.embodiment.unique())
    if args.holdout_embodiments:
        holdout = np.array([int(v) for v in args.holdout_embodiments.split(",")])
    else:
        rng = np.random.default_rng(args.seed)
        n = max(1, int(round(len(all_embodiments) * args.holdout_frac)))
        holdout = np.sort(rng.choice(all_embodiments, size=n, replace=False))
    train_emb = np.setdiff1d(all_embodiments, holdout)

    log(f"{len(all_embodiments)} embodiments, {index.pose.nunique()} poses, {len(index)} sampled frames")
    log(f"held out {len(holdout)}: {holdout.tolist()}")
    log(f"training  {len(train_emb)}")

    is_holdout = torch.tensor(index.embodiment.isin(holdout).to_numpy())
    poses = torch.tensor(index.pose.to_numpy())
    embodiments = torch.tensor(index.embodiment.to_numpy())

    towers = [("pretrained (init)", None)] + [(Path(t).parent.name, t) for t in args.tower]
    results = {}
    for name, path in towers:
        tower = load_tower(path, args.device)
        feats = encode_rows(tower, dataset, index, args.device)
        del tower
        torch.cuda.empty_cache()

        row = {}
        row["heldout"] = pair_metrics(feats[is_holdout], poses[is_holdout], embodiments[is_holdout])
        row["train"] = pair_metrics(feats[~is_holdout], poses[~is_holdout], embodiments[~is_holdout])
        row.update(pose_retrieval(feats[is_holdout], poses[is_holdout],
                                  feats[~is_holdout], poses[~is_holdout]))
        row.update(embodiment_probe(feats[is_holdout], embodiments[is_holdout], args.seed))
        results[name] = row

        h, t = row["heldout"], row["train"]
        log(f"{name:34s} held-out A {h['A_same_pose_diff_emb']:+.4f} B {h['B_diff_pose_same_emb']:+.4f} "
            f"gap {h['gap_A_minus_B']:+.4f} | train gap {t['gap_A_minus_B']:+.4f} | "
            f"retr@1 {row['pose_retrieval_top1']:.3f} | emb-probe {row.get('embodiment_probe_acc', float('nan')):.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"holdout": holdout.tolist(), "train": train_emb.tolist(), "results": results}, indent=2))
    log(f"wrote {args.out}")

    print()
    log("read it like this:")
    log("  held-out gap  > 0 and close to the train gap -> invariance transfers to unseen embodiments")
    log("  retr@1 >> chance                             -> pose is recoverable across embodiment change")
    log("  emb-probe near chance                        -> embodiment identity has been suppressed")
    log("  (high train gap + low held-out gap = memorised the training embodiments)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
