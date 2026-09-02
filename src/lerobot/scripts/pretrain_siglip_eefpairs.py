#!/usr/bin/env python
"""Contrastive pre-training of the SigLIP tower on the eef_pairs exports.

Separate from pretrain_siglip_visual_robust.py because the data is shaped differently, not because
the objective differs. The visual-robust exports put every embodiment's render of one instant in
its OWN COLUMN of a single row, so a positive group is "several keys of one item". eef_pairs
instead stores one image per row and labels it with observation.embodiment_index, so a positive
group is "several ROWS sharing an episode" -- one canonical EEF pose per episode, which is exactly
what the dataset card promises.

Built for the scaling question: --n-embodiments trains on a subset of the available embodiments,
so a ladder (8 -> 16 -> 32 -> ...) answers "how much embodiment diversity is enough" directly, and
everything not trained on stays available as a held-out set for
tools/eval_heldout_embodiment.py.

    python src/lerobot/scripts/pretrain_siglip_eefpairs.py --n-embodiments 16 --steps 5000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad  # noqa: E402
from lerobot.scripts.lerobot_train_with_visual_robust import _supervised_contrastive_loss  # noqa: E402

VLM_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEFAULT_ROOT = Path("/dataset/jiyun/dataset_git/eef_pairs")
IMAGE_KEY = "observation.images.agentview_right"


def log(msg: str) -> None:
    print(f"[pretrain_eefpairs] {msg}", flush=True)


def build_row_table(root: Path, subsets: list[str]) -> "object":
    """(global_row, pose, embodiment) for every frame, read from the parquet.

    Reading the columns rather than the dataset keeps this cheap: the images are only touched for
    rows a batch actually samples.
    """
    import glob

    import pandas as pd

    tables = []
    offset = 0
    for subset in subsets:
        files = sorted(glob.glob(str(root / subset / "data" / "**" / "*.parquet"), recursive=True))
        frames = [
            pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.embodiment_index"])
            for f in files
        ]
        table = pd.concat(frames, ignore_index=True)
        table = table.rename(columns={"observation.embodiment_index": "embodiment"})
        # Global row index within this subset's flat dataset = episode start + frame index. The
        # episode start is the cumulative length of preceding episodes.
        lengths = table.groupby("episode_index").size().sort_index()
        starts = lengths.cumsum().shift(fill_value=0)
        table["row"] = table.episode_index.map(starts) + table.frame_index + offset
        table["subset"] = subset
        # Poses are per-subset; offset them so two subsets never share a "pose" label.
        table["pose"] = table.episode_index + (1000 * subsets.index(subset))
        offset += len(table)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def sample_batch(table, embodiments, poses_per_batch, views_per_pose, rng):
    """One contrastive batch: a few poses, several embodiments each.

    Same structure the loss expects -- rows sharing a pose are positives, rows of different poses
    are negatives -- just assembled from rows instead of columns.
    """
    available = table[table.embodiment.isin(embodiments)]
    poses = rng.choice(available.pose.unique(), size=poses_per_batch, replace=False)
    rows, labels = [], []
    for label, pose in enumerate(poses):
        block = available[available.pose == pose]
        take = block.sample(n=min(views_per_pose, len(block)), random_state=int(rng.integers(1 << 30)))
        rows += take.row.tolist()
        labels += [label] * len(take)
    return rows, torch.tensor(labels)


def load_images(dataset, rows, device):
    images = []
    for row in rows:
        item = dataset[int(row)]
        image = item[IMAGE_KEY]
        image = image[-1] if image.ndim == 4 else image
        images.append(resize_with_pad(image.unsqueeze(0), 512, 512, pad_value=0)[0] * 2.0 - 1.0)
    return torch.stack(images).to(device)


def encode(tower, images, chunk, grad=True):
    from contextlib import nullcontext

    out = []
    for piece in images.split(chunk):
        with nullcontext() if grad else torch.no_grad():
            out.append(tower(pixel_values=piece, patch_attention_mask=None).last_hidden_state)
    return torch.cat(out)


@torch.no_grad()
def quick_gap(tower, dataset, table, embodiments, rng, device, chunk, poses=8, views=6):
    """Positive/negative cosine gap on a fixed-ish sample -- a progress signal, not the evaluation.

    The real scoring is tools/eval_heldout_embodiment.py, which uses embodiments this never saw.
    """
    tower.eval()
    rows, labels = sample_batch(table, embodiments, poses, views, rng)
    feats = encode(tower, load_images(dataset, rows, device), chunk, grad=False).mean(dim=1).float()
    feats = feats - feats.mean(dim=0, keepdim=True)
    feats = F.normalize(feats, dim=-1)
    sim = feats @ feats.T
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool)
    tower.train()
    return (sim[same & ~eye].mean() - sim[~same].mean()).item()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--subsets", default="56combo_48_bg12_closed",
                    help="comma-separated dataset dirs under --root")
    ap.add_argument("--n-embodiments", type=int, default=0,
                    help="train on this many embodiments (0 = all). The rest stay held out.")
    ap.add_argument("--holdout-embodiments", default=None,
                    help="explicit comma-separated indices to EXCLUDE from training")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--poses-per-batch", type=int, default=8)
    ap.add_argument("--views-per-pose", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--encoder-chunk", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    table = build_row_table(args.root, subsets)
    all_emb = np.sort(table.embodiment.unique())

    if args.holdout_embodiments:
        holdout = np.array([int(v) for v in args.holdout_embodiments.split(",")])
        train_emb = np.setdiff1d(all_emb, holdout)
    elif args.n_embodiments and args.n_embodiments < len(all_emb):
        # Deterministic given the seed, so a ladder at 8/16/32 is nested-ish and comparable.
        train_emb = np.sort(rng.choice(all_emb, size=args.n_embodiments, replace=False))
        holdout = np.setdiff1d(all_emb, train_emb)
    else:
        train_emb, holdout = all_emb, np.array([], dtype=int)

    log(f"subsets {subsets}: {len(table)} rows, {table.pose.nunique()} poses, {len(all_emb)} embodiments")
    log(f"training on {len(train_emb)}: {train_emb.tolist()}")
    log(f"held out    {len(holdout)}: {holdout.tolist()}")

    dataset = MultiLeRobotDataset(
        subsets, root=args.root, delta_timestamps={s: None for s in subsets},
        visual_cue_mode="vanilla", use_wrist_cam=False, use_state=True, cache_in_memory=False,
    )

    from transformers import AutoModelForImageTextToText

    log(f"loading {VLM_MODEL} vision tower (fp32) ...")
    tower = AutoModelForImageTextToText.from_pretrained(
        VLM_MODEL, dtype=torch.float32
    ).model.vision_model.to(device)
    tower.train()
    reference = {n: p.detach().clone() for n, p in tower.named_parameters()}

    optimiser = torch.optim.AdamW(tower.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: min(1.0, (s + 1) / max(1, args.warmup))
    )

    history = []
    for step in range(1, args.steps + 1):
        rows, labels = sample_batch(table, train_emb, args.poses_per_batch, args.views_per_pose, rng)
        images = load_images(dataset, rows, device)
        feats = encode(tower, images, args.encoder_chunk).mean(dim=1)
        loss = _supervised_contrastive_loss(feats, labels.to(device), args.temperature)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(tower.parameters(), 10.0)
        optimiser.step()
        scheduler.step()

        if step % args.eval_every == 0 or step == args.steps:
            drift_num = sum((p.detach() - reference[n]).pow(2).sum() for n, p in tower.named_parameters())
            drift_den = sum(t.pow(2).sum() for t in reference.values())
            drift = float((drift_num / drift_den) ** 0.5)
            gap = quick_gap(tower, dataset, table, train_emb, rng, device, args.encoder_chunk)
            history.append({"step": step, "loss": float(loss), "train_gap": gap, "drift": drift})
            log(f"step {step:5d}  loss {float(loss):6.3f}  train gap {gap:+.4f}  drift {drift:.5f}  "
                f"|g| {float(grad_norm):.2f}")

    from safetensors.torch import save_file

    tower_path = out_dir / "vision_tower.safetensors"
    save_file({n: p.detach().cpu().contiguous() for n, p in tower.state_dict().items()}, tower_path)
    (out_dir / "pretrain_info.json").write_text(json.dumps({
        "args": vars(args), "train_embodiments": train_emb.tolist(),
        "holdout_embodiments": holdout.tolist(), "history": history,
    }, indent=2, default=str))
    log(f"wrote {tower_path}")
    log(f"held-out embodiments for evaluation: {holdout.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
