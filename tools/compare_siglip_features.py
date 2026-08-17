#!/usr/bin/env python
"""Compare what SigLIP extracts from the visual-robust views, before and after policy training.

Motivation: the alignment objective reports ~0.996-1.0 cosine similarity between a frame's three
embodiment renders and barely moves during training, which invites the reading "the features are
already identical, so there is nothing to align". That reading conflates two different things:

  * the ABSOLUTE cosine between two mean-pooled deep features, which is ~1 for almost any pair
    because the representation lives in a narrow cone (anisotropy), and
  * the GAP between positive pairs (same frame, different embodiment) and negative pairs (different
    frames), which is what actually says whether the representation encodes the scene rather than
    the robot.

So this script measures both, for two encoders:
  A. the pretrained SmolVLM2 vision tower (the initialisation every run starts from)
  B. the vision tower from a finished policy checkpoint

and reports them raw AND after centering (subtracting the batch mean), which removes the shared
component that pins the raw cosines near 1.

Everything runs on CPU so it is safe next to a live training job.

Usage:
    python tools/compare_siglip_features.py \
        --checkpoint outputs/train/.../checkpoints/050000/pretrained_model \
        --root dataset_git/visual_robust_new_barx/new_barx \
        --out outputs/siglip_feature_analysis
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad  # noqa: E402
from lerobot.scripts.lerobot_train_with_visual_robust import (  # noqa: E402
    SameEpisodeBatchSampler,
    _select_visual_robust_image_keys,
)


def log(msg: str) -> None:
    print(f"[compare_siglip_features] {msg}", flush=True)


@torch.no_grad()
def encode(vision_model, images: torch.Tensor, chunk: int = 8) -> torch.Tensor:
    """Same path the trainer's auxiliary loss uses: SigLIP -> mean over patch tokens."""
    outs = []
    for part in images.split(chunk, dim=0):
        outs.append(
            vision_model(pixel_values=part.to(dtype=vision_model.dtype), patch_attention_mask=None)
            .last_hidden_state.mean(dim=1)
            .float()
        )
    return torch.cat(outs, dim=0)


def pos_neg(sim: np.ndarray, n_frames: int, n_views: int):
    """Split a similarity matrix into positive pairs (same frame) and negative pairs (different)."""
    frame_of = np.repeat(np.arange(n_frames), n_views)
    same = frame_of[:, None] == frame_of[None, :]
    off_diag = ~np.eye(len(sim), dtype=bool)
    return sim[same & off_diag], sim[~same]


def stats(feats: torch.Tensor, n_frames: int, n_views: int, center: bool):
    f = feats - feats.mean(dim=0, keepdim=True) if center else feats
    f = F.normalize(f, dim=-1)
    sim = (f @ f.T).numpy()
    p, n = pos_neg(sim, n_frames, n_views)
    return sim, p, n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "dataset_git/visual_robust_new_barx/new_barx")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/siglip_feature_analysis")
    parser.add_argument("--batch-size", type=int, default=8, help="frames per batch (same episode)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    repo_ids = sorted(f"{p.parent.parent.parent.name}/lerobot" for p in args.root.glob("*/lerobot/meta/info.json"))
    ds = MultiLeRobotDataset(
        repo_ids,
        root=args.root,
        delta_timestamps={r: None for r in repo_ids},
        visual_cue_mode="vanilla",
        use_wrist_cam=False,
        use_state=True,
        cache_in_memory=False,
    )
    sampler = SameEpisodeBatchSampler(ds.meta_episodes, batch_size=args.batch_size, shuffle=True)
    indices = next(iter(sampler))
    items = [ds[i] for i in indices]
    batch = {k: torch.stack([it[k] for it in items]) for k in items[0] if torch.is_tensor(items[0][k])}

    keys = _select_visual_robust_image_keys(batch, image_prefix="observation.image.")
    n_frames, n_views = len(items), len(keys)
    log(f"{n_frames} frames x {n_views} views: {[k.rsplit('.', 1)[-1] for k in keys]}")

    # Preprocess exactly as the policy does, so the features are the ones training actually sees.
    imgs = []
    for i in range(n_frames):
        for k in keys:
            img = batch[k][i]
            img = img[-1] if img.ndim == 4 else img
            img = resize_with_pad(img.unsqueeze(0), 512, 512, pad_value=0)
            imgs.append(img[0] * 2.0 - 1.0)
    images = torch.stack(imgs)
    log(f"image tensor: {tuple(images.shape)}")

    from transformers import AutoModelForImageTextToText

    log("loading pretrained SmolVLM2 vision tower ...")
    pre = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", dtype=torch.float32
    ).model.vision_model.eval()

    log(f"loading fine-tuned vision tower from {args.checkpoint} ...")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(str(args.checkpoint))
    tuned = policy.model.vlm_with_expert.get_vlm_model().vision_model.float().eval()

    models = {"pretrained (init)": pre, "fine-tuned (baseline ckpt)": tuned}
    feats = {}
    for name, m in models.items():
        log(f"encoding with {name} ...")
        feats[name] = encode(m, images)
        log(f"  features {tuple(feats[name].shape)}  norm mean={feats[name].norm(dim=-1).mean():.3f}")

    # ---- figure 1: similarity matrices, raw and centered -----------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    summary = {}
    for col, (name, f) in enumerate(feats.items()):
        for row, center in enumerate([False, True]):
            sim, p, n = stats(f, n_frames, n_views, center)
            summary[(name, center)] = (p.mean(), n.mean(), p.mean() - n.mean())
            ax = axes[row][col]
            im = ax.imshow(sim, cmap="viridis", vmin=sim.min(), vmax=1.0)
            ax.set_title(
                f"{name}\n{'centered' if center else 'raw'}  "
                f"pos={p.mean():.4f} neg={n.mean():.4f} gap={p.mean() - n.mean():+.4f}",
                fontsize=9,
            )
            for b in range(1, n_frames):
                ax.axhline(b * n_views - 0.5, color="w", lw=0.6)
                ax.axvline(b * n_views - 0.5, color="w", lw=0.6)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        "SigLIP mean-pooled feature cosine similarity\n"
        f"{n_frames} frames x {n_views} embodiment renders; white lines separate frames "
        "(diagonal blocks = positives)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p1 = args.out / "similarity_matrices.png"
    fig.savefig(p1, dpi=120); plt.close(fig)
    log(f"wrote {p1}")

    # ---- figure 2: positive vs negative distributions ---------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for col, (name, f) in enumerate(feats.items()):
        for row, center in enumerate([False, True]):
            _, p, n = stats(f, n_frames, n_views, center)
            ax = axes[row][col]
            bins = np.linspace(min(p.min(), n.min()), 1.0, 40)
            ax.hist(n, bins=bins, alpha=0.6, label=f"negative (diff frame)  μ={n.mean():.4f}", color="tab:red")
            ax.hist(p, bins=bins, alpha=0.6, label=f"positive (same frame)  μ={p.mean():.4f}", color="tab:blue")
            ax.set_title(f"{name} — {'centered' if center else 'raw'}  gap={p.mean() - n.mean():+.4f}", fontsize=9)
            ax.legend(fontsize=7)
            ax.set_xlabel("cosine similarity", fontsize=8)
    fig.suptitle(
        "Positive vs negative cosine similarity\n"
        "the alignment loss only pushes the blue mean to 1; what matters for the representation is "
        "the blue-red GAP",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    p2 = args.out / "pos_neg_distributions.png"
    fig.savefig(p2, dpi=120); plt.close(fig)
    log(f"wrote {p2}")

    # ---- console summary --------------------------------------------------------------------------
    log("")
    log(f"{'encoder':28s} {'space':10s} {'pos':>9s} {'neg':>9s} {'gap':>9s}")
    for (name, center), (pm, nm, gap) in summary.items():
        log(f"{name:28s} {'centered' if center else 'raw':10s} {pm:9.4f} {nm:9.4f} {gap:+9.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
