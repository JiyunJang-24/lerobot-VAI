#!/usr/bin/env python
"""Compare the SigLIP representation across every finished barx front-only checkpoint.

Answers, on identical data and with an identical measurement: does adding the visual-robust loss
change what the vision tower encodes, and does the objective (alignment vs contrastive) matter?

The headline number is the GAP:

    gap = mean cosine(positive pairs) - mean cosine(negative pairs)

  positive pair = two renders of the SAME frame with a different embodiment  (should be similar)
  negative pair = renders of DIFFERENT frames                                (should differ)

gap > 0 means the representation tracks the scene; gap < 0 means it tracks the robot. Both raw and
mean-centered cosines are reported, because deep features share a large common component that pins
every raw cosine near 1 and hides the structure.

Encoders are discovered automatically from outputs/train/*/*barx_frontonly*/checkpoints/050000.

Usage:
    python tools/compare_siglip_all_checkpoints.py --out outputs/siglip_feature_analysis
"""

import argparse
import json
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
    print(f"[compare_siglip_all] {msg}", flush=True)


@torch.no_grad()
def encode(vision_model, images, chunk=8):
    outs = []
    for part in images.split(chunk, dim=0):
        outs.append(
            vision_model(pixel_values=part.to(dtype=vision_model.dtype), patch_attention_mask=None)
            .last_hidden_state.mean(dim=1)
            .float()
        )
    return torch.cat(outs, dim=0)


def gap_stats(feats, n_frames, n_views, center):
    f = feats - feats.mean(0, keepdim=True) if center else feats
    f = F.normalize(f, dim=-1)
    sim = (f @ f.T).numpy()
    frame_of = np.repeat(np.arange(n_frames), n_views)
    same = frame_of[:, None] == frame_of[None, :]
    off = ~np.eye(len(sim), dtype=bool)
    p, n = sim[same & off], sim[~same]
    return sim, p.mean(), n.mean(), p.mean() - n.mean()


def discover(ckpt_glob: str):
    """Label each checkpoint by whichever auxiliary term it actually used.

    Every label must be distinct: `found` is keyed by it, so two runs sharing a label means one of
    them silently vanishes from the comparison and looks exactly like a checkpoint that was never
    trained. Five runs here carry no contrastive weight at all (plain baseline, frozen encoder, L2-SP,
    feature distillation, and two EEF-state variants), so the objective/weight pair alone is not
    enough to tell them apart.
    """
    found = {}
    for p in sorted(REPO_ROOT.glob(ckpt_glob)):
        cfg = json.loads((p / "train_config.json").read_text())
        ds, pol = cfg["dataset"], cfg["policy"]
        parts = []
        if ds.get("visual_robust_contrastive_weight"):
            parts.append(f"{ds.get('visual_robust_front_objective', 'vr')} w={ds['visual_robust_contrastive_weight']}")
        if pol.get("freeze_vision_encoder"):
            parts.append("frozen")
        if ds.get("vision_l2sp_weight"):
            parts.append(f"L2-SP w={ds['vision_l2sp_weight']}")
        if ds.get("vision_distill_weight"):
            parts.append(f"distill w={ds['vision_distill_weight']}")
        if ds.get("visual_robust_state_weight"):
            state = f"eef-state w={ds['visual_robust_state_weight']}"
            if ds.get("visual_robust_state_policy_weight"):
                state += f"+policy w={ds['visual_robust_state_policy_weight']}"
            parts.append(state)
        label = " / ".join(parts) if parts else "baseline (no aux)"
        if label in found:
            label = f"{label} [{p.parents[2].name[:5]}]"
        found[label] = p
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "dataset_git/visual_robust_new_barx/new_barx")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/siglip_feature_analysis")
    ap.add_argument("--ckpt-glob", type=str,
                    default="outputs/train/*/*barx_frontonly*/checkpoints/050000/pretrained_model")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-batches", type=int, default=3, help="average the gap over this many batches")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    repo_ids = sorted(f"{p.parent.parent.parent.name}/lerobot" for p in args.root.glob("*/lerobot/meta/info.json"))
    ds = MultiLeRobotDataset(
        repo_ids, root=args.root, delta_timestamps={r: None for r in repo_ids},
        visual_cue_mode="vanilla", use_wrist_cam=False, use_state=True, cache_in_memory=False,
    )
    sampler = SameEpisodeBatchSampler(ds.meta_episodes, batch_size=args.batch_size, shuffle=True)

    batches = []
    it = iter(sampler)
    for _ in range(args.n_batches):
        idx = next(it)
        items = [ds[i] for i in idx]
        batch = {k: torch.stack([x[k] for x in items]) for k in items[0] if torch.is_tensor(items[0][k])}
        keys = _select_visual_robust_image_keys(batch, image_prefix="observation.image.")
        imgs = []
        for i in range(len(items)):
            for k in keys:
                img = batch[k][i]
                img = img[-1] if img.ndim == 4 else img
                imgs.append(resize_with_pad(img.unsqueeze(0), 512, 512, pad_value=0)[0] * 2.0 - 1.0)
        batches.append((torch.stack(imgs), len(items), len(keys)))
    log(f"{args.n_batches} batches of {batches[0][1]} frames x {batches[0][2]} views")

    from transformers import AutoModelForImageTextToText
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    encoders = {}
    log("loading pretrained SmolVLM2 vision tower ...")
    encoders["pretrained (init)"] = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", dtype=torch.float32).model.vision_model.eval()

    for label, path in discover(args.ckpt_glob).items():
        log(f"loading {label} ...")
        encoders[label] = SmolVLAPolicy.from_pretrained(
            str(path)).model.vlm_with_expert.get_vlm_model().vision_model.float().eval()

    order = ["pretrained (init)", "baseline (no aux)"]
    order += sorted(k for k in encoders if k.startswith("alignment"))
    order += sorted(k for k in encoders if k.startswith("contrastive"))
    order = [k for k in order if k in encoders]
    # Anything the explicit ordering above does not name -- e.g. "baseline (no aux) frozen", or a
    # future l2sp label -- still goes in the report. Without this it would be loaded, encoded, and
    # then silently dropped, which looked exactly like the checkpoint not existing.
    order += [k for k in encoders if k not in order]

    results = {}
    for label in order:
        raws, cens, mats = [], [], None
        for images, nf, nv in batches:
            f = encode(encoders[label], images)
            sim, pr, nr, gr = gap_stats(f, nf, nv, center=False)
            _, pc, nc, gc = gap_stats(f, nf, nv, center=True)
            raws.append((pr, nr, gr)); cens.append((pc, nc, gc))
            if mats is None:
                mats = sim
        results[label] = {
            "raw": np.mean(raws, axis=0), "centered": np.mean(cens, axis=0), "sim": mats,
        }
        r, c = results[label]["raw"], results[label]["centered"]
        log(f"  {label:24s} raw gap={r[2]:+.4f}   centered gap={c[2]:+.4f}")

    # ---- figure: gap comparison ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(order))
    for ax, space in zip(axes, ["raw", "centered"]):
        gaps = [results[k][space][2] for k in order]
        colours = []
        for k in order:
            colours.append("0.55" if k.startswith("pretrained") else
                           "tab:purple" if "frozen" in k else
                           "tab:red" if "L2-SP" in k else
                           "tab:brown" if "distill" in k else
                           "tab:cyan" if "eef-state" in k else
                           "tab:orange" if k.startswith("baseline") else
                           "tab:blue" if k.startswith("alignment") else "tab:green")
        ax.bar(x, gaps, color=colours)
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace(" (init)", "\n(init)").replace(" (no aux)", "\n(no aux)").replace(" w=", "\nw=")
                            for k in order], fontsize=8)
        ax.set_ylabel("gap = pos - neg cosine")
        ax.set_title(f"{space} space", fontsize=11)
        for xi, g in zip(x, gaps):
            ax.text(xi, g, f"{g:+.3f}", ha="center",
                    va="bottom" if g >= 0 else "top", fontsize=8)
    fig.suptitle(
        "Does the auxiliary loss change what SigLIP encodes?\n"
        "gap > 0: same-frame renders (different robot) are MORE alike than different frames -> encodes the scene\n"
        "gap < 0: the encoder separates embodiments instead\n"
        "grey=init, orange=baseline, purple=frozen, red=L2-SP, blue=alignment, green=contrastive",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    p1 = args.out / "gap_by_checkpoint.png"
    fig.savefig(p1, dpi=120); plt.close(fig)
    log(f"wrote {p1}")

    # ---- figure: per-encoder similarity matrices (centered) ---------------------------------------
    n = len(order)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.9 * rows), squeeze=False)
    nf, nv = batches[0][1], batches[0][2]
    for i, label in enumerate(order):
        ax = axes[i // cols][i % cols]
        f = encode(encoders[label], batches[0][0])
        sim, p, nneg, g = gap_stats(f, nf, nv, center=True)
        im = ax.imshow(sim, cmap="viridis")
        for b in range(1, nf):
            ax.axhline(b * nv - 0.5, color="w", lw=0.7)
            ax.axvline(b * nv - 0.5, color="w", lw=0.7)
        ax.set_title(f"{label}\ncentered gap={g:+.4f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Centered cosine similarity (diagonal 3x3 blocks = same frame = positives)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p2 = args.out / "similarity_by_checkpoint.png"
    fig.savefig(p2, dpi=120); plt.close(fig)
    log(f"wrote {p2}")

    log("")
    log(f"{'encoder':26s} {'raw pos':>8s} {'raw neg':>8s} {'raw gap':>9s} {'cen pos':>8s} {'cen neg':>8s} {'cen gap':>9s}")
    for label in order:
        r, c = results[label]["raw"], results[label]["centered"]
        log(f"{label:26s} {r[0]:8.4f} {r[1]:8.4f} {r[2]:+9.4f} {c[0]:8.4f} {c[1]:8.4f} {c[2]:+9.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
