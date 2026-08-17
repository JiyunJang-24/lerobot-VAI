#!/usr/bin/env python
"""Annotated companion to compare_siglip_features.py: shows WHICH images form a positive pair and
which form a negative pair, next to the similarity matrix they produce.

The similarity figures alone are easy to misread, because "positive"/"negative" are defined by the
sampling, not by anything visible in the matrix. Concretely, for one batch:

  * every row of the batch is one FRAME (one timestep of one episode), rendered three times -- once
    per embodiment (IIWA / Panda / UR5e). Same scene, same instant, different robot.
  * POSITIVE pair = two renders of the SAME frame, different robot.  -> should be similar
  * NEGATIVE pair = two renders from DIFFERENT frames.                -> should be dissimilar

so a representation that encodes the scene rather than the robot has positives MORE similar than
negatives.

Usage:
    python tools/explain_siglip_pairs.py --checkpoint <ckpt> --out outputs/siglip_feature_analysis
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
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
    print(f"[explain_siglip_pairs] {msg}", flush=True)


def to_img(t):
    a = t.detach().float().cpu().numpy()
    if a.ndim == 4:
        a = a[-1]
    return np.clip(a.transpose(1, 2, 0), 0, 1)


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "dataset_git/visual_robust_new_barx/new_barx")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/siglip_feature_analysis")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    repo_ids = sorted(f"{p.parent.parent.parent.name}/lerobot" for p in args.root.glob("*/lerobot/meta/info.json"))
    ds = MultiLeRobotDataset(
        repo_ids, root=args.root, delta_timestamps={r: None for r in repo_ids},
        visual_cue_mode="vanilla", use_wrist_cam=False, use_state=True, cache_in_memory=False,
    )
    idx = next(iter(SameEpisodeBatchSampler(ds.meta_episodes, batch_size=args.batch_size, shuffle=True)))
    items = [ds[i] for i in idx]
    batch = {k: torch.stack([it[k] for it in items]) for k in items[0] if torch.is_tensor(items[0][k])}
    keys = _select_visual_robust_image_keys(batch, image_prefix="observation.image.")
    B, V = len(items), len(keys)
    names = [k.rsplit(".", 1)[-1] for k in keys]
    log(f"{B} frames x {V} views {names}")

    imgs, labels = [], []
    for i in range(B):
        for k, nm in zip(keys, names):
            img = batch[k][i]
            img = img[-1] if img.ndim == 4 else img
            imgs.append(resize_with_pad(img.unsqueeze(0), 512, 512, pad_value=0)[0] * 2.0 - 1.0)
            labels.append(f"f{int(items[i]['frame_index'])}\n{nm.replace('Omron','')}")
    images = torch.stack(imgs)

    from transformers import AutoModelForImageTextToText
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    log("loading encoders ...")
    pre = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", dtype=torch.float32).model.vision_model.eval()
    tuned = SmolVLAPolicy.from_pretrained(str(args.checkpoint)).model.vlm_with_expert.get_vlm_model().vision_model.float().eval()

    feats = {"pretrained (init)": encode(pre, images), "fine-tuned (baseline)": encode(tuned, images)}

    # one concrete positive pair (frame 0: view 0 vs view 1) and negative pair (frame 0 v0 vs frame 1 v0)
    pos_a, pos_b = 0, 1
    neg_a, neg_b = 0, V

    fig = plt.figure(figsize=(15, 4 + 2.6 * B))
    gs = fig.add_gridspec(B + 2, V + 2, height_ratios=[1.15] * B + [0.28, 2.4], hspace=0.35, wspace=0.25)

    # left: the batch laid out exactly as it feeds the matrix
    for i in range(B):
        for j in range(V):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(to_img(images[i * V + j] * 0.5 + 0.5))
            ax.set_xticks([]); ax.set_yticks([])
            colour = "tab:blue" if i == 0 else "0.6"
            for sp in ax.spines.values():
                sp.set_edgecolor(colour); sp.set_linewidth(2.5)
            if i == 0:
                ax.set_title(names[j].replace("Omron", ""), fontsize=10)
            if j == 0:
                ax.set_ylabel(f"frame {int(items[i]['frame_index'])}", fontsize=9)

    # annotate one positive and one negative
    axp = fig.add_subplot(gs[0, V])
    axp.axis("off")
    axp.text(0, 0.5,
             "POSITIVE\n같은 frame\n다른 robot\n→ 비슷해야 함",
             fontsize=10, color="tab:blue", va="center", weight="bold")
    axn = fig.add_subplot(gs[1, V])
    axn.axis("off")
    axn.text(0, 0.5,
             "NEGATIVE\n다른 frame\n→ 달라야 함",
             fontsize=10, color="tab:red", va="center", weight="bold")

    # bottom: matrices
    for col, (name, f) in enumerate(feats.items()):
        for k, center in enumerate([False, True]):
            g = f - f.mean(0, keepdim=True) if center else f
            g = F.normalize(g, dim=-1)
            sim = (g @ g.T).numpy()
            ax = fig.add_subplot(gs[B + 1, col * 2 + k])
            im = ax.imshow(sim, cmap="viridis")
            for b in range(1, B):
                ax.axhline(b * V - 0.5, color="w", lw=1.0)
                ax.axvline(b * V - 0.5, color="w", lw=1.0)
            ax.add_patch(mpatches.Rectangle((pos_b - 0.5, pos_a - 0.5), 1, 1, fill=False, edgecolor="tab:blue", lw=2.5))
            ax.add_patch(mpatches.Rectangle((neg_b - 0.5, neg_a - 0.5), 1, 1, fill=False, edgecolor="tab:red", lw=2.5))
            frame_of = np.repeat(np.arange(B), V)
            same = frame_of[:, None] == frame_of[None, :]
            off = ~np.eye(len(sim), dtype=bool)
            p, n = sim[same & off].mean(), sim[~same].mean()
            ax.set_title(f"{name}\n{'centered' if center else 'raw'}\npos={p:.3f} neg={n:.3f} gap={p-n:+.3f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        "왼쪽: 한 배치의 실제 이미지 (가로=embodiment, 세로=frame)   |   아래: 그 24장으로 만든 유사도 행렬\n"
        "파란 칸 = POSITIVE(같은 frame, 다른 robot)   빨간 칸 = NEGATIVE(다른 frame)\n"
        "gap = pos - neg 가 양수여야 '로봇이 아니라 장면을 인코딩'하는 것",
        fontsize=12,
    )
    out = args.out / "pairs_explained.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
