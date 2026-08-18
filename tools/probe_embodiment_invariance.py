#!/usr/bin/env python
"""Does the encoder ignore the robot but still see the robot MOVE?

The pos/neg gap alone cannot answer that. A gap can be positive because the encoder became
embodiment-invariant, or because it collapsed and made everything similar; and it says nothing
about whether two moments of the same episode are told apart. This splits the pairs three ways:

    A  same frame, different robot   -- should be HIGH  (the robot is a nuisance variable)
    B  different frame, same robot   -- should be LOW   (the scene changed, so should the feature)
    C  different frame, different robot

A useful encoder has A high and B low. A collapsed one has A ~ B ~ 1. An encoder that reads the
robot has B > A.

It also plots similarity against time separation, which is the direct form of "the arm moved, so
the feature moved".

    python tools/probe_embodiment_invariance.py \
        --tower outputs/siglip_pretrain/contrastive_mean_b16/vision_tower.safetensors
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad  # noqa: E402
from lerobot.scripts.lerobot_train_with_visual_robust import (  # noqa: E402
    _select_visual_robust_image_keys,
)

VLM_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def log(msg: str) -> None:
    print(f"[probe] {msg}", flush=True)


def load_tower(path: str | None, device):
    from transformers import AutoModelForImageTextToText

    tower = AutoModelForImageTextToText.from_pretrained(VLM_MODEL, dtype=torch.float32).model.vision_model
    if path:
        from safetensors.torch import load_file

        tower.load_state_dict(load_file(path), strict=True)
    return tower.to(device).eval()


@torch.no_grad()
def features_for_frames(tower, dataset, indices, device, chunk=8):
    """[n_frames, n_views, D] mean-pooled features, plus the view names."""
    items = [dataset[i] for i in indices]
    batch = {k: torch.stack([x[k] for x in items]) for k in items[0] if torch.is_tensor(items[0][k])}
    keys = _select_visual_robust_image_keys(batch, image_prefix="observation.image.")
    images = []
    for i in range(len(items)):
        for k in keys:
            img = batch[k][i]
            img = img[-1] if img.ndim == 4 else img
            images.append(resize_with_pad(img.unsqueeze(0), 512, 512, pad_value=0)[0] * 2.0 - 1.0)
    flat = torch.stack(images).to(device)
    out = []
    for piece in flat.split(chunk):
        out.append(tower(pixel_values=piece, patch_attention_mask=None).last_hidden_state.mean(dim=1))
    return torch.cat(out).view(len(items), len(keys), -1), keys


def analyse(feats, center=True):
    """feats: [F, V, D]. Returns the three pair means and the similarity-vs-time curve."""
    n_frames, n_views, _ = feats.shape
    flat = feats.reshape(n_frames * n_views, -1).float()
    if center:
        flat = flat - flat.mean(dim=0, keepdim=True)
    flat = F.normalize(flat, dim=-1)
    sim = flat @ flat.T

    frame_of = torch.arange(n_frames, device=flat.device).repeat_interleave(n_views)
    view_of = torch.arange(n_views, device=flat.device).repeat(n_frames)
    same_frame = frame_of[:, None] == frame_of[None, :]
    same_view = view_of[:, None] == view_of[None, :]
    eye = torch.eye(len(flat), dtype=torch.bool, device=flat.device)

    a = sim[same_frame & ~same_view]                 # same moment, different robot
    b = sim[~same_frame & same_view]                 # different moment, same robot
    c = sim[~same_frame & ~same_view]                # different moment, different robot

    # similarity as a function of |frame_i - frame_j|, same robot only
    dt = (frame_of[:, None] - frame_of[None, :]).abs()
    curve = {}
    for gap in sorted(set(dt[~eye].tolist())):
        mask = (dt == gap) & same_view & ~eye
        if mask.any():
            curve[gap] = sim[mask].mean().item()
    return a.mean().item(), b.mean().item(), c.mean().item(), curve


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "dataset_git/visual_robust_new_barx/new_barx")
    ap.add_argument("--tower", action="append", default=[],
                    help="path to a pre-trained vision tower; repeatable. Pretrained is always included.")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--stride", type=int, default=25, help="frames apart, so time separation is legible")
    ap.add_argument("--start", type=int, default=40)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    repo_ids = sorted(f"{p.parent.parent.parent.name}/lerobot" for p in args.root.glob("*/lerobot/meta/info.json"))
    dataset = MultiLeRobotDataset(
        repo_ids, root=args.root, delta_timestamps={r: None for r in repo_ids},
        visual_cue_mode="vanilla", use_wrist_cam=False, use_state=True, cache_in_memory=False,
    )
    indices = [args.start + i * args.stride for i in range(args.frames)]
    log(f"one episode, frames {indices[0]}..{indices[-1]} every {args.stride}")

    towers = [("pretrained", None)] + [(Path(t).parent.name, t) for t in args.tower]
    rows = []
    for name, path in towers:
        tower = load_tower(path, device)
        feats, keys = features_for_frames(tower, dataset, indices, device)
        a, b, c, curve = analyse(feats)
        rows.append((name, a, b, c, curve))
        del tower
        torch.cuda.empty_cache()

    log(f"views: {[k.split('.')[-1] for k in keys]}\n")
    print(f"{'encoder':28s} {'A 같은순간/다른로봇':>20s} {'B 다른순간/같은로봇':>20s} {'C 다른순간/다른로봇':>20s} {'A-B':>9s}")
    for name, a, b, c, _ in rows:
        print(f"{name:28s} {a:20.4f} {b:20.4f} {c:20.4f} {a - b:+9.4f}")

    print(f"\n{'encoder':28s} " + " ".join(f"{f'dt={g * args.stride}':>9s}" for g in sorted(rows[0][4])[:6]))
    for name, _, _, _, curve in rows:
        vals = [curve[g] for g in sorted(curve)[:6]]
        print(f"{name:28s} " + " ".join(f"{v:9.4f}" for v in vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
