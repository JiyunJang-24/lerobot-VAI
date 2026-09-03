#!/usr/bin/env python
"""The Jaco experiment: gripper localisation on a novel ARM, with real ground truth at last.

visual_robust_new_barx_ur5e renders the SAME 108 episodes from six NAMED embodiments and now
carries an eef_pixel column, which makes two things possible that were not before.

1. Section 9.5 measured "does the synthetic pixel head transfer to policy-corpus images" only
   indirectly, by fitting a camera to its own predictions, because no ground-truth pixel existed
   there. It does now, so that claim can be checked directly instead of inferred.

2. The three held-out categories the experiment brief asked for finally exist, because these
   embodiments are named rather than integer ids:

       PandaOmron / IIWAOmron / UR5eOmron    seen arm, seen gripper
       PandaOmronPandaGripper                seen arm, seen gripper (recombined)
       JacoOmronPandaGripper                 NOVEL ARM, seen gripper
       JacoOmron                             NOVEL ARM, novel gripper

   Comparing the two Jaco rows separates "the arm is unfamiliar" from "the gripper is unfamiliar",
   which the random integer split on eef_pairs could never do.

eef_pixel is one column per FRAME, not per embodiment -- correct, since every render places the
same EEF pose under the same camera. So the target is identical across the six and only the
appearance of the robot changes. That is exactly the right control.

    python tools/eval_jaco_eef_pixel.py --mode zeroshot
    python tools/eval_jaco_eef_pixel.py --mode train
"""

import argparse
import glob
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad  # noqa: E402
from lerobot.scripts.pretrain_siglip_eefpairs import RegressionHead  # noqa: E402

VR = Path("/dataset/jiyun/dataset_git/visual_robust_new_barx_ur5e/new_barx/"
          "UR5eOmron_PnPSinkToCounter/lerobot")
REPO_ID = "visual_robust/UR5eOmron_PnPSinkToCounter"
CATEGORY = {
    "PandaOmron": "seen arm, own gripper",
    "IIWAOmron": "seen arm, own gripper",
    "UR5eOmron": "seen arm, own gripper",
    "PandaOmronPandaGripper": "seen arm, seen gripper",
    "JacoOmronPandaGripper": "NOVEL arm, seen gripper",
    "JacoOmron": "NOVEL arm, novel gripper",
}
HELD_OUT = ["JacoOmron", "JacoOmronPandaGripper"]
W, H = 320, 180


def log(msg: str) -> None:
    print(f"[jaco] {msg}", flush=True)


def frame_table(max_episodes: int, stride: int) -> pd.DataFrame:
    files = sorted(glob.glob(str(VR / "data" / "**" / "*.parquet"), recursive=True))
    table = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    table["row"] = np.arange(len(table))
    table = table[table["eef_in_frame"].to_numpy().astype(bool)]
    keep = table.episode_index < max_episodes
    table = table[keep]
    table = table.iloc[::stride]
    log(f"{len(table)} in-frame samples from {table.episode_index.nunique()} episodes")
    return table.reset_index(drop=True)


def decode(embodiment: str, rows: np.ndarray) -> torch.Tensor:
    key = f"observation.images.{embodiment}.robot0_agentview_right"
    dataset = LeRobotDataset(REPO_ID, root=VR)
    out = torch.empty(len(rows), 3, H, W, dtype=torch.uint8)
    for i, row in enumerate(rows):
        frame = dataset[int(row)][key]
        frame = frame[-1] if frame.ndim == 4 else frame
        out[i] = (frame * 255).round().clamp(0, 255).to(torch.uint8)
    return out


def load_tower(path: str, device):
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText

    vlm = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", dtype=torch.float32)
    tower = vlm.model.vision_model
    if path:
        tower.load_state_dict(load_file(path), strict=True)
    return tower.to(device)


def tokens_of(tower, images_uint8, device, amp, grad=False):
    x = images_uint8.to(device=device, dtype=torch.float32) / 255.0
    x = resize_with_pad(x, 512, 512, pad_value=0) * 2.0 - 1.0
    ctx = nullcontext() if grad else torch.no_grad()
    out = []
    with ctx, (torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()):
        for piece in x.split(8):
            out.append(tower(pixel_values=piece, patch_attention_mask=None).last_hidden_state)
    return torch.cat(out)


def predict_px(tower, head, images, device, amp, batch=16):
    preds = []
    for start in range(0, len(images), batch):
        tok = tokens_of(tower, images[start:start + batch], device, amp)
        with torch.no_grad():
            preds.append(head(tok).float().cpu().numpy())
    pred = np.concatenate(preds)
    return np.stack([(pred[:, 0] + 1) / 2 * W, (pred[:, 1] + 1) / 2 * H], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="zeroshot", choices=["zeroshot", "train"])
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO_ROOT / "outputs/siglip_pretrain/all4_n42_all")
    ap.add_argument("--max-episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--eval-n", type=int, default=160)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/jaco_eef_pixel.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    rng = np.random.default_rng(args.seed)
    table = frame_table(args.max_episodes, args.stride)
    truth_all = np.stack(table["eef_pixel"].to_numpy()).astype(np.float64)
    rows_all = table["row"].to_numpy()

    log("decoding frames for each of the six embodiments (same rows, same labels) ...")
    images = {}
    for emb in CATEGORY:
        t0 = time.time()
        images[emb] = decode(emb, rows_all)
        log(f"  {emb:<24} {len(images[emb])} frames in {time.time() - t0:.0f}s")

    tower = load_tower(str(args.checkpoint / "vision_tower.safetensors"), device)
    head = RegressionHead(int(tower.config.hidden_size), 2, pool="attn").to(device)
    head.load_state_dict(torch.load(args.checkpoint / "head_eef_pixel.pt"))

    results = {"mode": args.mode, "checkpoint": args.checkpoint.name,
               "n_eval": int(min(args.eval_n, len(table))), "per_embodiment": {}}

    if args.mode == "train":
        train_emb = [e for e in CATEGORY if e not in HELD_OUT]
        log(f"training the pixel head on REAL frames of {train_emb}, holding out {HELD_OUT}")
        assert not (set(train_emb) & set(HELD_OUT)), "held-out embodiment leaked into training"
        idx_pool = np.arange(len(table) - args.eval_n)
        params = [{"params": list(head.parameters()), "lr": args.head_lr},
                  {"params": list(tower.parameters()), "lr": args.lr}]
        opt = torch.optim.AdamW(params, weight_decay=1e-4)
        tower.train(), head.train()
        t0 = time.time()
        for step in range(1, args.steps + 1):
            pick = rng.choice(idx_pool, size=args.batch_size, replace=False)
            emb = train_emb[int(rng.integers(len(train_emb)))]
            batch = images[emb][pick]
            target = truth_all[pick]
            target_n = torch.tensor(
                np.stack([target[:, 0] / W, target[:, 1] / H], 1) * 2 - 1,
                dtype=torch.float32, device=device)
            tok = tokens_of(tower, batch, device, amp, grad=True)
            pred = head(tok).float()
            loss = F.mse_loss(pred, target_n)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 5.0)
            opt.step()
            if step % 250 == 0 or step == args.steps:
                px = float((pred.detach() - target_n).abs().mean() * ((W + H) / 4))
                rate = step / (time.time() - t0)
                log(f"  step {step:5d}/{args.steps}  loss {float(loss):.5f}  ~{px:.1f} px  "
                    f"{rate:.2f} it/s  eta {(args.steps - step) / rate / 60:.0f} min")
        tower.eval(), head.eval()

    eval_idx = np.arange(len(table) - args.eval_n, len(table))
    truth = truth_all[eval_idx]
    log(f"{'embodiment':<24}{'category':<26}{'pixel error':>12}")
    for emb, category in CATEGORY.items():
        pred = predict_px(tower, head, images[emb][eval_idx], device, amp)
        err = np.linalg.norm(pred - truth, axis=1)
        results["per_embodiment"][emb] = {
            "category": category, "split": "heldout" if emb in HELD_OUT else "seen",
            "pixel_error_mean": float(err.mean()), "pixel_error_median": float(np.median(err)),
        }
        log(f"{emb:<24}{category:<26}{err.mean():>9.1f} px")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
