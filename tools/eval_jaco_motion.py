#!/usr/bin/env python
"""Two images in, the Cartesian motion between them out -- on a NOVEL ARM.

Experiments 1 and 2 answered this on synthetic eef_pairs renders, whose embodiments are unnamed
integer ids and do not include Jaco. The Jaco pixel experiment answered a DIFFERENT question --
one image, where is the gripper. This is the missing cell: the two-image motion task, on real
trajectories, with Jaco held out.

    image_t   --\\
                  SigLIP tower (shared) -> attn pool -> concat -> MLP -> d_position, d_rotation
    image_t+h --/

The six embodiments render the SAME 108 episodes, and observation.state is shared across all of
them, so the motion label is IDENTICAL for every render and only the robot's appearance changes.
That makes seen-vs-Jaco a clean comparison.

FRAME CAVEAT, stated rather than buried: this tree's xyz is WORLD frame (CLAUDE.md section 1), and
the base is parked differently per episode, so a given visual motion maps to different world
deltas across episodes. The image does show the kitchen, so base orientation is inferable and the
task is not ill-posed -- but it is harder than a base-frame target would be, and the absolute
numbers are correspondingly worse. The confound is identical for all six embodiments, so it does
not touch the seen-vs-Jaco comparison this script exists to make.

    python tools/eval_jaco_motion.py --steps 2000
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
from lerobot.scripts.motion_data import geodesic_deg, quat_conj, quat_mul  # noqa: E402
from lerobot.scripts.train_motion_prediction import MotionHead  # noqa: E402
from tools.eval_jaco_eef_pixel import CATEGORY, HELD_OUT, REPO_ID, VR, load_tower  # noqa: E402

W, H = 320, 180


def log(msg: str) -> None:
    print(f"[jaco_motion] {msg}", flush=True)


def build_pairs(max_episodes: int, horizon: int, stride: int):
    files = sorted(glob.glob(str(VR / "data" / "**" / "*.parquet"), recursive=True))
    table = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    table["row"] = np.arange(len(table))
    state = np.stack(table["observation.state"].to_numpy()).astype(np.float64)

    anchors, partners = [], []
    for ep in sorted(table.episode_index.unique())[:max_episodes]:
        rows = table.row.to_numpy()[table.episode_index.to_numpy() == ep]
        usable = rows[: len(rows) - horizon][::stride]
        anchors.append(usable)
        partners.append(usable + horizon)
    anchors = np.concatenate(anchors)
    partners = np.concatenate(partners)

    d_pos = state[partners, :3] - state[anchors, :3]
    q_rel = quat_mul(state[partners, 3:7], quat_conj(state[anchors, 3:7]))
    grip = np.sign(state[partners, 7] - state[anchors, 7]).astype(int) + 1

    # the sanity test again, on this corpus: applying the delta must reproduce the target
    back = state[anchors, :3] + d_pos
    assert np.abs(back - state[partners, :3]).max() < 1e-9
    q_back = quat_mul(q_rel, state[anchors, 3:7])
    sign = np.sign((q_back * state[partners, 3:7]).sum(-1, keepdims=True))
    assert np.abs(q_back * sign - state[partners, 3:7]).max() < 1e-6
    log(f"{len(anchors)} pairs over {min(max_episodes, table.episode_index.nunique())} episodes, "
        f"h={horizon}: |d| mean {np.linalg.norm(d_pos, axis=1).mean():.3f} m, "
        f"rot mean {geodesic_deg(q_rel, q_rel * 0 + np.array([0, 0, 0, 1.0])).mean():.1f} deg")
    return anchors, partners, d_pos, q_rel, grip


def decode(embodiment: str, rows: np.ndarray) -> dict:
    key = f"observation.images.{embodiment}.robot0_agentview_right"
    dataset = LeRobotDataset(REPO_ID, root=VR)
    out = {}
    for row in rows:
        frame = dataset[int(row)][key]
        frame = frame[-1] if frame.ndim == 4 else frame
        out[int(row)] = (frame * 255).round().clamp(0, 255).to(torch.uint8)
    return out


def stack(cache: dict, rows: np.ndarray) -> torch.Tensor:
    return torch.stack([cache[int(r)] for r in rows])


def tokens_of(tower, images, device, amp, grad):
    x = images.to(device=device, dtype=torch.float32) / 255.0
    x = resize_with_pad(x, 512, 512, pad_value=0) * 2.0 - 1.0
    ctx = nullcontext() if grad else torch.no_grad()
    out = []
    with ctx, (torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()):
        for piece in x.split(8):
            out.append(tower(pixel_values=piece, patch_attention_mask=None).last_hidden_state)
    return torch.cat(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO_ROOT / "outputs/siglip_pretrain/all4_n42_all")
    ap.add_argument("--max-episodes", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--eval-n", type=int, default=140)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/jaco_motion.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    rng = np.random.default_rng(args.seed)

    anchors, partners, d_pos, q_rel, grip = build_pairs(args.max_episodes, args.horizon, args.stride)
    needed = np.unique(np.concatenate([anchors, partners]))
    log(f"decoding {len(needed)} distinct frames x 6 embodiments ...")
    images = {}
    for emb in CATEGORY:
        t0 = time.time()
        images[emb] = decode(emb, needed)
        log(f"  {emb:<24} {len(images[emb])} frames in {time.time() - t0:.0f}s")

    n_eval = args.eval_n
    eval_idx = np.arange(len(anchors) - n_eval, len(anchors))
    train_idx = np.arange(len(anchors) - n_eval)
    pos_scale = float(np.linalg.norm(d_pos[train_idx], axis=1).std())
    train_emb = [e for e in CATEGORY if e not in HELD_OUT]
    assert not (set(train_emb) & set(HELD_OUT)), "held-out embodiment leaked into training"
    log(f"train on {train_emb}")
    log(f"HELD OUT: {HELD_OUT}   pos scale {pos_scale:.4f} m")

    tower = load_tower(str(args.checkpoint / "vision_tower.safetensors"), device)
    head = MotionHead(int(tower.config.hidden_size)).to(device)
    groups = [{"params": list(head.parameters()), "lr": args.head_lr},
              {"params": list(tower.parameters()), "lr": args.lr}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)

    tower.train(), head.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        pick = rng.choice(train_idx, size=args.batch_size, replace=False)
        emb = train_emb[int(rng.integers(len(train_emb)))]
        tok_t = tokens_of(tower, stack(images[emb], anchors[pick]), device, amp, grad=True)
        tok_h = tokens_of(tower, stack(images[emb], partners[pick]), device, amp, grad=True)
        p_pos, p_quat, p_grip = head(tok_t, tok_h)
        tgt_pos = torch.tensor(d_pos[pick] / pos_scale, dtype=torch.float32, device=device)
        tgt_quat = F.normalize(torch.tensor(q_rel[pick], dtype=torch.float32, device=device), dim=-1)
        loss = (F.mse_loss(p_pos.float(), tgt_pos)
                + 1.0 - (F.normalize(p_quat.float(), dim=-1) * tgt_quat).sum(-1).abs().clamp(max=1).mean()
                + 0.1 * F.cross_entropy(p_grip.float(),
                                        torch.tensor(grip[pick], device=device)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 5.0)
        opt.step()
        if step % 250 == 0 or step == args.steps:
            rate = step / (time.time() - t0)
            log(f"  step {step:5d}/{args.steps}  loss {float(loss):.4f}  {rate:.2f} it/s  "
                f"eta {(args.steps - step) / rate / 60:.0f} min")
    tower.eval(), head.eval()

    # Baselines: what predicting the training mean, and no rotation, would score.
    mean_pos = d_pos[train_idx].mean(0)
    base_mae = float(np.abs(d_pos[eval_idx] - mean_pos).mean())
    base_rot = float(geodesic_deg(np.tile([0, 0, 0, 1.0], (n_eval, 1)), q_rel[eval_idx]).mean())
    log(f"baselines: predict-the-mean {base_mae * 100:.2f} cm, identity-rotation {base_rot:.1f} deg")

    results = {"baselines": {"translation_mae_m": base_mae, "rotation_err_deg": base_rot},
               "horizon": args.horizon, "n_eval": int(n_eval), "per_embodiment": {}}
    log(f"{'embodiment':<24}{'category':<26}{'trans MAE':>11}{'dir cos':>9}{'rot err':>10}")
    for emb, category in CATEGORY.items():
        preds = []
        for start in range(0, n_eval, 16):
            idx = eval_idx[start:start + 16]
            tok_t = tokens_of(tower, stack(images[emb], anchors[idx]), device, amp, grad=False)
            tok_h = tokens_of(tower, stack(images[emb], partners[idx]), device, amp, grad=False)
            with torch.no_grad():
                p_pos, p_quat, _ = head(tok_t, tok_h)
            preds.append((p_pos.float().cpu().numpy() * pos_scale, p_quat.float().cpu().numpy()))
        pp = np.concatenate([p[0] for p in preds])
        pq = np.concatenate([p[1] for p in preds])
        tp, tq = d_pos[eval_idx], q_rel[eval_idx]
        results["per_embodiment"][emb] = {
            "category": category, "split": "heldout" if emb in HELD_OUT else "seen",
            "translation_mae_m": float(np.abs(pp - tp).mean()),
            "translation_dir_cos": float(((pp * tp).sum(1) / (
                np.linalg.norm(pp, axis=1) * np.linalg.norm(tp, axis=1)).clip(1e-9)).mean()),
            "rotation_err_deg": float(geodesic_deg(pq, tq).mean()),
        }
        r = results["per_embodiment"][emb]
        log(f"{emb:<24}{category:<26}{r['translation_mae_m'] * 100:>8.2f}cm"
            f"{r['translation_dir_cos']:>9.3f}{r['rotation_err_deg']:>7.1f}deg")
    args.out.write_text(json.dumps(results, indent=2))
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
