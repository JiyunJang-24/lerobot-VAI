#!/usr/bin/env python
"""Experiment 1 & 2: can visual motion understanding generalise to unseen robot morphologies?

    (image_t, image_t+h) -> delta_position, delta_rotation, gripper change

trained on synthetic embodiments and scored on embodiments that never appear in training. The whole
design question is whether the model can be answering from the image pair rather than from having
recognised the robot, so two guards run before any training happens:

  * every embodiment draws from ONE shared pool of pose pairs, so embodiment identity carries zero
    information about the label (assert_no_motion_shortcut)
  * held-out embodiments never appear in a training batch (assert_no_leakage, re-checked per batch)

Experiment 2 is the same script with --num-embodiments varied, under either sampling protocol:
  --samples-per-embodiment N   total data grows with embodiment count (protocol A)
  --fixed-total-samples N      total data held constant (protocol B) -- isolates morphology
                               diversity from simply having more images

    python src/lerobot/scripts/train_motion_prediction.py --tag exp1_n42 --steps 3000
"""

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.scripts.motion_data import (  # noqa: E402
    SUBSETS, assert_no_leakage, assert_no_motion_shortcut, build_table, geodesic_deg, load_cache,
    load_images, motion_labels, pose_states, quat_angle_deg, sample_pose_pairs,
    verify_reconstruction,
)

N_BG, N_VIEW = 12, 4


def log(msg: str) -> None:
    print(f"[motion] {msg}", flush=True)


def build_index(table, subsets):
    """[embodiment, pose, subset, background, view] -> cache position, or -1 where not rendered."""
    n_emb = int(table.embodiment.max()) + 1
    n_pose = int(table.pose.max()) + 1
    n_color = int(table.color.max()) + 1
    # The colour axis is REQUIRED, not optional. Without it, three rows -- the three robot paint
    # jobs -- collapse into one cell and only the last survives, so two thirds of every render is
    # unreachable and a pair drawn across two cells can silently change the robot's colour.
    index = np.full((n_emb, n_pose, len(subsets), N_BG, N_VIEW, n_color), -1, dtype=np.int64)
    sub_id = {s: i for i, s in enumerate(subsets)}
    index[
        table.embodiment.to_numpy(), table.pose.to_numpy(),
        table.subset.map(sub_id).to_numpy(), table.background.to_numpy(), table.view.to_numpy(),
        table.color.to_numpy(),
    ] = table.cache_pos.to_numpy()
    filled = (index >= 0).mean()
    log(f"render index: {index.shape}, {100 * filled:.0f}% of combinations rendered")
    return index


class MotionHead(torch.nn.Module):
    """concat(pooled_t, pooled_t+h) -> delta. Deliberately minimal; the question is the data.

    Pooling is attention-based rather than a mean, for the reason measured in CLAUDE.md section 3:
    a mean over 1024 position-tagged patches is close to location-blind, and a displacement is
    nothing but a change of location.
    """

    def __init__(self, in_dim: int, hidden: int = 512, pool: str = "attn"):
        super().__init__()
        self.pool = pool
        self.score = torch.nn.Linear(in_dim, 1) if pool == "attn" else None
        per_image = in_dim * 2 if pool == "attn" else in_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(per_image * 2, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, 3 + 4 + 3),
        )

    def _pool(self, tokens):
        pooled = tokens.mean(dim=1)
        if self.pool == "attn":
            weights = torch.softmax(self.score(tokens), dim=1)
            pooled = torch.cat([pooled, (weights * tokens).sum(dim=1)], dim=-1)
        return pooled

    def forward(self, tokens_t, tokens_h):
        out = self.net(torch.cat([self._pool(tokens_t), self._pool(tokens_h)], dim=-1))
        return out[:, :3], out[:, 3:7], out[:, 7:10]


def load_tower(path: str, device):
    from transformers import AutoModelForImageTextToText

    vlm = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", dtype=torch.float32)
    tower = vlm.model.vision_model
    if path:
        from safetensors.torch import load_file

        tower.load_state_dict(load_file(path), strict=True)
        log(f"tower initialised from {path}")
    return tower.to(device)


def encode(tower, images, chunk=16):
    return torch.cat([
        tower(pixel_values=piece, patch_attention_mask=None).last_hidden_state
        for piece in images.split(chunk)
    ])


def make_sampler(index, pairs, embodiments, rng, allow_gripper_change: bool):
    n_sub = index.shape[2]
    """Draw (embodiment, pose_i, pose_j, subset_t, subset_h, bg, view) that are actually rendered."""
    def draw(n):
        out = np.empty((n, 7), dtype=np.int64)
        filled = 0
        while filled < n:
            k = (n - filled) * 3
            emb = rng.choice(embodiments, size=k)
            pair = pairs[rng.integers(0, len(pairs), size=k)]
            # Subsets are ordered [closed, open, closed_furniture, open_furniture], so bit 1 is
            # the furniture variant and bit 0 is the gripper. The furniture must be IDENTICAL in
            # both frames -- the kitchen does not get repainted mid-motion, and letting it change
            # inserts a scene change unrelated to the motion being labelled. The gripper may
            # differ, because opening or closing it IS part of the motion.
            furniture = rng.integers(0, max(1, n_sub // 2), size=k) * 2
            sub_t = furniture + rng.integers(0, min(2, n_sub), size=k)
            sub_h = furniture + (rng.integers(0, min(2, n_sub), size=k)
                                 if allow_gripper_change else sub_t - furniture)
            bg = rng.integers(0, N_BG, size=k)
            view = rng.integers(0, N_VIEW, size=k)
            pos_t = index[emb, pair[:, 0], sub_t, bg, view]
            pos_h = index[emb, pair[:, 1], sub_h, bg, view]
            ok = (pos_t >= 0) & (pos_h >= 0)
            take = min(int(ok.sum()), n - filled)
            sel = np.where(ok)[0][:take]
            out[filled:filled + take] = np.stack(
                [emb[sel], pair[sel, 0], pair[sel, 1], sub_t[sel], sub_h[sel], bg[sel], view[sel]], 1)
            filled += take
        return out
    return draw


def evaluate(tower, head, index, pairs, states, embodiments, cache, device, rng, n, batch,
             amp, pos_scale):
    """Per-embodiment translation / rotation / gripper error."""
    tower.eval(), head.eval()
    rows = []
    for emb in embodiments:
        draw = make_sampler(index, pairs, np.array([emb]), rng, allow_gripper_change=True)
        spec = draw(n)
        preds, targets = [], []
        for start in range(0, n, batch):
            chunk = spec[start:start + batch]
            pos_t = index[chunk[:, 0], chunk[:, 1], chunk[:, 3], chunk[:, 5], chunk[:, 6]]
            pos_h = index[chunk[:, 0], chunk[:, 2], chunk[:, 4], chunk[:, 5], chunk[:, 6]]
            with torch.no_grad(), (torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()):
                t_tok = encode(tower, load_images(cache, pos_t, device))
                h_tok = encode(tower, load_images(cache, pos_h, device))
                d_pos, d_quat, grip = head(t_tok, h_tok)
            preds.append((d_pos.float().cpu().numpy(), d_quat.float().cpu().numpy(),
                          grip.float().cpu().numpy()))
            targets.append(chunk)
        # The head regresses NORMALISED displacement; the labels are metres. Comparing the two
        # directly reports ~50 cm regardless of how well the model does, which looks like a failed
        # experiment rather than a unit mismatch.
        d_pos = np.concatenate([p[0] for p in preds]) * pos_scale
        d_quat = np.concatenate([p[1] for p in preds])
        grip = np.concatenate([p[2] for p in preds])
        spec = np.concatenate(targets)
        t_pos, t_quat = motion_labels(states, spec[:, 1:3])
        # gripper of a subset: SUBSETS index 0,2 are closed, 1,3 are open
        g_t = (spec[:, 3] % 2).astype(int)
        g_h = (spec[:, 4] % 2).astype(int)
        g_label = (g_h - g_t) + 1  # 0 open->closed, 1 unchanged, 2 closed->open
        rows.append({
            "embodiment": int(emb),
            "translation_mae_m": float(np.abs(d_pos - t_pos).mean()),
            "translation_dir_cos": float((
                (d_pos * t_pos).sum(1)
                / (np.linalg.norm(d_pos, axis=1) * np.linalg.norm(t_pos, axis=1)).clip(1e-9)).mean()),
            "rotation_err_deg": float(geodesic_deg(d_quat, t_quat).mean()),
            "gripper_acc": float((grip.argmax(1) == g_label).mean()),
            "n": int(len(spec)),
        })
    tower.train(), head.train()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--subsets", default=",".join(SUBSETS),
                    help="comma-separated eef_pairs subsets; must match a built image cache")
    ap.add_argument("--limit-poses", type=int, default=0,
                    help="use only N of the available poses. The pose ladder subsamples WITHIN one "
                         "subset so no cross-export render difference can confound it.")
    ap.add_argument("--tower", default="", help="empty = stock SigLIP")
    ap.add_argument("--heldout-embodiments", default="0,1,3,8,12,14,23,27,28,33,34,36,42,49")
    ap.add_argument("--num-embodiments", type=int, default=0, help="0 = all non-held-out")
    ap.add_argument("--samples-per-embodiment", type=int, default=0, help="protocol A")
    ap.add_argument("--fixed-total-samples", type=int, default=0, help="protocol B")
    ap.add_argument("--motion-horizon", type=float, default=0.15, help="max |delta position|, metres")
    ap.add_argument("--max-rot-deg", type=float, default=180.0)
    ap.add_argument("--pose-pairs", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--freeze-tower", action="store_true")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--eval-samples", type=int, default=96)
    ap.add_argument("--log-freq", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "outputs/motion_prediction"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    subsets = args.subsets.split(",")
    table = build_table(subsets)
    states = pose_states(table)
    index = build_index(table, subsets)
    n_pose_all = len(states)
    if args.limit_poses and args.limit_poses < n_pose_all:
        keep = np.sort(rng.choice(n_pose_all, size=args.limit_poses, replace=False))
        mask = np.zeros(n_pose_all, dtype=bool)
        mask[keep] = True
        log(f"pose ladder: using {args.limit_poses} of {n_pose_all} poses")
    else:
        mask = np.ones(n_pose_all, dtype=bool)
        log(f"using all {n_pose_all} poses")

    pairs = sample_pose_pairs(states, 10 ** 9, args.motion_horizon, rng)
    pairs = pairs[mask[pairs[:, 0]] & mask[pairs[:, 1]]]
    if len(pairs) > args.pose_pairs:
        pairs = pairs[rng.choice(len(pairs), size=args.pose_pairs, replace=False)]
    log(f"{len(pairs)} pose pairs after the pose limit and the {args.motion_horizon} m cap")
    if args.max_rot_deg < 180:
        _, q_rel = motion_labels(states, pairs)
        pairs = pairs[quat_angle_deg(q_rel) <= args.max_rot_deg]
        log(f"after rotation filter <= {args.max_rot_deg} deg: {len(pairs)} pairs")
    verify_reconstruction(states, pairs)

    heldout = [int(x) for x in args.heldout_embodiments.split(",") if x != ""]
    available = sorted(set(range(int(table.embodiment.max()) + 1)) - set(heldout))
    train_emb = available if args.num_embodiments <= 0 else list(
        rng.choice(available, size=min(args.num_embodiments, len(available)), replace=False))
    train_emb = sorted(int(e) for e in train_emb)
    assert_no_leakage(train_emb, heldout)
    assert_no_motion_shortcut({e: pairs for e in train_emb})

    if args.fixed_total_samples:
        budget = args.fixed_total_samples
        protocol = f"B: {budget} pairs total across {len(train_emb)} embodiments"
    elif args.samples_per_embodiment:
        budget = args.samples_per_embodiment * len(train_emb)
        protocol = f"A: {args.samples_per_embodiment}/embodiment x {len(train_emb)} = {budget} pairs"
    else:
        budget = args.steps * args.batch_size
        protocol = f"unbounded ({budget} pairs seen)"
    epochs = budget / (args.steps * args.batch_size)
    log(f"protocol {protocol}  ->  {epochs:.2f} passes over the sample budget")

    cache = load_cache(subsets)
    tower = load_tower(args.tower, device)
    head = MotionHead(int(tower.config.hidden_size)).to(device)
    if args.freeze_tower:
        for p in tower.parameters():
            p.requires_grad = False
        tower.eval()
    pos_scale = float(np.linalg.norm(motion_labels(states, pairs)[0], axis=1).std())
    log(f"trainable: tower {'no' if args.freeze_tower else 'yes'}, "
        f"head {sum(p.numel() for p in head.parameters()) / 1e6:.1f}M, pos scale {pos_scale:.4f} m")

    groups = [{"params": list(head.parameters()), "lr": args.head_lr}]
    if not args.freeze_tower:
        groups.append({"params": list(tower.parameters()), "lr": args.lr})
    optimiser = torch.optim.AdamW(groups, weight_decay=1e-4)

    # Protocol B means a FIXED pool of pairs, revisited; protocol A means a bigger pool.
    draw = make_sampler(index, pairs, np.array(train_emb), rng, allow_gripper_change=True)
    pool = draw(budget)
    assert not (set(pool[:, 0].tolist()) & set(heldout)), "held-out embodiment leaked into the pool"
    log(f"training pool: {len(pool)} pairs over {len(set(pool[:, 0].tolist()))} embodiments")

    history, t0 = [], time.time()
    amp = args.amp and device.type == "cuda"
    for step in range(1, args.steps + 1):
        spec = pool[rng.integers(0, len(pool), size=args.batch_size)]
        pos_t = index[spec[:, 0], spec[:, 1], spec[:, 3], spec[:, 5], spec[:, 6]]
        pos_h = index[spec[:, 0], spec[:, 2], spec[:, 4], spec[:, 5], spec[:, 6]]
        t_pos, t_quat = motion_labels(states, spec[:, 1:3])
        g_label = torch.tensor((spec[:, 4] % 2) - (spec[:, 3] % 2) + 1, device=device)
        target_pos = torch.tensor(t_pos / pos_scale, dtype=torch.float32, device=device)
        target_quat = torch.tensor(t_quat, dtype=torch.float32, device=device)

        with torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext():
            t_tok = encode(tower, load_images(cache, pos_t, device))
            h_tok = encode(tower, load_images(cache, pos_h, device))
            d_pos, d_quat, grip = head(t_tok, h_tok)
        d_pos, d_quat, grip = d_pos.float(), d_quat.float(), grip.float()

        pos_loss = F.mse_loss(d_pos, target_pos)
        q_pred = F.normalize(d_quat, dim=-1)
        q_true = F.normalize(target_quat, dim=-1)
        rot_loss = 1.0 - (q_pred * q_true).sum(-1).abs().clamp(max=1.0).mean()
        grip_loss = F.cross_entropy(grip, g_label)
        loss = pos_loss + rot_loss + 0.1 * grip_loss

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in groups for p in g["params"] if p.requires_grad], 5.0)
        optimiser.step()

        if step % args.log_freq == 0 or step == args.steps:
            mae = float((d_pos.detach() - target_pos).abs().mean() * pos_scale)
            rot = float(geodesic_deg(q_pred.detach().cpu().numpy(), t_quat).mean())
            rate = step / (time.time() - t0)
            history.append({"step": step, "pos_loss": float(pos_loss), "rot_loss": float(rot_loss),
                            "grip_loss": float(grip_loss), "train_mae_m": mae, "train_rot_deg": rot})
            log(f"step {step:5d}/{args.steps}  pos {float(pos_loss):.4f}  rot {float(rot_loss):.4f}  "
                f"grip {float(grip_loss):.3f}  |  MAE {mae * 100:.2f} cm  rot {rot:.1f} deg  "
                f"{rate:.2f} it/s  eta {(args.steps - step) / rate / 60:.0f} min")

    log("evaluating ...")
    eval_seen = evaluate(tower, head, index, pairs, states, train_emb[:14], cache, device,
                         np.random.default_rng(1234), args.eval_samples, args.batch_size, amp,
                         pos_scale)
    eval_held = evaluate(tower, head, index, pairs, states, heldout, cache, device,
                         np.random.default_rng(1234), args.eval_samples, args.batch_size, amp,
                         pos_scale)
    # A model that ignores the images entirely still scores something; without this the numbers
    # above have no floor to be read against.
    mean_pred = motion_labels(states, pairs)[0].mean(0)
    all_targets = motion_labels(states, pairs)[0]
    baseline = {
        "translation_mae_m": float(np.abs(all_targets - mean_pred).mean()),
        "rotation_err_deg": float(quat_angle_deg(motion_labels(states, pairs)[1]).mean()),
    }
    log(f"predict-the-mean baseline: {baseline['translation_mae_m'] * 100:.2f} cm, "
        f"identity-rotation baseline: {baseline['rotation_err_deg']:.1f} deg")

    def summarise(rows, name):
        return {
            "split": name, "n_embodiments": len(rows),
            "translation_mae_m": float(np.mean([r["translation_mae_m"] for r in rows])),
            "translation_dir_cos": float(np.mean([r["translation_dir_cos"] for r in rows])),
            "rotation_err_deg": float(np.mean([r["rotation_err_deg"] for r in rows])),
            "gripper_acc": float(np.mean([r["gripper_acc"] for r in rows])),
        }

    summary = {
        "tag": args.tag, "args": vars(args), "protocol": protocol,
        "n_poses_used": int(mask.sum()), "n_poses_available": int(n_pose_all),
        "n_train_embodiments": len(train_emb), "train_embodiments": train_emb,
        "heldout_embodiments": heldout, "n_pairs_pool": int(len(pairs)),
        "n_samples": int(len(pool)),
        "seen": summarise(eval_seen, "seen"), "heldout": summarise(eval_held, "heldout"),
        "baseline": baseline,
        "per_embodiment": {"seen": eval_seen, "heldout": eval_held},
        "history": history,
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    torch.save({"head": head.state_dict()}, out_dir / "motion_head.pt")
    # Save the tower too, so the real-data experiments can start from what this learned rather
    # than only from the head. That transfer IS the hypothesis, end to end.
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in tower.state_dict().items()},
              out_dir / "vision_tower.safetensors")

    log(f"{'split':<10}{'trans MAE':>12}{'dir cos':>10}{'rot deg':>10}{'grip acc':>10}")
    for row in (summary["seen"], summary["heldout"]):
        log(f"{row['split']:<10}{row['translation_mae_m'] * 100:>10.2f}cm"
            f"{row['translation_dir_cos']:>10.3f}{row['rotation_err_deg']:>10.1f}"
            f"{row['gripper_acc']:>10.3f}")
    log(f"wrote {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
