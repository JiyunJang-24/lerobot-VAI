#!/usr/bin/env python
"""Train the VLM on the redesigned export, and score it on held-out robots.

    [image_t, image_t+h] + "How did the gripper move from the first image to the second?"
        -> "move forward moderately, move right slightly, close gripper"

Differences from the v1 trainer, all forced by the new data rather than chosen:
  * one pair per ROW, so nothing is assembled and no nuisance axis has to be held constant
  * the label comes from `delta_eef_cam`, the CAMERA-frame displacement. With 27 cameras a
    world-frame label could not be read off the images without first identifying the camera.
  * `--high` uses the 512x910 renders. At 180x320 a 5 degree rotation moved the gripper's
    extremities 0.87 px -- under a ninth of a patch -- which is why rotation was never learnable.
    Running both resolutions on identical data is the direct test of that explanation.

    python src/lerobot/scripts/train_vlm_motion_v2.py --tag v2_cam --steps 4000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.scripts.motion_data_v2 import (  # noqa: E402
    DEFAULT, build_table, cached_table, deltas, embodiment_names, exclude, load_cache,
    split_embodiments, usable,
)
from lerobot.scripts.motion_language import (  # noqa: E402
    MotionDescriber, fit_thresholds, quat_to_euler,
)

MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
QUESTION = "How did the gripper move from the first image to the second?"


def log(msg: str) -> None:
    print(f"[vlm_v2] {msg}", flush=True)


def describe_rows(describer, d_pos, d_quat, grip):
    """MotionDescriber wants two states; here the delta is already given, so feed it a zero
    anchor and a state equal to the delta. Identical arithmetic, no re-derivation."""
    n = len(d_pos)
    zero = np.zeros((n, 8))
    zero[:, 6] = 1.0  # identity quaternion xyzw
    nxt = np.concatenate([d_pos, d_quat, np.zeros((n, 1))], axis=1)
    nxt[:, 7] = np.maximum(grip, 0)
    zero[:, 7] = np.maximum(-grip, 0)
    return describer.describe(zero, nxt)


def encode_batch(processor, pairs_uint8, answers, device):
    from PIL import Image

    texts = [processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                      {"type": "text", "text": QUESTION}]},
         {"role": "assistant", "content": [{"type": "text", "text": a}]}],
        tokenize=False) for a in answers]
    images = [[Image.fromarray(p[0].permute(1, 2, 0).numpy()),
               Image.fromarray(p[1].permute(1, 2, 0).numpy())] for p in pairs_uint8]
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    labels[labels == processor.tokenizer.convert_tokens_to_ids("<image>")] = -100
    marker = processor.tokenizer("Assistant:", add_special_tokens=False)["input_ids"]
    for r in range(labels.shape[0]):
        ids = batch["input_ids"][r].tolist()
        start = 0
        for i in range(len(ids) - len(marker)):
            if ids[i:i + len(marker)] == marker:
                start = i + len(marker)
        labels[r, :start] = -100
    out = {k: v.to(device) for k, v in batch.items()}
    out["labels"] = labels.to(device)
    return out


@torch.no_grad()
def evaluate(model, processor, cache, table, describer, embs, rng, device, n, frame, batch=8):
    sys.path.insert(0, str(REPO_ROOT))
    from tools.eval_vlm_motion import score

    from PIL import Image

    model.eval()
    sub = table[table.embodiment.isin(embs)]
    pick = sub.iloc[rng.choice(len(sub), size=min(n, len(sub)), replace=False)]
    d_pos, d_quat, grip = deltas(pick, frame)
    truth = describe_rows(describer, d_pos, d_quat, grip)
    positions = pick["cache_pos"].to_numpy()
    preds = []
    for start in range(0, len(positions), batch):
        chunk = positions[start:start + batch]
        texts = [processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                          {"type": "text", "text": QUESTION}]}],
            tokenize=False, add_generation_prompt=True) for _ in chunk]
        images = [[Image.fromarray(cache[int(i)][0].permute(1, 2, 0).numpy()),
                   Image.fromarray(cache[int(i)][1].permute(1, 2, 0).numpy())] for i in chunk]
        enc = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
        gen = model.generate(**enc, max_new_tokens=48, do_sample=False)
        preds += [s.strip() for s in
                  processor.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    model.train()
    result = score(preds, truth)
    result["distinct_pred"] = len(set(preds))
    result["distinct_truth"] = len(set(truth))
    return result, preds, truth


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--subset", default=DEFAULT)
    ap.add_argument("--high", action="store_true")
    ap.add_argument("--frame", default="cam", choices=["cam", "world"])
    ap.add_argument("--n-heldout", type=int, default=14)
    ap.add_argument("--exclude-grippers", default="xarm7_gripper",
                    help="comma-separated grippers to drop entirely. xarm7_gripper by default: "
                         "its renders do not match its labels (see motion_data_v2.exclude)")
    ap.add_argument("--exclude-arms", default="")
    ap.add_argument("--holdout-axis", default="random", choices=["random", "arm", "gripper"],
                    help="'arm' holds out every embodiment using a chosen arm, 'gripper' likewise. "
                         "Possible for the first time now that embodiments.json ships names.")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--no-rotation", action="store_true")
    ap.add_argument("--eval-n", type=int, default=96)
    ap.add_argument("--eval-freq", type=int, default=1000)
    ap.add_argument("--log-freq", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "outputs/vlm_motion_v2"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    table = usable(build_table(args.subset))
    table = exclude(table,
                    arms=[a for a in args.exclude_arms.split(",") if a],
                    grippers=[g for g in args.exclude_grippers.split(",") if g],
                    subset=args.subset)
    table = cached_table(table, args.subset, args.high)
    train_emb, heldout = split_embodiments(table, args.n_heldout, seed=0,
                                          subset=args.subset, axis=args.holdout_axis)
    names = embodiment_names(args.subset)
    if names:
        log(f"embodiment names available; held out e.g. "
            f"{[names.get(str(e), e) for e in heldout[:3]]}")
    train_table = table[table.embodiment.isin(train_emb)].reset_index(drop=True)

    # thresholds from TRAINING rows only
    d_pos, d_quat, grip = deltas(train_table, args.frame)
    describer = MotionDescriber(fit_thresholds(d_pos), fit_thresholds(quat_to_euler(d_quat)),
                                grip_threshold=0.5, include_rotation=not args.no_rotation,
                                frame="fixed")
    log(f"labels from delta_eef{'_cam' if args.frame == 'cam' else ''}: "
        f"|d| mean {np.linalg.norm(d_pos, axis=1).mean():.3f} m, "
        f"rotation mean {np.abs(np.degrees(quat_to_euler(d_quat))).mean():.1f} deg")

    cache = load_cache(args.subset, args.high)
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL)
    processor.image_processor.do_image_splitting = False
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    model.train()
    model.config.use_cache = False
    log(f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params, "
        f"{'HIGH-res 512x910' if args.high else 'low-res 180x320'}, "
        f"rotation words {'OFF' if args.no_rotation else 'ON'}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        pick = train_table.iloc[rng.choice(len(train_table), size=args.batch_size, replace=False)]
        dp, dq, dg = deltas(pick, args.frame)
        answers = describe_rows(describer, dp, dq, dg)
        batch = encode_batch(processor, cache[pick["cache_pos"].to_numpy()], answers, device)
        loss = model(**batch).loss
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        if step % args.log_freq == 0 or step == args.steps:
            rate = step / (time.time() - t0)
            history.append({"step": step, "loss": float(loss)})
            log(f"step {step:5d}/{args.steps}  CE {float(loss):.4f}  |g| {float(grad):6.2f}  "
                f"{rate:.2f} it/s  eta {(args.steps - step) / rate / 60:.0f} min")
        if step % args.eval_freq == 0 or step == args.steps:
            seen, _, _ = evaluate(model, processor, cache, train_table, describer,
                                  train_emb[:14], np.random.default_rng(7), device,
                                  args.eval_n, args.frame)
            held, preds, truth = evaluate(model, processor, cache, table, describer, heldout,
                                          np.random.default_rng(7), device, args.eval_n, args.frame)
            log(f"  eval @{step}  seen all {seen['all']:.3f} tr {seen['translation']:.3f}  |  "
                f"HELD all {held['all']:.3f} tr {held['translation']:.3f} "
                f"rot {held['rotation']:.3f}  distinct {held['distinct_pred']}/"
                f"{held['distinct_truth']}")
            history[-1].update({"seen": seen, "heldout": held})
            (out_dir / "results.json").write_text(json.dumps(
                {"args": vars(args), "train_embodiments": train_emb, "heldout": heldout,
                 "history": history, "examples": list(zip(preds[:5], truth[:5], strict=True))},
                indent=2, default=str))
    model.save_pretrained(out_dir / "model")
    processor.save_pretrained(out_dir / "model")
    log(f"done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
