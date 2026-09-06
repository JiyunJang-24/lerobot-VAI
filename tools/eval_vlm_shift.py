#!/usr/bin/env python
"""Shift both frames of a pair together, and see whether the model still reads the motion.

The stratified evaluation (eval_vlm_by_region.py) can only probe positions the data actually
visits, and the gripper never leaves u 242-637 of 910 -- so it cannot say anything about the outer
57% of the frame, which is exactly where the policy corpus often puts the arm.

Translating BOTH frames by the same offset moves the gripper into that unvisited region while
leaving the label untouched: the EEF displacement between the two frames is what is being
predicted, and a common translation does not change it. So any drop is the model failing on
position alone, with the task held fixed.

This is the same probe section 9.5 applied to the vision tower's features, where a 40 px shift
moved them 31% of the way toward an unrelated image while swapping the ROBOT moved them 4%. Here it
is applied end to end, to the answer rather than the representation.

    python tools/eval_vlm_shift.py --model outputs/vlm_motion_v2/high/model --tag high --high
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from lerobot.scripts.motion_data_v2 import (  # noqa: E402
    DEFAULT, build_table, cached_table, deltas, exclude, load_cache, split_embodiments, usable,
)
from lerobot.scripts.motion_language import MotionDescriber, fit_thresholds, quat_to_euler  # noqa: E402
from lerobot.scripts.train_vlm_motion_v2 import QUESTION, describe_rows  # noqa: E402
from tools.eval_vlm_motion import score  # noqa: E402


def log(msg: str) -> None:
    print(f"[shift] {msg}", flush=True)


def shift_pair(pair: torch.Tensor, dx: int) -> torch.Tensor:
    """Translate both frames horizontally by dx, replicating the edge column into the gap.

    Edge replication rather than black padding: a black bar is itself a strong out-of-distribution
    cue, and would confound "the model cannot handle this position" with "the model has never seen
    a black bar".
    """
    if dx == 0:
        return pair
    out = torch.roll(pair, shifts=dx, dims=-1)
    if dx > 0:
        out[..., :dx] = out[..., dx:dx + 1]
    else:
        out[..., dx:] = out[..., dx - 1:dx]
    return out


@torch.no_grad()
def generate(model, processor, images, device, batch=8):
    from PIL import Image

    out = []
    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        texts = [processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                          {"type": "text", "text": QUESTION}]}],
            tokenize=False, add_generation_prompt=True) for _ in chunk]
        pil = [[Image.fromarray(p[0].permute(1, 2, 0).numpy()),
                Image.fromarray(p[1].permute(1, 2, 0).numpy())] for p in chunk]
        enc = processor(text=texts, images=pil, return_tensors="pt", padding=True).to(device)
        gen = model.generate(**enc, max_new_tokens=48, do_sample=False)
        out += [s.strip() for s in processor.batch_decode(
            gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--high", action="store_true")
    ap.add_argument("--holdout-axis", default="random", choices=["random", "arm", "gripper"])
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    table = exclude(usable(build_table(DEFAULT)), grippers=["xarm7_gripper"], subset=DEFAULT)
    table = cached_table(table, DEFAULT, args.high)
    train_emb, heldout = split_embodiments(table, 12, seed=0, subset=DEFAULT,
                                           axis=args.holdout_axis)
    train_table = table[table.embodiment.isin(train_emb)].reset_index(drop=True)
    dp, dq, _ = deltas(train_table, "cam")
    describer = MotionDescriber(fit_thresholds(dp), fit_thresholds(quat_to_euler(dq)),
                                grip_threshold=0.5, frame="fixed")

    held = table[table.embodiment.isin(heldout)].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    pick = held.iloc[rng.choice(len(held), size=args.n, replace=False)]
    d_pos, d_quat, grip = deltas(pick, "cam")
    truth = describe_rows(describer, d_pos, d_quat, grip)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    processor.image_processor.do_image_splitting = False
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    model.eval()
    cache = load_cache(DEFAULT, args.high)
    pairs = cache[pick["cache_pos"].to_numpy()].clone()

    width = pairs.shape[-1]
    # shifts scaled to the frame, so the low-res (320) and high-res (910) runs move the gripper by
    # the same FRACTION of the image and stay comparable
    fractions = [-0.28, -0.17, -0.08, 0.0, 0.08, 0.17, 0.28]
    results = {"tag": args.tag, "width": int(width), "shifts": []}
    log(f"{'shift px':>10}{'frac':>8}{'all':>8}{'trans':>8}{'rot':>8}{'distinct':>10}")
    for frac in fractions:
        dx = int(round(frac * width))
        moved = torch.stack([shift_pair(p, dx) for p in pairs])
        preds = generate(model, processor, moved, device)
        s = score(preds, truth)
        results["shifts"].append({"dx": dx, "frac": frac, "distinct": len(set(preds)), **s})
        log(f"{dx:>10}{frac:>8.2f}{s['all']:>8.3f}{s['translation']:>8.3f}{s['rotation']:>8.3f}"
            f"{len(set(preds)):>6}/{len(set(truth)):<3}")
    out = REPO_ROOT / f"outputs/vlm_shift_{args.tag}.json"
    out.write_text(json.dumps(results, indent=2))
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
