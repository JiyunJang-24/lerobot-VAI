#!/usr/bin/env python
"""Score a trained VLM by WHERE the gripper is in the frame, on held-out embodiments.

The aggregate held-out number is measured on the same distribution the model trained on, and that
distribution is concentrated: over all 448,000 pairs the gripper spans u 242-637 of 910, with
std 55, so the outer 57% of the frame is never visited (CLAUDE.md 10.12). A model can score well
on that aggregate while being good only near the mode.

So the question this answers is: in the region where the data is DENSEST -- where the model has
had the most chances to learn -- does a held-out embodiment do as well as a seen one? That is the
generous test of morphology generalisation. A gap there cannot be blamed on sparse coverage,
because the core is where coverage is best; and if there is no gap there, the aggregate number is
being dragged down by the thin tails rather than by morphology.

Both splits are scored in every bin, on the SAME position distribution, so the seen-vs-held-out
comparison is not contaminated by the two splits sitting at different places in the frame.

Bins are equal-COUNT, not equal-width: equal-width bins at the tails would hold a handful of
samples and the comparison would be noise.

    python tools/eval_vlm_by_region.py --model outputs/vlm_motion_v2/high/model --tag high
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
    print(f"[by_region] {msg}", flush=True)


@torch.no_grad()
def generate(model, processor, cache, positions, device, batch=8):
    from PIL import Image

    out = []
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
        out += [s.strip() for s in processor.batch_decode(
            gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--high", action="store_true")
    ap.add_argument("--holdout-axis", default="random", choices=["random", "arm", "gripper"])
    ap.add_argument("--n-heldout", type=int, default=12)
    ap.add_argument("--bins", type=int, default=4)
    ap.add_argument("--per-bin", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    table = exclude(usable(build_table(DEFAULT)), grippers=["xarm7_gripper"], subset=DEFAULT)
    table = cached_table(table, DEFAULT, args.high)
    train_emb, heldout = split_embodiments(table, args.n_heldout, seed=0, subset=DEFAULT,
                                           axis=args.holdout_axis)

    # thresholds from TRAINING rows, exactly as the trainer fitted them
    train_table = table[table.embodiment.isin(train_emb)].reset_index(drop=True)
    dp, dq, _ = deltas(train_table, "cam")
    describer = MotionDescriber(fit_thresholds(dp), fit_thresholds(quat_to_euler(dq)),
                                grip_threshold=0.5, frame="fixed")

    import glob

    import pandas as pd

    raw = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index",
                                                 "observation.eef_pixel"])
                     for f in sorted(glob.glob(str(
                         Path("/dataset/jiyun/dataset_git/eef_pairs") / DEFAULT /
                         "data" / "**" / "*.parquet"), recursive=True))], ignore_index=True)
    lengths = raw.groupby("episode_index").size().sort_index()
    starts = lengths.cumsum().shift(fill_value=0)
    raw["row"] = raw.episode_index.map(starts) + raw.frame_index
    pix = dict(zip(raw["row"].to_numpy(),
                   np.stack(raw["observation.eef_pixel"].to_numpy())[:, 0], strict=True))

    splits = {"seen": table[table.embodiment.isin(train_emb)].reset_index(drop=True),
              "heldout": table[table.embodiment.isin(heldout)].reset_index(drop=True)}
    for name, frame in splits.items():
        frame["u"] = [pix[int(r)] for r in frame["row"].to_numpy()]

    # Bin edges come from the HELD-OUT split and are applied to both, so the two are compared on
    # the same position distribution rather than each on its own.
    mode = float(np.median(splits["heldout"]["u"]))
    edges = np.quantile(np.abs(splits["heldout"]["u"] - mode), np.linspace(0, 1, args.bins + 1))
    log(f"seen {len(splits['seen'])} / held-out {len(splits['heldout'])} pairs, "
        f"u median {mode:.0f}, |u-mode| edges {np.round(edges, 0).tolist()}")

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    processor.image_processor.do_image_splitting = False
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    model.eval()
    cache = load_cache(DEFAULT, args.high)

    rng = np.random.default_rng(args.seed)
    results = {"tag": args.tag, "u_mode": mode, "bins": []}
    log(f"{'bin':<5}{'|u-mode|':>11}{'u range':>15}"
        f"{'SEEN all':>10}{'HELD all':>10}{'gap':>8}{'SEEN tr':>9}{'HELD tr':>9}")
    for b in range(args.bins):
        lo, hi = edges[b], edges[b + 1]
        row = {"bin": b, "lo": float(lo), "hi": float(hi)}
        for name, frame in splits.items():
            distance = np.abs(frame["u"].to_numpy() - mode)
            mask = (distance >= lo) & (distance <= hi if b == args.bins - 1 else distance < hi)
            pool = frame[mask]
            if len(pool) == 0:
                continue
            pick = pool.iloc[rng.choice(len(pool), size=min(args.per_bin, len(pool)),
                                        replace=False)]
            d_pos, d_quat, grip = deltas(pick, "cam")
            truth = describe_rows(describer, d_pos, d_quat, grip)
            preds = generate(model, processor, cache, pick["cache_pos"].to_numpy(), device)
            s = score(preds, truth)
            row[name] = {"n": int(len(pick)), "u_min": float(pool["u"].min()),
                         "u_max": float(pool["u"].max()), "distinct_pred": len(set(preds)),
                         "distinct_truth": len(set(truth)), **s}
        if "seen" not in row or "heldout" not in row:
            continue
        results["bins"].append(row)
        held_row, seen_row = row["heldout"], row["seen"]
        span = f"{held_row['u_min']:.0f}-{held_row['u_max']:.0f}"
        log(f"{b:<5}{f'{lo:.0f}-{hi:.0f}':>11}{span:>15}"
            f"{seen_row['all']:>10.3f}{held_row['all']:>10.3f}"
            f"{seen_row['all'] - held_row['all']:>+8.3f}"
            f"{seen_row['translation']:>9.3f}{held_row['translation']:>9.3f}")
    out = REPO_ROOT / f"outputs/vlm_region_{args.tag}.json"
    out.write_text(json.dumps(results, indent=2))
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
