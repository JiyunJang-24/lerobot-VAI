#!/usr/bin/env python
"""Train the VLM itself on the two-image motion task, answering in LAP-style English.

    [image_t, image_t+h] + "How did the gripper move from the first image to the second?"
        -> "move forward moderately, move right slightly, close gripper"

Why the VLM and not the MLP head already in train_motion_prediction.py:

  * the head trains only the 86M vision tower; this trains the 362M text model as well
  * 10.7b found the head transfers DIRECTION to a novel arm but not MAGNITUDE, because pixel
    displacement scale depends on how large the arm looks -- exactly what a new morphology changes.
    Language quantises magnitude into three buckets, a coarser claim that may survive it.
  * the output is the same architecture the policy uses, so a VLM pre-trained this way can
    initialise SmolVLA directly

Cost is not the obstacle it looks like. The connector pools each image's 1024 patches to 64 tokens,
and `do_image_splitting` is turned OFF because these frames are 180x320 and tiling them into 26
crops inflates one sample from 165 tokens to 1755 for no added detail.

Loss is cross-entropy on the ANSWER tokens only -- the prompt and the 128 image placeholders are
masked out, or the model would spend its capacity re-predicting a fixed prompt.

    python src/lerobot/scripts/train_vlm_motion.py --tag vlm_synth --steps 4000
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

from lerobot.scripts.motion_data import (  # noqa: E402
    EEF_ROOT, assert_no_leakage, build_table, cache_path, sample_pose_pairs,
)
from lerobot.scripts.motion_language import (  # noqa: E402
    MotionDescriber, content_words, fit_thresholds, motion_deltas,
)

MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
QUESTION = "How did the gripper move from the first image to the second?"
SUBSETS = ["56combo_144_bg12_closed", "56combo_144_bg12_open",
           "56combo_144_bg12_closed_furniture", "56combo_144_bg12_open_furniture"]
N_BG, N_VIEW = 12, 4


def log(msg: str) -> None:
    print(f"[vlm_motion] {msg}", flush=True)


def build_index(table, subsets):
    n_emb = int(table.embodiment.max()) + 1
    n_pose = int(table.pose.max()) + 1
    n_color = int(table.color.max()) + 1
    # The colour axis is REQUIRED, not optional. Without it, three rows -- the three robot paint
    # jobs -- collapse into one cell and only the last survives, so two thirds of every render is
    # unreachable and a pair drawn across two cells can silently change the robot's colour.
    index = np.full((n_emb, n_pose, len(subsets), N_BG, N_VIEW, n_color), -1, dtype=np.int64)
    sub_id = {s: i for i, s in enumerate(subsets)}
    index[table.embodiment.to_numpy(), table.pose.to_numpy(),
          table.subset.map(sub_id).to_numpy(), table.background.to_numpy(),
          table.view.to_numpy(), table.color.to_numpy()] = table.cache_pos.to_numpy()
    return index


def make_batcher(index, pairs, embodiments, rng, vary=("background", "furniture")):
    """Draw pairs. What is allowed to differ between the two frames is the design decision here.

    ALWAYS held identical, because they define "the same robot":
        embodiment      the arm and gripper morphology
        color_variant   the robot's paint. cv 0/1/2 repaint the SAME arm silver / yellow / pink,
                        so letting it change makes the model solve a correspondence puzzle that
                        never occurs in a real trajectory.

    Optionally allowed to differ, as nuisance-appearance augmentation:
        background      the backdrop only -- verified not to touch the robot
        furniture       adds scene objects; the arm configuration is untouched
        view            the camera pose. OFF by default and different in kind from the other two:
                        a real camera does not teleport mid-motion, and the label is a 3-D
                        displacement that would have to be inferred across a viewpoint change.
    """
    n_sub, n_color = index.shape[2], index.shape[5]
    vary = set(vary)

    def draw(n):
        out = np.empty((n, 10), dtype=np.int64)
        filled = 0
        while filled < n:
            k = (n - filled) * 3
            emb = rng.choice(embodiments, size=k)
            color = rng.integers(0, n_color, size=k)          # the robot: identical in both frames
            pair = pairs[rng.integers(0, len(pairs), size=k)]
            # subset = furniture bit * 2 + gripper bit; the gripper always may differ, since
            # opening or closing it is part of the motion being described.
            furn_t = rng.integers(0, max(1, n_sub // 2), size=k) * 2
            furn_h = (rng.integers(0, max(1, n_sub // 2), size=k) * 2
                      if "furniture" in vary else furn_t)
            sub_t = furn_t + rng.integers(0, min(2, n_sub), size=k)
            sub_h = furn_h + rng.integers(0, min(2, n_sub), size=k)
            bg_t = rng.integers(0, N_BG, size=k)
            bg_h = rng.integers(0, N_BG, size=k) if "background" in vary else bg_t
            view_t = rng.integers(0, N_VIEW, size=k)
            view_h = rng.integers(0, N_VIEW, size=k) if "view" in vary else view_t
            pos_t = index[emb, pair[:, 0], sub_t, bg_t, view_t, color]
            pos_h = index[emb, pair[:, 1], sub_h, bg_h, view_h, color]
            ok = (pos_t >= 0) & (pos_h >= 0)
            take = min(int(ok.sum()), n - filled)
            sel = np.where(ok)[0][:take]
            out[filled:filled + take] = np.stack(
                [emb[sel], pair[sel, 0], pair[sel, 1], sub_t[sel], sub_h[sel],
                 bg_t[sel], bg_h[sel], view_t[sel], view_h[sel], color[sel]], 1)
            filled += take
        return out
    return draw


def positions_of(index, spec):
    """spec columns: emb, pose_t, pose_h, sub_t, sub_h, bg_t, bg_h, view_t, view_h, color."""
    pos_t = index[spec[:, 0], spec[:, 1], spec[:, 3], spec[:, 5], spec[:, 7], spec[:, 9]]
    pos_h = index[spec[:, 0], spec[:, 2], spec[:, 4], spec[:, 6], spec[:, 8], spec[:, 9]]
    return pos_t, pos_h


def encode_batch(processor, images_t, images_h, answers, device):
    """Tokenise, and mask everything that is not the answer."""
    from PIL import Image

    texts, image_lists = [], []
    for answer in answers:
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                         {"type": "text", "text": QUESTION}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        texts.append(processor.apply_chat_template(messages, tokenize=False))
    for t, h in zip(images_t, images_h, strict=True):
        image_lists.append([Image.fromarray(t.permute(1, 2, 0).numpy()),
                            Image.fromarray(h.permute(1, 2, 0).numpy())])
    batch = processor(text=texts, images=image_lists, return_tensors="pt", padding=True)

    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    # mask the image placeholders, and everything up to and including "Assistant:"
    image_token = processor.tokenizer.convert_tokens_to_ids("<image>")
    labels[labels == image_token] = -100
    marker = processor.tokenizer("Assistant:", add_special_tokens=False)["input_ids"]
    for row in range(labels.shape[0]):
        ids = batch["input_ids"][row].tolist()
        start = 0
        for i in range(len(ids) - len(marker)):
            if ids[i:i + len(marker)] == marker:
                start = i + len(marker)
        labels[row, :start] = -100
    batch = {k: v.to(device) for k, v in batch.items()}
    batch["labels"] = labels.to(device)
    return batch


@torch.no_grad()
def evaluate(model, processor, cache, index, pairs, states, describer, embodiments, rng,
             device, n, batch_size, vary=("background", "furniture")):
    """Content-word accuracy: does the generated sentence use the right direction/size words?

    Not token accuracy -- that rewards reproducing "move" and the commas. CLAUDE.md 8 used the same
    measure for the earlier LAP objective, so the numbers are comparable.
    """
    model.eval()
    draw = make_batcher(index, pairs, np.array(embodiments), rng, vary)
    spec = draw(n)
    exact, jaccard, count = 0, 0.0, 0
    for start in range(0, n, batch_size):
        chunk = spec[start:start + batch_size]
        pos_t, pos_h = positions_of(index, chunk)
        s_t = np.concatenate([states[chunk[:, 1]], (chunk[:, 3:4] % 2 == 0).astype(float)], axis=1)
        s_h = np.concatenate([states[chunk[:, 2]], (chunk[:, 4:5] % 2 == 0).astype(float)], axis=1)
        truth = describer.describe(s_t, s_h)
        from PIL import Image

        texts = [processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                          {"type": "text", "text": QUESTION}]}],
            tokenize=False, add_generation_prompt=True) for _ in truth]
        images = [[Image.fromarray(cache[int(a)].permute(1, 2, 0).numpy()),
                   Image.fromarray(cache[int(b)].permute(1, 2, 0).numpy())]
                  for a, b in zip(pos_t, pos_h, strict=True)]
        enc = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, max_new_tokens=40, do_sample=False)
        gen = processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for pred, want in zip(gen, truth, strict=True):
            p, w = content_words(pred), content_words(want)
            exact += int(p == w)
            jaccard += len(p & w) / max(len(p | w), 1)
            count += 1
    model.train()
    return {"exact_set_match": exact / count, "content_word_jaccard": jaccard / count, "n": count}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--heldout-embodiments", default="0,1,3,8,12,14,23,27,28,33,34,36,42,49")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--motion-horizon", type=float, default=0.15)
    ap.add_argument("--pose-pairs", type=int, default=8000)
    ap.add_argument("--no-rotation", action="store_true",
                    help="drop rotation words. Real trajectories rotate ~6 deg over the horizon and "
                         "half of that cancels out, so those words carry little signal -- see 10.11")
    ap.add_argument("--vary", default="background,furniture",
                    help="which nuisance axes may differ between the two frames. The robot "
                         "(embodiment + color_variant) is always held fixed. Add 'view' to also "
                         "move the camera, which is a much harder and unphysical variant.")
    ap.add_argument("--eval-n", type=int, default=64)
    ap.add_argument("--eval-freq", type=int, default=1000)
    ap.add_argument("--log-freq", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "outputs/vlm_motion"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    table = build_table(SUBSETS)
    index = build_index(table, SUBSETS)
    first = table.drop_duplicates("pose").sort_values("pose")
    states = np.stack(first["observation.state"].to_numpy()).astype(np.float64)[:, :7]

    pairs = sample_pose_pairs(states, args.pose_pairs, args.motion_horizon, rng)
    heldout = [int(x) for x in args.heldout_embodiments.split(",") if x != ""]
    train_emb = sorted(set(range(int(table.embodiment.max()) + 1)) - set(heldout))
    assert_no_leakage(train_emb, heldout)

    # thresholds from the TRAINING pairs only
    padded_a = np.concatenate([states[pairs[:, 0]], np.zeros((len(pairs), 1))], axis=1)
    padded_b = np.concatenate([states[pairs[:, 1]], np.zeros((len(pairs), 1))], axis=1)
    d_pos, d_euler, _ = motion_deltas(padded_a, padded_b, "fixed")
    describer = MotionDescriber(fit_thresholds(d_pos), fit_thresholds(d_euler),
                                grip_threshold=0.5, include_rotation=not args.no_rotation,
                                frame="fixed")

    path = cache_path(SUBSETS)
    if not path.exists():
        raise FileNotFoundError(f"{path} -- build it with tools/build_motion_cache.py --subsets "
                                f"{','.join(SUBSETS)}")
    log(f"loading {path} ({path.stat().st_size / 1e9:.0f} GB) ...")
    cache = torch.load(path)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL)
    # 180x320 frames do not need tiling; leaving it on inflates one sample 165 -> 1755 tokens.
    processor.image_processor.do_image_splitting = False
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    model.train()
    model.config.use_cache = False
    log(f"VLM {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params, "
        f"rotation words {'OFF' if args.no_rotation else 'ON'}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    vary = tuple(x for x in args.vary.split(",") if x)
    log(f"nuisance axes allowed to differ between the two frames: {vary or 'none'}")
    draw = make_batcher(index, pairs, np.array(train_emb), rng, vary)

    history, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        spec = draw(args.batch_size)
        pos_t, pos_h = positions_of(index, spec)
        # gripper: subsets index 0 = closed, 1 = open; describer wants 1 = closed
        s_t = np.concatenate([states[spec[:, 1]], (spec[:, 3:4] % 2 == 0).astype(float)], axis=1)
        s_h = np.concatenate([states[spec[:, 2]], (spec[:, 4:5] % 2 == 0).astype(float)], axis=1)
        answers = describer.describe(s_t, s_h)
        batch = encode_batch(processor, cache[pos_t], cache[pos_h], answers, device)
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
            seen = evaluate(model, processor, cache, index, pairs, states, describer,
                            train_emb[:14], np.random.default_rng(7), device, args.eval_n, 8, vary)
            held = evaluate(model, processor, cache, index, pairs, states, describer,
                            heldout, np.random.default_rng(7), device, args.eval_n, 8, vary)
            log(f"  eval @{step}: seen exact {seen['exact_set_match']:.3f} "
                f"jaccard {seen['content_word_jaccard']:.3f}  |  HELD-OUT exact "
                f"{held['exact_set_match']:.3f} jaccard {held['content_word_jaccard']:.3f}")
            history[-1]["seen"] = seen
            history[-1]["heldout"] = held
            (out_dir / "results.json").write_text(json.dumps(
                {"args": vars(args), "train_embodiments": train_emb,
                 "heldout_embodiments": heldout, "history": history}, indent=2, default=str))

    model.save_pretrained(out_dir / "model")
    processor.save_pretrained(out_dir / "model")
    log(f"done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
