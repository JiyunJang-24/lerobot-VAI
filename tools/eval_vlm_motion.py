#!/usr/bin/env python
"""Score a VLM on the two-image motion question, on synthetic held-out robots AND on real Jaco.

Works with an untrained checkpoint too, which is the point: the zero-shot number is the floor every
trained result has to beat, and without it "0.42 jaccard" means nothing.

    python tools/eval_vlm_motion.py --model stock       --domain both
    python tools/eval_vlm_motion.py --model outputs/vlm_motion/rot_on/model --domain both

WHY THE SCORE IS SPLIT BY WORD TYPE
  The synthetic and real corpora agree on what translation words mean -- bucket thresholds
  0.028/0.073/0.110 m versus 0.026/0.072/0.107, within 7% -- but disagree 7x on rotation
  (9.7/29.2/63.8 degrees versus 1.0/3.9/8.8). A model trained on synthetic rotation words is
  therefore answering a DIFFERENT question on real data, and a single blended score would hide
  that. Translation, rotation and gripper are scored separately.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.scripts.motion_data import (  # noqa: E402
    EEF_ROOT, build_table, load_cache, sample_pose_pairs,
)
from lerobot.scripts.motion_language import (  # noqa: E402
    AXIS_WORDS, MAGNITUDES, MotionDescriber, fit_thresholds, motion_deltas,
)
from lerobot.scripts.train_vlm_motion import (  # noqa: E402
    MODEL, QUESTION, SUBSETS, build_index, make_batcher, positions_of,
)

VR = Path("/dataset/jiyun/dataset_git/visual_robust_new_barx_ur5e/new_barx/"
          "UR5eOmron_PnPSinkToCounter/lerobot")
VR_REPO = "visual_robust/UR5eOmron_PnPSinkToCounter"
JACO = ["JacoOmron", "JacoOmronPandaGripper"]
REAL_EMB = ["PandaOmron", "IIWAOmron", "UR5eOmron", "PandaOmronPandaGripper", *JACO]

TRANSLATION = {w for pair in AXIS_WORDS for w in pair}
ROTATION = {"roll", "tilt", "turn", "left", "right", "up", "down"} - TRANSLATION
GRIPPER = {"open", "close"}


def log(msg: str) -> None:
    print(f"[eval_vlm] {msg}", flush=True)


def split_words(sentence: str):
    """Separate a sentence into translation / rotation / gripper claims.

    Parsed by clause rather than by bare word: "move left slightly" and "roll left slightly" share
    the token "left", so a bag-of-words split would mix the two categories.
    """
    trans, rot, grip = set(), set(), set()
    for clause in sentence.split(","):
        parts = clause.strip().split()
        if not parts:
            continue
        if parts[0] == "move" and len(parts) >= 3:
            trans.add(" ".join(parts[1:3]))
        elif parts[0] in {"roll", "tilt", "turn"} and len(parts) >= 3:
            rot.add(" ".join(parts[:3]))
        elif parts[0] in GRIPPER:
            grip.add(parts[0])
    return trans, rot, grip


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def score(preds, truths):
    """Per-category Jaccard, plus a gripper score computed only where it means anything.

    Most pairs mention no gripper change at all, and two empty sets score a perfect 1.0, so a plain
    gripper average sits near 0.9 for a model that never emits the word -- the stock VLM scores
    0.927 that way. `gripper_when_present` restricts to pairs whose TRUTH has a gripper word, which
    is the only subset on which the number can distinguish anything.
    """
    out = {"translation": [], "rotation": [], "gripper": [], "all": [], "gripper_when_present": []}
    for pred, want in zip(preds, truths, strict=True):
        pt, pr, pg = split_words(pred)
        tt, tr, tg = split_words(want)
        out["translation"].append(jaccard(pt, tt))
        out["rotation"].append(jaccard(pr, tr))
        out["gripper"].append(jaccard(pg, tg))
        out["all"].append(jaccard(pt | pr | pg, tt | tr | tg))
        if tg:
            out["gripper_when_present"].append(jaccard(pg, tg))
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in out.items()}


@torch.no_grad()
def generate(model, processor, images_t, images_h, device, batch=8):
    from PIL import Image

    out = []
    for start in range(0, len(images_t), batch):
        t = images_t[start:start + batch]
        h = images_h[start:start + batch]
        texts = [processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                          {"type": "text", "text": QUESTION}]}],
            tokenize=False, add_generation_prompt=True) for _ in t]
        images = [[Image.fromarray(a.permute(1, 2, 0).numpy()),
                   Image.fromarray(b.permute(1, 2, 0).numpy())] for a, b in zip(t, h, strict=True)]
        enc = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
        gen = model.generate(**enc, max_new_tokens=48, do_sample=False)
        out += processor.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [s.strip() for s in out]


def synthetic_eval(model, processor, device, n, seed):
    table = build_table(SUBSETS)
    index = build_index(table, SUBSETS)
    first = table.drop_duplicates("pose").sort_values("pose")
    states = np.stack(first["observation.state"].to_numpy()).astype(np.float64)[:, :7]
    rng = np.random.default_rng(seed)
    pairs = sample_pose_pairs(states, 8000, 0.15, rng)
    a = np.concatenate([states[pairs[:, 0]], np.zeros((len(pairs), 1))], axis=1)
    b = np.concatenate([states[pairs[:, 1]], np.zeros((len(pairs), 1))], axis=1)
    d_pos, d_euler, _ = motion_deltas(a, b, "fixed")
    describer = MotionDescriber(fit_thresholds(d_pos), fit_thresholds(d_euler), 0.5, frame="fixed")
    heldout = [0, 1, 3, 8, 12, 14, 23, 27, 28, 33, 34, 36, 42, 49]
    train_emb = sorted(set(range(int(table.embodiment.max()) + 1)) - set(heldout))
    cache = load_cache(SUBSETS)

    results = {}
    for name, embs in (("seen", train_emb[:14]), ("heldout", heldout)):
        spec = make_batcher(index, pairs, np.array(embs), np.random.default_rng(7))(n)
        pos_t, pos_h = positions_of(index, spec)
        s_t = np.concatenate([states[spec[:, 1]], (spec[:, 3:4] % 2 == 0).astype(float)], axis=1)
        s_h = np.concatenate([states[spec[:, 2]], (spec[:, 4:5] % 2 == 0).astype(float)], axis=1)
        truth = describer.describe(s_t, s_h)
        preds = generate(model, processor, cache[pos_t], cache[pos_h], device)
        results[name] = score(preds, truth)
        results[name]["n"] = int(n)
        log(f"  synthetic {name:<8} " + "  ".join(
            f"{k} {results[name][k]:.3f}" for k in
            ("all", "translation", "rotation", "gripper_when_present")))
        if name == "heldout":
            for p, t in list(zip(preds, truth, strict=True))[:3]:
                log(f'      pred "{p[:88]}"')
                log(f'      true "{t[:88]}"')
    return results


def real_eval(model, processor, device, n_per_emb, horizon, stride, seed):
    files = sorted(glob.glob(str(VR / "data" / "**" / "*.parquet"), recursive=True))
    table = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    table["row"] = np.arange(len(table))
    states = np.stack(table["observation.state"].to_numpy()).astype(np.float64)
    episodes = table.episode_index.to_numpy()
    anchors, partners = [], []
    for ep in np.unique(episodes):
        rows = table.row.to_numpy()[episodes == ep]
        usable = rows[: len(rows) - horizon][::stride]
        anchors.append(usable)
        partners.append(usable + horizon)
    anchors, partners = np.concatenate(anchors), np.concatenate(partners)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(anchors), size=min(n_per_emb, len(anchors)), replace=False)
    anchors, partners = anchors[pick], partners[pick]

    # World xyz with the base parked differently per episode, so rotate into the gripper's own
    # frame at time t -- episode-independent, and visible in the image (CLAUDE.md 10.9).
    d_pos, d_euler, _ = motion_deltas(states[anchors], states[partners], "eef")
    describer = MotionDescriber(fit_thresholds(d_pos), fit_thresholds(d_euler),
                               float(np.quantile(np.abs(states[partners, 7] - states[anchors, 7]),
                                                 0.9)) or 0.5, frame="eef")
    truth = describer.describe(states[anchors], states[partners])

    dataset = LeRobotDataset(VR_REPO, root=VR)
    results = {}
    for emb in REAL_EMB:
        key = f"observation.images.{emb}.robot0_agentview_right"

        def grab(rows, key=key):
            out = torch.empty(len(rows), 3, 180, 320, dtype=torch.uint8)
            for i, r in enumerate(rows):
                frame = dataset[int(r)][key]
                frame = frame[-1] if frame.ndim == 4 else frame
                out[i] = (frame * 255).round().clamp(0, 255).to(torch.uint8)
            return out

        preds = generate(model, processor, grab(anchors), grab(partners), device)
        results[emb] = score(preds, truth)
        results[emb]["split"] = "heldout_jaco" if emb in JACO else "seen"
        log(f"  real {emb:<24} {results[emb]['split']:<12} " + "  ".join(
            f"{k} {results[emb][k]:.3f}" for k in
            ("all", "translation", "rotation", "gripper_when_present")))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="stock", help="'stock' or a saved model directory")
    ap.add_argument("--domain", default="both", choices=["synthetic", "real", "both"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--real-n", type=int, default=96)
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    from transformers import AutoModelForImageTextToText, AutoProcessor

    source = MODEL if args.model == "stock" else args.model
    processor = AutoProcessor.from_pretrained(source if args.model != "stock" else MODEL)
    processor.image_processor.do_image_splitting = False
    model = AutoModelForImageTextToText.from_pretrained(source, dtype=torch.bfloat16).to(device)
    model.eval()
    tag = args.tag or ("stock" if args.model == "stock" else Path(args.model).parent.name)
    log(f"model: {source}   tag: {tag}")

    out = {"model": source, "tag": tag}
    if args.domain in ("synthetic", "both"):
        out["synthetic"] = synthetic_eval(model, processor, device, args.n, args.seed)
    if args.domain in ("real", "both"):
        out["real"] = real_eval(model, processor, device, args.real_n, args.horizon,
                                args.stride, args.seed)
    path = args.out or REPO_ROOT / f"outputs/vlm_eval_{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str))
    log(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
