#!/usr/bin/env python
"""Print the English labels the LAP objective would train on, straight from the corpus.

    python tools/preview_lap_language_actions.py
    python tools/preview_lap_language_actions.py --robot ur5e --num 15 --style numeric

Reads raw (unnormalized) actions from the prepared parquet, which is exactly what the policy
rebuilds with `unnormalize_actions` before describing them. Use this to sanity-check the
direction words, the magnitude buckets and the gripper sign before spending GPU-days on them.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lerobot.policies.smolvla.language_action_text import LanguageActionTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "dataset_git/barx_frontonly_p900_i1000_u1000/raw"


class _NoTokenizer:
    """`describe()` needs no tokenizer; this keeps the preview free of model downloads."""

    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="all", choices=["all", "panda_mg", "iiwa", "ur5e"])
    parser.add_argument("--chunk", type=int, default=50)
    parser.add_argument("--num", type=int, default=8, help="sentences printed per robot")
    parser.add_argument("--style", default="rough", choices=["rough", "numeric"])
    parser.add_argument("--cm-per-unit", type=float, default=0.0)
    args = parser.parse_args()

    labeller = LanguageActionTokenizer(
        text_tokenizer=_NoTokenizer(),
        style=args.style,
        cm_per_unit=args.cm_per_unit,
    )

    robots = ["panda_mg", "iiwa", "ur5e"] if args.robot == "all" else [args.robot]
    for robot in robots:
        path = ROOT / robot / "data/chunk-000/file-000.parquet"
        df = pd.read_parquet(path, columns=["episode_index", "action"])
        print(f"\n=== {robot} ===")
        counter = Counter()
        shown = 0
        for episode in df.episode_index.unique()[:30]:
            actions = np.stack(df[df.episode_index == episode]["action"].values)
            for start in range(0, len(actions) - args.chunk, args.chunk):
                chunk = actions[start : start + args.chunk]
                sentence = labeller.describe(chunk)
                counter[sentence.split(",")[0]] += 1
                if shown < args.num and episode == df.episode_index.unique()[0]:
                    totals = chunk.sum(axis=0)
                    print(
                        f"  t={start:4d}  sum(xyz)=[{totals[5]:+6.1f} {totals[6]:+6.1f} {totals[7]:+6.1f}]"
                        f"  grip={chunk[-1, 11]:+.0f}  ->  {sentence}"
                    )
                    shown += 1
        print(f"  most common opening phrase over 30 episodes:")
        for phrase, count in counter.most_common(5):
            print(f"    {count:5d}  {phrase}")


if __name__ == "__main__":
    raise SystemExit(main())
