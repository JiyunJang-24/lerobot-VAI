#!/usr/bin/env python
"""Write the deterministic Experiment 1 split once, to disk, and never again.

The split is the experiment. Two decisions in it matter more than any hyper-parameter:

  SCENES are held out whole. selfws_v2/kitchen renders 32 distinct RoboCasa kitchens, so a
  held-out scene is a kitchen the encoder has never seen -- different walls, counters, sink,
  camera pose.

  EMBODIMENTS are held out BY ARM, not by arm-gripper combination. The earlier eef_pairs work
  held out 14 of 56 combinations while every arm and every gripper still appeared somewhere in
  training, so "+0.974 on held-out embodiments" measured generalisation to an unfamiliar
  combination of familiar parts. Holding out the arm is the claim that was actually meant.

    python tools/exp1_split.py --train-scenes 24 --heldout-arms JacoOmron,VX300SMobile
"""

import argparse
import json
from pathlib import Path

CACHE = Path("/dataset/jiyun/exp1_cache")


def arm_of(embodiment: str) -> str:
    return embodiment.split("_")[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=CACHE / "selfws_kitchen")
    ap.add_argument("--train-scenes", type=int, default=24)
    ap.add_argument("--heldout-arms", default="JacoOmron,VX300SMobile")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/exp1/split.json"))
    args = ap.parse_args()

    embs = json.loads((args.cache / "embodiments.json").read_text())
    scenes = sorted(int(p.stem[2:]) for p in args.cache.glob("ep*.npz"))
    arms = sorted({arm_of(e) for e in embs})

    heldout_arms = [a for a in args.heldout_arms.split(",") if a]
    unknown = [a for a in heldout_arms if a not in arms]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; available {arms}")

    # scenes: seeded permutation, so the split is reproducible without being the identity order
    import numpy as np
    order = np.random.default_rng(args.seed).permutation(scenes).tolist()
    train_scenes = sorted(order[: args.train_scenes])
    heldout_scenes = sorted(order[args.train_scenes :])

    train_embs = sorted(e for e in embs if arm_of(e) not in heldout_arms)
    heldout_embs = sorted(e for e in embs if arm_of(e) in heldout_arms)

    split = {
        "cache": str(args.cache),
        "seed": args.seed,
        "arms": arms,
        "embodiments": embs,
        "heldout_arms": heldout_arms,
        "train_embodiments": train_embs,
        "heldout_embodiments": heldout_embs,
        "train_scenes": train_scenes,
        "heldout_scenes": heldout_scenes,
        "groups": {
            "seen_scene/seen_emb": {"scenes": train_scenes, "embodiments": train_embs},
            "unseen_scene/seen_emb": {"scenes": heldout_scenes, "embodiments": train_embs},
            "seen_scene/unseen_emb": {"scenes": train_scenes, "embodiments": heldout_embs},
            "unseen_scene/unseen_emb": {"scenes": heldout_scenes, "embodiments": heldout_embs},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(split, indent=2))
    print(f"[split] {len(train_scenes)} train / {len(heldout_scenes)} held-out scenes")
    print(f"[split] {len(train_embs)} train / {len(heldout_embs)} held-out embodiments")
    print(f"[split] held-out arms: {heldout_arms}")
    print(f"[split] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
