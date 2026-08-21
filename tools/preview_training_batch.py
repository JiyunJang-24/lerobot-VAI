#!/usr/bin/env python
"""Show what one training step actually feeds the model, on both paths at once.

The run underway optimises a single loss built from two very different batches, and they are easy
to conflate. This dumps a real sample of each, side by side, with the exact text the model reads:

  POLICY path  (barx_pnpsink corpus, 48 frames/rank)
      in   one front-camera image, the task sentence, the 16-d state
      out  a 50-step action chunk (flow matching) and -- under knowledge insulation -- an English
           sentence describing that chunk, which the LM head must predict

  VQA path     (visual-robust renders, 12 frames x 6 embodiments)
      in   one render, the question
      out  the gripper-state sentence, identical across all six renders of a frame

    python tools/preview_training_batch.py
"""

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset  # noqa: E402
from lerobot.policies.smolvla.language_action_text import LanguageActionTokenizer  # noqa: E402
from lerobot.policies.smolvla.vqa_state_text import (  # noqa: E402
    VQAStateTokenizer,
    quaternion_xyzw_to_yaw_deg,
    wrap_deg,
)

POLICY_ROOT = REPO_ROOT / "dataset_git/barx_pnpsink_p900_i1000_u1000/raw"
POLICY_REPOS = ["panda_mg", "iiwa", "ur5e"]
VR_ROOT = REPO_ROOT / "dataset_git/visual_robust_new_barx_ur5e/new_barx"
VR_REPO = "UR5eOmron_PnPSinkToCounter/lerobot"
CHUNK = 50


class _NoTokenizer:
    eos_token_id, pad_token_id = 0, 0

    def encode(self, text, add_special_tokens=False):
        return []


def wrap(text, width=64, indent=" " * 9):
    return textwrap.fill(text, width=width, subsequent_indent=indent)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-samples", type=int, default=3)
    ap.add_argument("--vqa-frames", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/training_batch_preview.png")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # ---- policy path -------------------------------------------------------------------------
    policy_ds = MultiLeRobotDataset(
        POLICY_REPOS, root=POLICY_ROOT,
        delta_timestamps={r: {"action": [i / 20 for i in range(CHUNK)]} for r in POLICY_REPOS},
        visual_cue_mode="vanilla", use_wrist_cam=False, use_state=True, cache_in_memory=False,
    )
    lap = LanguageActionTokenizer(_NoTokenizer(), include_rotation=False)
    policy_rows = []
    for idx in rng.choice(len(policy_ds), args.policy_samples, replace=False):
        item = policy_ds[int(idx)]
        action = item["action"].numpy()
        policy_rows.append({
            "image": item["observation.images.robot0_agentview_right"],
            "robot": POLICY_REPOS[int(item["dataset_index"])],
            "task": item["task"],
            "state": item["observation.state"].numpy(),
            "action": action,
            # The label the LM head is trained on under --policy.ki_objective=lap. Built from the
            # RAW action here; in training the policy rebuilds it by un-normalizing first.
            "lap": lap.describe(action),
        })

    # ---- VQA path ----------------------------------------------------------------------------
    vr_ds = MultiLeRobotDataset(
        [VR_REPO], root=VR_ROOT, delta_timestamps={VR_REPO: None},
        visual_cue_mode="vanilla", use_wrist_cam=False, use_state=True, cache_in_memory=False,
    )
    vqa = VQAStateTokenizer(_NoTokenizer())
    probe = vr_ds[0]
    view_keys = sorted(
        k for k, v in probe.items()
        if k.startswith("observation.image") and torch.is_tensor(v) and v.ndim == 3
        and "wrist" not in k and "eye_in_hand" not in k
    )
    episodes = vr_ds.meta_episodes
    vqa_rows = []
    for idx in rng.choice(len(vr_ds), args.vqa_frames, replace=False):
        item = vr_ds[int(idx)]
        episode = int(item["episode_index"])
        start = int(episodes["dataset_from_index"][episode])
        reference = vr_ds[start]["observation.state"].numpy()
        state = item["observation.state"].numpy()
        vqa_rows.append({
            "images": [item[k] for k in view_keys],
            "episode": episode,
            "answer": vqa.sentences(state[None, :], reference[None, :])[0],
            "raw": (
                100 * (state[0] - reference[0]), 100 * (state[1] - reference[1]), 100 * state[2],
                float(wrap_deg(quaternion_xyzw_to_yaw_deg(state[None, 3:7])
                               - quaternion_xyzw_to_yaw_deg(reference[None, 3:7]))[0]),
                state[7],
            ),
        })

    # ---- print ---------------------------------------------------------------------------------
    print("=" * 100)
    print("POLICY path -- one image + task sentence + state  ->  50-step action chunk (+ LAP sentence)")
    print("=" * 100)
    for r in policy_rows:
        a = r["action"]
        print(f"\n[{r['robot']}]")
        print(f"  IN  image  {tuple(r['image'].shape)}  range [{r['image'].min():.2f}, {r['image'].max():.2f}]")
        print(f"  IN  task   {wrap(r['task'])}")
        print(f"  IN  state  16-d, eef xyz = {np.round(r['state'][7:10], 3)}  gripper = {np.round(r['state'][14:16], 2)}")
        print(f"  GT  action {a.shape}  arm xyz sum = {np.round(a[:, 5:8].sum(0), 2)}  gripper last = {a[-1, 11]:+.0f}")
        print(f"  GT  LAP    \"{r['lap']}\"")

    print("\n" + "=" * 100)
    print(f"VQA path -- one render + question  ->  gripper-state sentence ({len(view_keys)} renders share it)")
    print("=" * 100)
    print(f"\n  IN  question  \"{VQAStateTokenizer.QUESTION}\"")
    for r in vqa_rows:
        dx, dy, z, dyaw, grip = r["raw"]
        print(f"\n[episode {r['episode']}]  renders: {[k.split('.')[2] for k in view_keys]}")
        print(f"  raw   dx={dx:+.1f}cm dy={dy:+.1f}cm z={z:.1f}cm dyaw={dyaw:+.1f}deg grip={grip:+.0f}")
        print(f"  GT    \"{r['answer']}\"")

    # ---- figure --------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_vqa_cols = len(view_keys)
    fig = plt.figure(figsize=(3.1 * n_vqa_cols, 3.4 * (len(policy_rows) + len(vqa_rows)) + 1.2))
    grid = fig.add_gridspec(len(policy_rows) + len(vqa_rows), n_vqa_cols, hspace=0.75)

    for r, row in enumerate(policy_rows):
        ax = fig.add_subplot(grid[r, 0])
        ax.imshow(row["image"].permute(1, 2, 0).numpy().clip(0, 1))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"POLICY · {row['robot']}", fontsize=10, loc="left")
        text = (
            f'IN  task:   "{row["task"]}"\n'
            f'IN  state:  eef xyz {np.round(row["state"][7:10], 3)}\n'
            f'GT  action: {row["action"].shape[0]} steps x {row["action"].shape[1]} dims\n'
            f'GT  LAP:    "{row["lap"]}"'
        )
        ax_t = fig.add_subplot(grid[r, 1:])
        ax_t.axis("off")
        ax_t.text(0, 1, text, va="top", ha="left", fontsize=10, family="monospace", wrap=True)

    for r, row in enumerate(vqa_rows):
        base = len(policy_rows) + r
        for c, img in enumerate(row["images"]):
            ax = fig.add_subplot(grid[base, c])
            ax.imshow(img.permute(1, 2, 0).numpy().clip(0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(view_keys[c].split(".")[2], fontsize=8)
            if c == 0:
                ax.set_ylabel(f"VQA · ep {row['episode']}", fontsize=9)
        ax0 = fig.axes[-n_vqa_cols]
        ax0.text(
            0, -0.16,
            f'IN  Q: "{VQAStateTokenizer.QUESTION}"\nGT  A: "{row["answer"]}"',
            transform=ax0.transAxes, fontsize=10, va="top", ha="left", family="monospace",
        )

    fig.suptitle(
        "one optimisation step: POLICY batch (flow matching + LAP token CE) and VQA batch, summed into one loss",
        fontsize=12,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=105, bbox_inches="tight")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
