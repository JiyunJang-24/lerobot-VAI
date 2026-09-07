#!/usr/bin/env python
"""Does the contrastive tower give the same feature for the same EEF state on the REAL Jaco?

The synthetic test (analyze_pose_invariance.py) answered this on eef_pairs renders, where the
answer was an emphatic yes. This is the same question on real robot images, which matters more:
the tower places a real RoboCasa frame 102% of the way from a synthetic frame to a completely
unrelated image, so whether the property survives the domain change is an open question, not a
formality.

visual_robust_new_barx_ur5e is the right corpus for it. The SAME 108 episodes are rendered from
six NAMED embodiments, and observation.state is shared across all six -- so one row is one EEF
state seen on six different arms, which is exactly the comparison. Jaco appears in two of them.

Measured, with the stock SigLIP as control:
  * same EEF state, Jaco vs each other embodiment
  * the contrast: different EEF state, same embodiment
  * retrieval -- given a Jaco frame, does the nearest Panda/IIWA/UR5e frame share the state?

    python tools/analyze_jaco_representation.py
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad  # noqa: E402

VR = Path("/dataset/jiyun/dataset_git/visual_robust_new_barx_ur5e/new_barx/"
          "UR5eOmron_PnPSinkToCounter/lerobot")
REPO_ID = "visual_robust/UR5eOmron_PnPSinkToCounter"
EMB = ["PandaOmron", "IIWAOmron", "UR5eOmron", "PandaOmronPandaGripper",
       "JacoOmron", "JacoOmronPandaGripper"]
JACO = {"JacoOmron", "JacoOmronPandaGripper"}


def log(msg: str) -> None:
    print(f"[jaco_rep] {msg}", flush=True)


def load_tower(path: str, device):
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText

    vlm = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", dtype=torch.float32)
    tower = vlm.model.vision_model
    if path:
        tower.load_state_dict(load_file(path), strict=True)
    return tower.to(device).eval()


@torch.no_grad()
def features(tower, images, device, batch=16):
    out = []
    for piece in images.split(batch):
        x = piece.to(device=device, dtype=torch.float32) / 255.0
        x = resize_with_pad(x, 512, 512, pad_value=0) * 2.0 - 1.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(tower(pixel_values=x, patch_attention_mask=None).last_hidden_state.mean(1))
    return torch.cat(out).float().cpu()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO_ROOT / "outputs/siglip_pretrain/all4_n42_all")
    ap.add_argument("--states", type=int, default=40, help="distinct EEF states to sample")
    ap.add_argument("--per-episode", type=int, default=4,
                    help="frames per episode; >1 is what makes the scene control possible")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/jaco_representation.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    files = sorted(glob.glob(str(VR / "data" / "**" / "*.parquet"), recursive=True))
    table = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index",
                                                   "observation.state"]) for f in files],
                      ignore_index=True)
    table["row"] = np.arange(len(table))
    states = np.stack(table["observation.state"].to_numpy()).astype(np.float64)

    rng = np.random.default_rng(args.seed)
    # Spread the sampled states across episodes: consecutive frames are nearly the same pose, and
    # sampling them would make "different EEF state" a much easier contrast than it should be.
    episodes = table.episode_index.to_numpy()
    chosen = []
    n_ep = max(2, args.states // args.per_episode)
    for ep in rng.choice(np.unique(episodes), size=min(n_ep, len(np.unique(episodes))),
                         replace=False):
        rows = np.sort(table.row.to_numpy()[episodes == ep])
        # spread within the episode so the within-episode pairs are genuinely different poses
        take = np.linspace(0, len(rows) - 1, args.per_episode).astype(int)
        chosen += [int(rows[t]) for t in np.unique(take)]
    chosen = np.array(sorted(chosen))
    log(f"{len(chosen)} EEF states, one per episode, each rendered from {len(EMB)} embodiments")

    dataset = LeRobotDataset(REPO_ID, root=VR)
    images, meta = [], []
    for emb in EMB:
        key = f"observation.images.{emb}.robot0_agentview_right"
        for r in chosen:
            img = dataset[int(r)][key]
            img = img[-1] if img.ndim == 4 else img
            images.append((img * 255).round().clamp(0, 255).to(torch.uint8))
            meta.append((int(r), emb))
    images = torch.stack(images)
    rows_of = np.array([m[0] for m in meta])
    emb_of = np.array([m[1] for m in meta])
    log(f"decoded {len(images)} frames")

    # "same EEF state" = the same source row, which carries one observation.state shared by every
    # render. Distances between states let us report how wrong a mismatch is, in centimetres.
    xyz = states[rows_of][:, :3]

    results = {"checkpoint": args.checkpoint.name, "n_states": len(chosen),
               "embodiments": EMB, "towers": {}}
    for name, path in (("contrastive (all4_n42_all)",
                        str(args.checkpoint / "vision_tower.safetensors")),
                       ("stock SigLIP (control)", "")):
        tower = load_tower(path, device)
        feats = features(tower, images, device)
        cen = F.normalize(feats - feats.mean(0, keepdim=True), dim=-1)
        raw = F.normalize(feats, dim=-1)
        sim = (cen @ cen.T).numpy()
        sim_raw = (raw @ raw.T).numpy()
        n = len(feats)
        eye = np.eye(n, dtype=bool)
        same_state = (rows_of[:, None] == rows_of[None, :]) & ~eye
        diff_state = rows_of[:, None] != rows_of[None, :]
        is_jaco = np.isin(emb_of, list(JACO))

        pairs = {}
        for label, mask in (
            ("same state, non-Jaco x non-Jaco", same_state & ~is_jaco[:, None] & ~is_jaco[None, :]),
            ("same state, non-Jaco x JACO", same_state & ~is_jaco[:, None] & is_jaco[None, :]),
            ("same state, JACO x JACO", same_state & is_jaco[:, None] & is_jaco[None, :]),
            ("DIFFERENT state, same embodiment",
             diff_state & (emb_of[:, None] == emb_of[None, :])),
        ):
            if mask.sum():
                pairs[label] = {"centred": float(sim[mask].mean()),
                                "raw": float(sim_raw[mask].mean()), "n": int(mask.sum())}

        # retrieval: a Jaco frame against the non-Jaco frames
        j = np.flatnonzero(is_jaco)
        o = np.flatnonzero(~is_jaco)
        scores = sim[np.ix_(j, o)]
        best = scores.argmax(1)
        hit = rows_of[o][best] == rows_of[j]
        err = np.linalg.norm(xyz[o][best] - xyz[j], axis=1) * 100
        top5 = np.argsort(-scores, axis=1)[:, :5]
        hit5 = (rows_of[o][top5] == rows_of[j][:, None]).any(1)

        per_emb = {}
        for emb in EMB:
            mine = emb_of == emb
            m = same_state & mine[:, None] & ~mine[None, :]
            if m.sum():
                per_emb[emb] = float(sim[m].mean())

        # THE CONTROL. Two renders of the same row share the entire scene -- same kitchen, same
        # object positions, same lighting -- and differ only in the arm. A tower that matches on
        # SCENE rather than on pose scores perfectly on that without reading the robot at all.
        # So: pairs from the SAME episode at DIFFERENT times. The scene is nearly identical and
        # only the arm has moved, which is the comparison that cannot be won by background.
        same_ep = (episodes[rows_of][:, None] == episodes[rows_of][None, :])
        within = same_ep & diff_state
        control = {}
        if within.sum():
            control["same episode, DIFFERENT state"] = {
                "centred": float(sim[within].mean()), "n": int(within.sum())}
        cross_ep = (~same_ep) & diff_state
        control["different episode, different state"] = {
            "centred": float(sim[cross_ep].mean()), "n": int(cross_ep.sum())}
        control["same state (different episode impossible)"] = {
            "centred": float(sim[same_state].mean()), "n": int(same_state.sum())}

        results["towers"][name] = {
            "pairs": pairs, "scene_control": control, "per_embodiment_same_state": per_emb,
            "jaco_retrieval": {"top1": float(hit.mean()), "top5": float(hit5.mean()),
                               "median_state_error_cm": float(np.median(err)),
                               "n": int(len(j))},
        }
        log(f"--- {name}")
        for k, v in pairs.items():
            log(f"    {k:<36} centred {v['centred']:+.3f}  raw {v['raw']:+.3f}  (n={v['n']})")
        r = results["towers"][name]["jaco_retrieval"]
        log(f"    JACO frame -> nearest non-Jaco frame: same state top1 {r['top1']:.3f} "
            f"top5 {r['top5']:.3f}, median EEF error {r['median_state_error_cm']:.1f} cm")
        log(f"    per embodiment: " + "  ".join(f"{k}={v:+.3f}" for k, v in per_emb.items()))
        log("    SCENE CONTROL:")
        for k, v in control.items():
            log(f"      {k:<44} centred {v['centred']:+.3f}  (n={v['n']})")
        del tower
        torch.cuda.empty_cache()

    args.out.write_text(json.dumps(results, indent=2))
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
