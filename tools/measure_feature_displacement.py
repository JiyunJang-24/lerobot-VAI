#!/usr/bin/env python
"""Does a wrong pixel mean the FEATURES moved, or only that the head's readout broke?

The pixel head is a small MLP on top of the tower. Its failure under a shifted camera says the
head+tower together broke; it does not say which. That distinction decides what the finding means
for the policy, because the policy reads the tower's features and never touches this head.

So: move one image, measure how far its feature moves, and express that in the only unit that means
anything here -- how far the feature moves for a REAL change of pose. A representation that is
doing its job should move a lot for a new pose, barely at all for a new background or a new robot,
and ideally barely at all for a camera shift.

Cosine distance is reported centred as well as raw, because SigLIP features are anisotropic enough
that raw cosine sits near 0.99 for every pair and hides the structure (CLAUDE.md section 4).

    python tools/measure_feature_displacement.py --checkpoint outputs/siglip_pretrain/all4_n42_all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad  # noqa: E402
from tools.diagnose_eef_pixel_sensitivity import subset_table, translate, zoom_out  # noqa: E402
from tools.predict_eef_pixel import load_tower_and_head  # noqa: E402

EEF_ROOT = Path("/dataset/jiyun/dataset_git/eef_pairs")
PLAIN, FURN = "56combo_48_bg12_closed", "56combo_48_bg12_closed_furniture"
BARX = REPO_ROOT / "dataset_git/barx_panda_ur5e_iiwa"
POLICY_REPO = "IIWAOmron/pretrain/PnPSinkToCounter/lerobot"
W, H = 320, 180


def log(msg: str) -> None:
    print(f"[displacement] {msg}", flush=True)


@torch.no_grad()
def features(tower, images_uint8, device):
    x = images_uint8.to(device=device, dtype=torch.float32) / 255.0
    x = resize_with_pad(x, 512, 512, pad_value=0) * 2.0 - 1.0
    out = []
    for piece in x.split(16):
        out.append(tower(pixel_values=piece, patch_attention_mask=None).last_hidden_state.mean(1))
    return torch.cat(out).float().cpu()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO_ROOT / "outputs/siglip_pretrain/all4_n42_all")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/feature_displacement.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = json.loads((args.checkpoint / "pretrain_info.json").read_text())
    holdout = info["holdout_embodiments"]
    tower, _ = load_tower_and_head(args.checkpoint, device)
    log(f"{args.checkpoint.name}, held-out embodiments only")

    plain = subset_table(PLAIN)
    plain = plain[plain.embodiment.isin(holdout)].reset_index(drop=True)
    furn = subset_table(FURN)
    furn = furn[furn.embodiment.isin(holdout)].reset_index(drop=True)
    key = ["episode_index", "embodiment", "background", "view"]
    furn_lookup = furn.drop_duplicates(key).set_index(key)

    rng = np.random.default_rng(args.seed)
    anchors, partners = {c: [] for c in ("bg", "furn", "emb", "pose")}, None
    picks = []
    for _, row in plain.sample(frac=1.0, random_state=args.seed).iterrows():
        ep, em, bg, vw = (int(row[k]) for k in key)
        # every partner must differ in exactly ONE thing, otherwise the comparison is not clean
        cand_bg = plain[(plain.episode_index == ep) & (plain.embodiment == em)
                        & (plain.view == vw) & (plain.background != bg)]
        cand_emb = plain[(plain.episode_index == ep) & (plain.background == bg)
                         & (plain.view == vw) & (plain.embodiment != em)]
        cand_pose = plain[(plain.episode_index != ep) & (plain.embodiment == em)
                          & (plain.background == bg) & (plain.view == vw)]
        if not len(cand_bg) or not len(cand_emb) or not len(cand_pose):
            continue
        if (ep, em, bg, vw) not in furn_lookup.index:
            continue
        picks.append({
            "anchor": int(row["row"]),
            "bg": int(cand_bg.sample(1, random_state=int(rng.integers(1 << 30))).iloc[0]["row"]),
            "emb": int(cand_emb.sample(1, random_state=int(rng.integers(1 << 30))).iloc[0]["row"]),
            "pose": int(cand_pose.sample(1, random_state=int(rng.integers(1 << 30))).iloc[0]["row"]),
            "furn": int(furn_lookup.loc[(ep, em, bg, vw)]["row"]),
        })
        if len(picks) >= args.n:
            break
    log(f"built {len(picks)} matched sets (each partner differs in exactly one factor)")

    ds_plain = LeRobotDataset(f"eef_pairs/{PLAIN}", root=EEF_ROOT / PLAIN)
    ds_furn = LeRobotDataset(f"eef_pairs/{FURN}", root=EEF_ROOT / FURN)

    def grab(dataset, rows):
        out = torch.empty(len(rows), 3, H, W, dtype=torch.uint8)
        for i, r in enumerate(rows):
            frame = dataset[int(r)]["observation.images.agentview_right"]
            frame = frame[-1] if frame.ndim == 4 else frame
            out[i] = (frame * 255).round().clamp(0, 255).to(torch.uint8)
        return out

    anchor_img = grab(ds_plain, [p["anchor"] for p in picks])
    banks = {
        "different background\n(same pose, same robot)": grab(ds_plain, [p["bg"] for p in picks]),
        "furniture recolour\n(same pose, same robot)": grab(ds_furn, [p["furn"] for p in picks]),
        "different robot\n(same pose)": grab(ds_plain, [p["emb"] for p in picks]),
    }
    dummy = np.zeros((len(picks), 2))
    banks["shift 40 px\n(same image, moved)"] = translate(anchor_img, dummy, 40, 0)[0]
    banks["zoom out 0.75\n(same image, smaller)"] = zoom_out(anchor_img, dummy, 0.75)[0]
    banks["DIFFERENT POSE\n(the reference scale)"] = grab(ds_plain, [p["pose"] for p in picks])

    f_anchor = features(tower, anchor_img, device)
    centre = f_anchor.mean(0, keepdim=True)

    rows = {}
    for name, bank in banks.items():
        f = features(tower, bank, device)
        n = min(len(f_anchor), len(f))
        raw = 1 - F.cosine_similarity(f_anchor[:n], f[:n], dim=-1)
        cen = 1 - F.cosine_similarity(f_anchor[:n] - centre, f[:n] - centre, dim=-1)
        rows[name] = (float(raw.mean()), float(cen.mean()))

    # A pose change may already be FULL decorrelation rather than merely a large step, and that
    # changes how every percentage below is read. Shuffling the anchors against themselves gives
    # the unrelated-pair floor to compare it against.
    perm = rng.permutation(len(f_anchor))
    unrelated = float((1 - F.cosine_similarity(f_anchor - centre,
                                               f_anchor[perm] - centre, dim=-1)).mean())
    ref_raw, ref_cen = rows["DIFFERENT POSE\n(the reference scale)"]
    log(f"(unrelated-pair floor, anchors shuffled against themselves: {unrelated:.4f} centred — "
        f"a pose change is {100 * ref_cen / unrelated:.0f}% of it)")
    log(f"{'condition':<40}{'raw':>9}{'centred':>10}{'% of a pose change':>22}")
    for name, (raw, cen) in rows.items():
        log(f"{name.replace(chr(10), ' '):<40}{raw:>9.4f}{cen:>10.4f}{100 * cen / ref_cen:>21.0f}%")

    # Where do the policy corpus's own frames sit relative to all of this?
    ds_pol = LeRobotDataset(POLICY_REPO, root=BARX / POLICY_REPO)
    pol_rows = rng.choice(len(ds_pol), size=len(picks), replace=False)
    pol_img = torch.empty(len(pol_rows), 3, H, W, dtype=torch.uint8)
    for i, r in enumerate(sorted(pol_rows)):
        frame = ds_pol[int(r)]["observation.images.robot0_agentview_right"]
        frame = frame[-1] if frame.ndim == 4 else frame
        pol_img[i] = (frame * 255).round().clamp(0, 255).to(torch.uint8)
    f_pol = features(tower, pol_img, device)
    pol_cen = float((1 - F.cosine_similarity(f_anchor - centre, f_pol - centre, dim=-1)).mean())
    log(f"{'policy corpus frame (whole other scene)':<40}{'':>9}{pol_cen:>10.4f}"
        f"{100 * pol_cen / ref_cen:>21.0f}%")
    rows["POLICY CORPUS\n(a whole other scene)"] = (float("nan"), pol_cen)

    names = list(rows)
    values = [100 * rows[n][1] / ref_cen for n in names]
    colours = ["#3d5573"] * 3 + ["#b3153b"] * 2 + ["#5a6572", "#7a4fb3"]
    fig, ax = plt.subplots(figsize=(13.0, 5.4))
    ax.bar(range(len(names)), values, color=colours)
    for i, v in enumerate(values):
        ax.annotate(f"{v:.0f}%", (i, v), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=10)
    ceiling = 100 * unrelated / ref_cen
    ax.axhline(ceiling, ls="--", c="gray", lw=1.2)
    ax.annotate("two unrelated images — nothing left in common",
                (-0.45, ceiling), textcoords="offset points", xytext=(0, 6),
                ha="left", fontsize=9, color="dimgray")
    ax.set_ylim(0, ceiling * 1.16)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8.5, rotation=18, ha="right")
    ax.set_ylabel("feature displacement, as % of a real pose change")
    ax.grid(alpha=0.3, axis="y")
    ax.set_title(
        f"How far does the FEATURE move? ({args.checkpoint.name}, held-out embodiments)\n"
        "centred cosine distance, normalised by a genuine pose change — which is itself 98% of the "
        "way to two unrelated images,\nso read these as 'fraction of the way to nothing in common'",
        fontsize=11.5)
    fig.tight_layout()
    fig.savefig(args.out, dpi=115, bbox_inches="tight")
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
