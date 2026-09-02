#!/usr/bin/env python
"""Run a pre-trained tower's EEF-pixel head on the POLICY corpus and draw what it predicts.

The head was trained only on eef_pairs renders -- 56 embodiments frozen at 48 canonical poses on a
static background. The policy corpus is a different thing entirely: real trajectories, three
embodiments, a kitchen that changes as objects get moved. Nothing was fine-tuned for it, so this is
a zero-shot transfer check, and the point is to see whether "where is the gripper" survived the
change of domain.

There is no ground-truth pixel in the policy corpus (section 9, and no camera intrinsics either),
so that half can only be eyeballed. The eef_pairs held-out panel above it is the control: same
head, same code path, on embodiments the tower never trained on but in the domain it was trained
in. If the control is tight and the policy panel is not, the gap is domain transfer rather than a
broken checkpoint.

    python tools/predict_eef_pixel.py --checkpoint outputs/siglip_pretrain/all4_n42_all
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad  # noqa: E402
from lerobot.scripts.pretrain_siglip_eefpairs import RegressionHead, build_row_table  # noqa: E402

EEF_ROOT = Path("/dataset/jiyun/dataset_git/eef_pairs")
BARX = REPO_ROOT / "dataset_git/barx_panda_ur5e_iiwa"
POLICY_REPOS = [
    "IIWAOmron/pretrain/PnPSinkToCounter/lerobot",
    "PandaOmron/pretrain/PnPSinkToCounter/lerobot",
    "UR5eOmron/pretrain/PnPSinkToCounter/lerobot",
]
POLICY_CAMERA = "observation.images.robot0_agentview_right"
W, H = 320, 180


def log(msg: str) -> None:
    print(f"[eef_pixel] {msg}", flush=True)


def load_tower_and_head(checkpoint: Path, device):
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText

    vlm = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", dtype=torch.float32
    )
    tower = vlm.model.vision_model
    tower.load_state_dict(load_file(checkpoint / "vision_tower.safetensors"), strict=True)
    head = RegressionHead(int(tower.config.hidden_size), 2, pool="attn")
    head.load_state_dict(torch.load(checkpoint / "head_eef_pixel.pt"))
    return tower.to(device).eval(), head.to(device).eval()


@torch.no_grad()
def predict(tower, head, images_uint8, device):
    """images_uint8: (N, 3, 180, 320) -> (N, 2) pixel coordinates in the ORIGINAL 320x180 frame.

    The pre-processing has to match load_images() exactly: /255, resize_with_pad to 512, then to
    [-1, 1]. The head regresses u/320 and v/180 mapped to [-1, 1], so the padding the resize
    introduces never enters the coordinate frame -- denormalising is against 320x180, not 512x512.
    """
    x = images_uint8.to(device=device, dtype=torch.float32) / 255.0
    x = resize_with_pad(x, 512, 512, pad_value=0) * 2.0 - 1.0
    out = []
    for piece in x.split(16):
        tokens = tower(pixel_values=piece, patch_attention_mask=None).last_hidden_state
        out.append(head(tokens).float())
    pred = torch.cat(out).cpu().numpy()
    return np.stack([(pred[:, 0] + 1) / 2 * W, (pred[:, 1] + 1) / 2 * H], axis=1)


def sample_eefpairs(n_per_embodiment: int, holdout: list[int], seed: int):
    """Frames from embodiments the tower never trained on, with their ground-truth pixel."""
    subset = "56combo_48_bg12_closed"
    table = build_row_table(EEF_ROOT, [subset], share_poses=True)
    table = table[table.embodiment.isin(holdout)]
    rng = np.random.default_rng(seed)
    picks = table.sample(n=n_per_embodiment, random_state=int(rng.integers(1 << 30)))

    dataset = LeRobotDataset(f"eef_pairs/{subset}", root=EEF_ROOT / subset)
    images, truth, tags = [], [], []
    for _, row in picks.iterrows():
        frame = dataset[int(row["row"])]["observation.images.agentview_right"]
        frame = frame[-1] if frame.ndim == 4 else frame
        images.append((frame * 255).round().clamp(0, 255).to(torch.uint8))
        state = np.asarray(row["observation.state"], dtype=np.float32)
        truth.append(state[7:9])  # verified identical to observation.eef_pixel
        tags.append(f"held-out emb {int(row['embodiment'])}")
    return torch.stack(images), np.stack(truth), tags


def sample_policy(n_per_robot: int, seed: int):
    rng = np.random.default_rng(seed)
    images, tags = [], []
    for repo in POLICY_REPOS:
        dataset = LeRobotDataset(repo, root=BARX / repo)
        rows = rng.choice(len(dataset), size=n_per_robot, replace=False)
        for row in sorted(rows):
            frame = dataset[int(row)][POLICY_CAMERA]
            frame = frame[-1] if frame.ndim == 4 else frame
            images.append((frame * 255).round().clamp(0, 255).to(torch.uint8))
            tags.append(repo.split("/")[0].replace("Omron", ""))
    return torch.stack(images), tags


def draw(ax, image_uint8, pred, truth=None, title=""):
    ax.imshow(image_uint8.permute(1, 2, 0).numpy())
    if truth is not None:
        ax.plot(*truth, "o", ms=13, mfc="none", mec="#00ff66", mew=2.2, label="ground truth")
        ax.plot([truth[0], pred[0]], [truth[1], pred[1]], "-", c="w", lw=1.0, alpha=0.8)
    ax.plot(*pred, "x", ms=11, c="#ff2d55", mew=2.6, label="predicted")
    ax.set_title(title, fontsize=8.5)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_xticks([]); ax.set_yticks([])


def fit_camera(xyz, uv):
    """Least-squares 3x4 pinhole projection from EEF xyz to predicted pixel, and its residual.

    This is the check the picture cannot give. Within ONE episode the camera is fixed and the base
    does not move, so base-relative xyz maps to the image through a single projection matrix. If the
    head is actually finding the gripper, one 11-DOF camera explains every frame to a few pixels. If
    it is emitting a learned prior, no camera can, because the output barely depends on the input.

    Affine would be the wrong model here -- the mapping is projective -- so this solves the standard
    DLT rather than a linear regression, and reports residuals in pixels.
    """
    n = len(xyz)
    ones = np.ones((n, 1))
    X = np.hstack([xyz, ones])
    A = np.zeros((2 * n, 12))
    A[0::2, 0:4] = X
    A[0::2, 8:12] = -uv[:, 0:1] * X
    A[1::2, 4:8] = X
    A[1::2, 8:12] = -uv[:, 1:2] * X
    _, _, vt = np.linalg.svd(A)
    P = vt[-1].reshape(3, 4)
    proj = X @ P.T
    denom = proj[:, 2:3]
    denom[np.abs(denom) < 1e-9] = 1e-9
    return np.linalg.norm(proj[:, :2] / denom - uv, axis=1)


def track_episodes(tower, head, device, n_episodes, n_frames, seed, out_path):
    """Per-episode: does one camera matrix explain the predictions?"""
    rng = np.random.default_rng(seed)
    rows, panels, all_uv = [], [], []
    for repo in POLICY_REPOS:
        dataset = LeRobotDataset(repo, root=BARX / repo)
        episodes = dataset.meta.episodes
        picks = rng.choice(len(episodes), size=n_episodes, replace=False)
        for ep in picks:
            lo = int(episodes[int(ep)]["dataset_from_index"])
            hi = int(episodes[int(ep)]["dataset_to_index"])
            idx = np.linspace(lo, hi - 1, n_frames).astype(int)
            images, xyz = [], []
            for i in idx:
                sample = dataset[int(i)]
                frame = sample[POLICY_CAMERA]
                frame = frame[-1] if frame.ndim == 4 else frame
                images.append((frame * 255).round().clamp(0, 255).to(torch.uint8))
                xyz.append(np.asarray(sample["observation.state"], dtype=np.float64)[7:10])
            images = torch.stack(images)
            xyz = np.stack(xyz)
            uv = predict(tower, head, images, device).astype(np.float64)
            resid = fit_camera(xyz, uv)
            # Same fit against a permuted pairing: this is what "no relationship" scores, and it is
            # the number that makes the real residual mean something.
            chance = fit_camera(xyz, uv[rng.permutation(len(uv))])
            rows.append({"robot": repo.split("/")[0], "episode": int(ep),
                         "resid": float(np.median(resid)), "chance": float(np.median(chance)),
                         "spread": float(uv.std(axis=0).mean())})
            if not any(pan[2] == repo.split("/")[0] for pan in panels):
                panels.append((images[0], uv, repo.split("/")[0], np.median(resid)))
            all_uv.append(uv)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.0))
    for ax, (frame, uv, robot, res) in zip(np.atleast_1d(axes), panels, strict=True):
        ax.imshow(frame.permute(1, 2, 0).numpy())
        ax.scatter(uv[:, 0], uv[:, 1], c=np.arange(len(uv)), cmap="plasma", s=26,
                   edgecolors="w", linewidths=0.5)
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{robot} — one episode, {len(uv)} frames\ncamera-fit residual {res:.1f} px",
                     fontsize=9)
    fig.suptitle("Predicted EEF pixel over a single episode (colour = time, frame 0 shown)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=115, bbox_inches="tight")
    log(f"wrote {out_path}")

    # If the head has fallen back to its training prior, the predictions collapse toward the mean
    # of the pixel distribution it was trained on, and their spread collapses with them.
    stacked = np.concatenate(all_uv)
    log(f"policy predictions: mean ({stacked[:, 0].mean():.1f}, {stacked[:, 1].mean():.1f}) "
        f"std ({stacked[:, 0].std():.1f}, {stacked[:, 1].std():.1f})")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO_ROOT / "outputs/siglip_pretrain/all4_n42_all")
    ap.add_argument("--n-cols", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/eef_pixel_on_policy.png")
    ap.add_argument("--track", action="store_true",
                    help="also fit a camera per episode -- the quantitative version of the picture")
    ap.add_argument("--track-episodes", type=int, default=3)
    ap.add_argument("--track-frames", type=int, default=40)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import json

    info = json.loads((args.checkpoint / "pretrain_info.json").read_text())
    holdout = info["holdout_embodiments"]
    log(f"checkpoint {args.checkpoint.name}, final training pixel error "
        f"{info['history'][-1].get('pix_px', float('nan')):.2f} px")

    tower, head = load_tower_and_head(args.checkpoint, device)

    log("control: eef_pairs, embodiments the tower never trained on ...")
    ctrl_img, ctrl_truth, ctrl_tags = sample_eefpairs(args.n_cols, holdout, args.seed)
    ctrl_pred = predict(tower, head, ctrl_img, device)
    err = np.linalg.norm(ctrl_pred - ctrl_truth, axis=1)
    log(f"  held-out pixel error: mean {err.mean():.2f} px, median {np.median(err):.2f}, "
        f"max {err.max():.2f}  (image is {W}x{H})")

    log("transfer: policy corpus, PnPSinkToCounter ...")
    pol_img, pol_tags = sample_policy(args.n_cols, args.seed)
    pol_pred = predict(tower, head, pol_img, device)

    n_rows = 1 + len(POLICY_REPOS)
    fig, axes = plt.subplots(n_rows, args.n_cols, figsize=(2.5 * args.n_cols, 1.75 * n_rows))
    for i in range(args.n_cols):
        draw(axes[0, i], ctrl_img[i], ctrl_pred[i], ctrl_truth[i],
             f"{ctrl_tags[i]} — {err[i]:.1f} px")
    for r in range(len(POLICY_REPOS)):
        for c in range(args.n_cols):
            k = r * args.n_cols + c
            draw(axes[r + 1, c], pol_img[k], pol_pred[k], None, pol_tags[k])
    axes[0, 0].set_ylabel("eef_pairs\n(held-out emb)", fontsize=9)
    for r, repo in enumerate(POLICY_REPOS):
        axes[r + 1, 0].set_ylabel(repo.split("/")[0], fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10, frameon=False)
    fig.suptitle(
        f"EEF-pixel head from {args.checkpoint.name}, applied with NO fine-tuning\n"
        f"top row = the control (eef_pairs, unseen embodiments, has ground truth: "
        f"mean {err.mean():.1f} px)   •   rows below = the policy corpus, no ground truth exists",
        fontsize=12)
    fig.tight_layout(rect=[0, 0.035, 1, 0.90])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=115, bbox_inches="tight")
    log(f"wrote {args.out}")

    if args.track:
        log("per-episode camera fit (the camera is fixed within an episode) ...")
        rows = track_episodes(tower, head, device, args.track_episodes, args.track_frames,
                              args.seed, args.out.with_name(args.out.stem + "_track.png"))
        log(f"{'robot':<12}{'ep':>6}{'resid px':>10}{'shuffled':>10}{'pred spread':>13}")
        for r in rows:
            log(f"{r['robot']:<12}{r['episode']:>6}{r['resid']:>10.2f}{r['chance']:>10.2f}"
                f"{r['spread']:>13.2f}")
        med = np.median([r["resid"] for r in rows])
        chance = np.median([r["chance"] for r in rows])
        log(f"median over episodes: {med:.2f} px vs {chance:.2f} px shuffled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
