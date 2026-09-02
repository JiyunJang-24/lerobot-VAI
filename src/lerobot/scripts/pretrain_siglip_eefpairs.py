#!/usr/bin/env python
"""Contrastive pre-training of the SigLIP tower on the eef_pairs exports.

Separate from pretrain_siglip_visual_robust.py because the data is shaped differently, not because
the objective differs. The visual-robust exports put every embodiment's render of one instant in
its OWN COLUMN of a single row, so a positive group is "several keys of one item". eef_pairs
instead stores one image per row and labels it with observation.embodiment_index, so a positive
group is "several ROWS sharing an episode" -- one canonical EEF pose per episode, which is exactly
what the dataset card promises.

Built for the scaling question: --n-embodiments trains on a subset of the available embodiments,
so a ladder (8 -> 16 -> 32 -> ...) answers "how much embodiment diversity is enough" directly, and
everything not trained on stays available as a held-out set for
tools/eval_heldout_embodiment.py.

    python src/lerobot/scripts/pretrain_siglip_eefpairs.py --n-embodiments 16 --steps 5000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad  # noqa: E402
from lerobot.scripts.lerobot_train_with_visual_robust import _supervised_contrastive_loss  # noqa: E402

VLM_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEFAULT_ROOT = Path("/dataset/jiyun/dataset_git/eef_pairs")
IMAGE_KEY = "observation.images.agentview_right"


def log(msg: str) -> None:
    print(f"[pretrain_eefpairs] {msg}", flush=True)


def _gripper_state(subset: str) -> str:
    """closed / open, from the subset name (matches meta/info.json's gripper_state)."""
    return "open" if "_open" in subset else "closed"


def build_row_table(root: Path, subsets: list[str], share_poses: bool = True) -> "object":
    """(global_row, pose, embodiment) for every frame, read from the parquet.

    Reading the columns rather than the dataset keeps this cheap: the images are only touched for
    rows a batch actually samples.
    """
    import glob

    import pandas as pd

    tables = []
    offset = 0
    for subset in subsets:
        files = sorted(glob.glob(str(root / subset / "data" / "**" / "*.parquet"), recursive=True))
        frames = [
            pd.read_parquet(f, columns=[
                "episode_index", "frame_index", "observation.embodiment_index",
                "observation.state", "observation.camera_view_index",
            ])
            for f in files
        ]
        table = pd.concat(frames, ignore_index=True)
        table = table.rename(columns={"observation.embodiment_index": "embodiment"})
        # Global row index within this subset's flat dataset = episode start + frame index. The
        # episode start is the cumulative length of preceding episodes.
        lengths = table.groupby("episode_index").size().sort_index()
        starts = lengths.cumsum().shift(fill_value=0)
        table["row"] = table.episode_index.map(starts) + table.frame_index + offset
        table["subset"] = subset
        # What counts as "the same pose" decides what the loss pulls together, and the two axes
        # these subsets vary along are NOT the same kind of thing:
        #
        #   colour/texture (the _furniture variants) is nuisance appearance -- the arm is in the
        #   identical configuration, only rendered differently, so those pairs SHOULD be positive.
        #   Verified: episode 0 carries EEF xyz [0.5222, 0.1529, 0.4192] in all four subsets.
        #
        #   gripper open vs closed is a real difference in the robot's STATE, not its appearance.
        #   Pulling those together would train the encoder to discard whether the gripper is open,
        #   which is information a policy needs. So the pose label keeps them apart.
        #
        # Hence: share the pose across colour variants, split it across gripper states.
        gripper = _gripper_state(subset)
        table["gripper"] = gripper
        if share_poses:
            table["pose"] = table.episode_index + (1000 if gripper == "open" else 0)
        else:
            table["pose"] = table.episode_index + 1000 * subsets.index(subset)
        offset += len(table)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def sample_batch(table, embodiments, poses_per_batch, views_per_pose, rng):
    """One contrastive batch: a few poses, several embodiments each.

    Same structure the loss expects -- rows sharing a pose are positives, rows of different poses
    are negatives -- just assembled from rows instead of columns.
    """
    available = table[table.embodiment.isin(embodiments)]
    poses = rng.choice(available.pose.unique(), size=poses_per_batch, replace=False)
    positions, labels = [], []
    for label, pose in enumerate(poses):
        block = available[available.pose == pose]
        take = block.sample(n=min(views_per_pose, len(block)), random_state=int(rng.integers(1 << 30)))
        positions += take.cache_pos.tolist()
        labels += [label] * len(take)
    positions = torch.tensor(positions)
    return positions, torch.tensor(labels)


def preload_images(dataset, rows, cache_path: Path | None = None):
    """Decode every row ONCE into a uint8 tensor held in RAM.

    Decoding inside the training loop is what makes this data expensive: each step wants ~48 frames
    and each frame is a seek into an mp4, so eight concurrent runs drove load average past 500 on a
    96-core box while the GPUs sat at 0%. The images are 180x320x3 uint8 -- the whole 56combo subset
    is a few GB -- so decoding up front and indexing RAM afterwards removes the bottleneck
    entirely. Stored pre-resize to keep the cache small; the resize is cheap.
    """
    if cache_path is not None and cache_path.exists():
        log(f"reusing image cache {cache_path}")
        return torch.load(cache_path)

    log(f"decoding {len(rows)} frames once into RAM ...")
    out = torch.empty(len(rows), 3, 180, 320, dtype=torch.uint8)
    for i, row in enumerate(rows):
        image = dataset[int(row)][IMAGE_KEY]
        image = image[-1] if image.ndim == 4 else image
        out[i] = (image * 255).round().clamp(0, 255).to(torch.uint8)
        if (i + 1) % 5000 == 0:
            log(f"  decoded {i + 1}/{len(rows)}")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, cache_path)
        log(f"wrote image cache {cache_path}")
    return out


def load_images(cache, positions, device):
    """positions index into the preloaded cache, not the dataset."""
    images = cache[positions].to(device=device, dtype=torch.float32) / 255.0
    return resize_with_pad(images, 512, 512, pad_value=0) * 2.0 - 1.0


class RegressionHead(torch.nn.Module):
    """Predicts a target from the token grid, with attention pooling for the spatial ones.

    pool="attn" for the pixel target specifically: where the gripper appears IS a location, and a
    mean over 1024 position-tagged patches is close to location-blind. That difference was measured
    on the earlier EEF head (CLAUDE.md section 3) and it was decisive there.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, pool: str = "attn"):
        super().__init__()
        self.pool = pool
        self.score = torch.nn.Linear(in_dim, 1) if pool == "attn" else None
        mlp_in = in_dim * 2 if pool == "attn" else in_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(mlp_in, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, out_dim)
        )

    def forward(self, tokens):
        pooled = tokens.mean(dim=1)
        if self.pool == "attn":
            weights = torch.softmax(self.score(tokens), dim=1)
            pooled = torch.cat([pooled, (weights * tokens).sum(dim=1)], dim=-1)
        return self.net(pooled)


def eef_state_loss(pred, target):
    """xyz MSE + a sign-invariant quaternion term.

    NOT an MSE on the quaternion: q and -q are the same rotation, so a plain MSE punishes a correct
    prediction that happens to carry the other sign. 1 - |<q_pred, q_target>| removes the double
    cover instead of trying to pick a side -- the same reasoning as the visual-robust EEF head.
    """
    position = F.mse_loss(pred[:, :3], target[:, :3])
    q_pred = F.normalize(pred[:, 3:7], dim=-1)
    q_target = F.normalize(target[:, 3:7], dim=-1)
    rotation = 1.0 - (q_pred * q_target).sum(-1).abs().clamp(max=1.0).mean()
    return position + rotation, position.detach(), rotation.detach()


def encode(tower, images, chunk, grad=True):
    from contextlib import nullcontext

    out = []
    for piece in images.split(chunk):
        with nullcontext() if grad else torch.no_grad():
            out.append(tower(pixel_values=piece, patch_attention_mask=None).last_hidden_state)
    return torch.cat(out)


@torch.no_grad()
def quick_gap(tower, cache, table, embodiments, rng, device, chunk, poses=8, views=6):
    """Positive/negative cosine gap on a fixed-ish sample -- a progress signal, not the evaluation.

    The real scoring is tools/eval_heldout_embodiment.py, which uses embodiments this never saw.
    """
    tower.eval()
    positions, labels = sample_batch(table, embodiments, poses, views, rng)
    feats = encode(tower, load_images(cache, positions, device), chunk, grad=False).mean(dim=1).float()
    feats = feats - feats.mean(dim=0, keepdim=True)
    feats = F.normalize(feats, dim=-1)
    sim = feats @ feats.T
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool)
    tower.train()
    return (sim[same & ~eye].mean() - sim[~same].mean()).item()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--subsets", default="56combo_48_bg12_closed",
                    help="comma-separated dataset dirs under --root")
    ap.add_argument("--separate-poses-per-subset", action="store_true",
                    help="give every subset its own pose labels. Off by default: colour variants of "
                         "one pose are shared (nuisance appearance -> positive) while open and "
                         "closed gripper states are kept apart (real state -> negative).")
    ap.add_argument("--n-embodiments", type=int, default=0,
                    help="train on this many embodiments (0 = all). The rest stay held out.")
    ap.add_argument("--holdout-embodiments", default=None,
                    help="explicit comma-separated indices to EXCLUDE from training")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--poses-per-batch", type=int, default=8)
    ap.add_argument("--views-per-pose", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--objective", default="contrastive",
                    choices=["contrastive", "eef_state", "eef_pixel", "all"],
                    help="contrastive: pull same-pose renders together. "
                         "eef_state: regress xyz+quaternion (constant within a pose). "
                         "eef_pixel: regress where the gripper appears in THIS image -- unlike the "
                         "others it varies within a pose, because the 4 camera views see the same "
                         "pose from different angles. all: sum of the three.")
    ap.add_argument("--eef-state-weight", type=float, default=1.0)
    ap.add_argument("--eef-pixel-weight", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--encoder-chunk", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    table = build_row_table(args.root, subsets, share_poses=not args.separate_poses_per_subset)
    all_emb = np.sort(table.embodiment.unique())

    if args.holdout_embodiments:
        holdout = np.array([int(v) for v in args.holdout_embodiments.split(",")])
        train_emb = np.setdiff1d(all_emb, holdout)
    elif args.n_embodiments and args.n_embodiments < len(all_emb):
        # Deterministic given the seed, so a ladder at 8/16/32 is nested-ish and comparable.
        train_emb = np.sort(rng.choice(all_emb, size=args.n_embodiments, replace=False))
        holdout = np.setdiff1d(all_emb, train_emb)
    else:
        train_emb, holdout = all_emb, np.array([], dtype=int)

    log(f"subsets {subsets}: {len(table)} rows, {table.pose.nunique()} poses, {len(all_emb)} embodiments")
    if "gripper" in table:
        for gripper, block in table.groupby("gripper"):
            log(f"  gripper={gripper}: {len(block)} rows, {block.pose.nunique()} poses")
    log(f"training on {len(train_emb)}: {train_emb.tolist()}")
    log(f"held out    {len(holdout)}: {holdout.tolist()}")

    dataset = MultiLeRobotDataset(
        subsets, root=args.root, delta_timestamps={s: None for s in subsets},
        visual_cue_mode="vanilla", use_wrist_cam=False, use_state=True, cache_in_memory=False,
    )

    # Only the rows any run could sample need decoding. Keyed by subset list so the closed-only and
    # closed+open runs do not share a cache by accident.
    table = table.reset_index(drop=True)
    table["cache_pos"] = np.arange(len(table))
    cache_key = "_".join(subsets)
    cache = preload_images(dataset, table.row.tolist(),
                           Path("/dev/shm") / f"eefpairs_cache_{cache_key}.pt")

    from transformers import AutoModelForImageTextToText

    log(f"loading {VLM_MODEL} vision tower (fp32) ...")
    tower = AutoModelForImageTextToText.from_pretrained(
        VLM_MODEL, dtype=torch.float32
    ).model.vision_model.to(device)
    tower.train()
    reference = {n: p.detach().clone() for n, p in tower.named_parameters()}

    # Targets, normalised so the regression terms are on a comparable scale to the contrastive one.
    # xyz spans ~0.26 m and pixels span the 320x180 image, so raw MSEs would differ by orders of
    # magnitude and whichever is larger would dominate the sum.
    state = np.stack(table["observation.state"].to_numpy()).astype(np.float32)
    xyz, quat, pixel = state[:, :3], state[:, 3:7], state[:, 7:9]
    xyz_mean, xyz_std = xyz.mean(0), xyz.std(0) + 1e-6
    targets_state = torch.tensor(np.concatenate([(xyz - xyz_mean) / xyz_std, quat], axis=1))
    # Pixels to [-1, 1] against the actual image size rather than the observed range, so the target
    # keeps its geometric meaning.
    targets_pixel = torch.tensor(np.stack([pixel[:, 0] / 320.0, pixel[:, 1] / 180.0], axis=1) * 2 - 1)
    log(f"targets: xyz std {np.round(xyz_std, 4).tolist()}, "
        f"pixel range u [{pixel[:, 0].min():.0f},{pixel[:, 0].max():.0f}] "
        f"v [{pixel[:, 1].min():.0f},{pixel[:, 1].max():.0f}]")

    hidden_dim = tower.config.hidden_size
    heads = {}
    want = {"contrastive", "eef_state", "eef_pixel"} if args.objective == "all" else {args.objective}
    if "eef_state" in want:
        # mean pooling: xyz+quat is one value for the whole pose, not a location in the frame.
        heads["eef_state"] = RegressionHead(hidden_dim, 7, pool="mean").to(device)
    if "eef_pixel" in want:
        heads["eef_pixel"] = RegressionHead(hidden_dim, 2, pool="attn").to(device)
    log(f"objective={args.objective} -> terms {sorted(want)}"
        + (f", heads {sorted(heads)}" if heads else ""))

    trainable = list(tower.parameters())
    for head in heads.values():
        trainable += list(head.parameters())
    optimiser = torch.optim.AdamW(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: min(1.0, (s + 1) / max(1, args.warmup))
    )

    history = []
    for step in range(1, args.steps + 1):
        positions, labels = sample_batch(table, train_emb, args.poses_per_batch, args.views_per_pose, rng)
        images = load_images(cache, positions, device)
        tokens = encode(tower, images, args.encoder_chunk)

        loss = torch.zeros((), device=device)
        parts = {}
        if "contrastive" in want:
            term = _supervised_contrastive_loss(tokens.mean(dim=1), labels.to(device), args.temperature)
            loss = loss + term
            parts["con"] = float(term)
        if "eef_state" in want:
            term, pos_term, rot_term = eef_state_loss(
                heads["eef_state"](tokens).float(), targets_state[positions].to(device)
            )
            loss = loss + args.eef_state_weight * term
            parts["xyz"] = float(pos_term)
            parts["rot"] = float(rot_term)
        if "eef_pixel" in want:
            pred = heads["eef_pixel"](tokens).float()
            target = targets_pixel[positions].to(device)
            term = F.mse_loss(pred, target)
            loss = loss + args.eef_pixel_weight * term
            # Report in pixels, which is readable, rather than in the normalised units it trains on.
            with torch.no_grad():
                err = ((pred - target) * torch.tensor([160.0, 90.0], device=device)).norm(dim=-1).mean()
            parts["pix_px"] = float(err)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 10.0)
        optimiser.step()
        scheduler.step()

        if step % args.eval_every == 0 or step == args.steps:
            drift_num = sum((p.detach() - reference[n]).pow(2).sum() for n, p in tower.named_parameters())
            drift_den = sum(t.pow(2).sum() for t in reference.values())
            drift = float((drift_num / drift_den) ** 0.5)
            gap = quick_gap(tower, cache, table, train_emb, rng, device, args.encoder_chunk)
            history.append({"step": step, "loss": float(loss), "train_gap": gap, "drift": drift, **parts})
            detail = "".join(f"  {k} {v:6.3f}" for k, v in parts.items())
            log(f"step {step:5d}  loss {float(loss):6.3f}{detail}  train gap {gap:+.4f}  "
                f"drift {drift:.5f}  |g| {float(grad_norm):.2f}")

    from safetensors.torch import save_file

    for name, head in heads.items():
        torch.save(head.state_dict(), out_dir / f"head_{name}.pt")

    tower_path = out_dir / "vision_tower.safetensors"
    save_file({n: p.detach().cpu().contiguous() for n, p in tower.state_dict().items()}, tower_path)
    (out_dir / "pretrain_info.json").write_text(json.dumps({
        "args": vars(args), "train_embodiments": train_emb.tolist(),
        "holdout_embodiments": holdout.tolist(), "history": history,
    }, indent=2, default=str))
    log(f"wrote {tower_path}")
    log(f"held-out embodiments for evaluation: {holdout.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
