#!/usr/bin/env python
"""Language-conditioned Diffusion Policy on PnPSinkToCounter, with the three ways of using the
synthetic embodiment data.

This is the intermediate experiment: a policy WITHOUT a VLM, so if the embodiment representation
helps here but not in the VLA, the problem is VLM integration rather than the representation, and
if it helps in neither the representation itself is what needs revisiting.

    --mode online     start from the pre-trained tower and keep applying the embodiment contrastive
                      objective on 56combo jointly with policy learning
    --mode frozen     load the pre-trained tower and freeze it
    --mode finetune   load the pre-trained tower and let policy learning adapt it
    --mode scratch    stock SigLIP, trainable (the control: no embodiment data at all)

All four share everything else -- same corpus, same batch, same schedule -- so the only variable is
how the embodiment data enters.

    python src/lerobot/scripts/train_dp_embodiment.py --mode frozen \\
        --tower outputs/siglip_pretrain/all4_n42_all/vision_tower.safetensors
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset  # noqa: E402
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig  # noqa: E402
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # noqa: E402
from lerobot.scripts.lerobot_train_with_visual_robust import _supervised_contrastive_loss  # noqa: E402

BARX = REPO_ROOT / "dataset_git/barx_panda_ur5e_iiwa"
EEF_ROOT = Path("/dataset/jiyun/dataset_git/eef_pairs")
CAMERA = "observation.images.robot0_agentview_right"


def log(msg: str) -> None:
    print(f"[dp_embodiment] {msg}", flush=True)


def build_policy_dataset(episodes_per_robot: int, chunk: int):
    """The three SinkToCounter trees -- one task, three embodiments."""
    repos = [
        "IIWAOmron/pretrain/PnPSinkToCounter/lerobot",
        "PandaOmron/pretrain/PnPSinkToCounter/lerobot",
        "UR5eOmron/pretrain/PnPSinkToCounter/lerobot",
    ]
    # DiffusionPolicy asserts the observations carry a time axis of exactly n_obs_steps, so the
    # observation keys need a delta_timestamps entry too -- [0.0] for the single current frame.
    delta = {
        r: {
            "action": [i / 20 for i in range(chunk)],
            CAMERA: [0.0],
            "observation.state": [0.0],
        }
        for r in repos
    }
    dataset = MultiLeRobotDataset(
        repos, root=BARX, delta_timestamps=delta, visual_cue_mode="vanilla",
        use_wrist_cam=False, use_state=True, cache_in_memory=False,
    )
    return dataset


def make_policy(args, dataset, device):
    sample = dataset[0]
    state_dim = sample["observation.state"].shape[-1]
    action_dim = sample["action"].shape[-1]
    image_shape = tuple(sample[CAMERA].shape)

    config = DiffusionConfig(
        n_obs_steps=1,
        horizon=args.chunk,
        n_action_steps=args.chunk,
        crop_shape=None,
        language_conditioned=True,
        use_siglip_encoder=True,
        siglip_encoder_path=args.tower if args.mode in ("frozen", "finetune", "online") else "",
        freeze_vision_encoder=(args.mode == "frozen"),
    )
    config.input_features = {
        CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=image_shape),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MEAN_STD,
    }
    config.device = str(device)

    stats = dataset._datasets[0].meta.stats
    policy = DiffusionPolicy(config, dataset_stats=stats).to(device)
    return policy, config


def build_embodiment_batcher(args, device):
    """Rows of 56combo grouped by canonical pose -- the online auxiliary term's data.

    Reuses the cache the pre-training built, so this adds a forward pass but no decoding.
    """
    from lerobot.scripts.pretrain_siglip_eefpairs import build_row_table, load_images, sample_batch

    subsets = args.eef_subsets.split(",")
    table = build_row_table(EEF_ROOT, subsets, share_poses=True).reset_index(drop=True)
    table["cache_pos"] = np.arange(len(table))
    cache_path = Path("/dev/shm") / f"eefpairs_cache_{'_'.join(subsets)}.pt"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} not found -- run pretrain_siglip_eefpairs.py once to build the image cache."
        )
    log(f"loading embodiment image cache {cache_path} ...")
    cache = torch.load(cache_path)
    embodiments = np.sort(table.embodiment.unique())
    log(f"online auxiliary data: {len(table)} rows, {table.pose.nunique()} poses, "
        f"{len(embodiments)} embodiments")

    rng = np.random.default_rng(args.seed)

    def next_batch():
        positions, labels = sample_batch(table, embodiments, args.aux_poses, args.aux_views, rng)
        # load_images returns [-1, 1] (what the tower wants), but this goes through
        # SiglipRgbEncoder, which re-scales [0, 1] -> [-1, 1] itself because that is what the DP
        # dataloader hands it. Passing [-1, 1] here fed the tower [-3, 1] and the contrastive loss
        # sat at exactly ln(N-1), i.e. chance. Hand over [0, 1] so both callers agree.
        images = (load_images(cache, positions, device) + 1.0) / 2.0
        return images, labels.to(device)

    return next_batch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["online", "frozen", "finetune", "scratch"])
    ap.add_argument("--tower", default=str(REPO_ROOT / "outputs/siglip_pretrain/all4_n42_all/vision_tower.safetensors"))
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--aux-poses", type=int, default=8)
    ap.add_argument("--aux-views", type=int, default=6)
    ap.add_argument("--aux-temperature", type=float, default=0.1)
    ap.add_argument("--eef-subsets",
                    default="56combo_48_bg12_closed,56combo_48_bg12_open,"
                            "56combo_48_bg12_closed_furniture,56combo_48_bg12_open_furniture")
    ap.add_argument("--log-freq", type=int, default=250)
    ap.add_argument("--save-freq", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_policy_dataset(0, args.chunk)
    log(f"policy corpus: {len(dataset)} frames, {dataset.num_episodes} episodes")

    policy, config = make_policy(args, dataset, device)
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    log(f"mode={args.mode}  trainable {trainable / 1e6:.1f}M / {total / 1e6:.1f}M  "
        f"tower={'pretrained' if args.mode != 'scratch' else 'stock'}  "
        f"frozen={config.freeze_vision_encoder}")

    aux_batch = build_embodiment_batcher(args, device) if args.mode == "online" else None

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0,
    )
    optimiser = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-6
    )

    history = []
    step = 0
    t0 = time.time()
    policy.train()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            tasks = batch.get("task")
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            if tasks is not None:
                batch["task"] = tasks
            # This dataset path returns images WITHOUT the time axis that DiffusionPolicy expects
            # (state arrives as (B, 1, D) but the camera as (B, C, H, W)), so add the n_obs_steps=1
            # axis here rather than fighting the loader.
            if batch[CAMERA].ndim == 4:
                batch[CAMERA] = batch[CAMERA].unsqueeze(1)

            loss, parts = policy.forward(batch)
            action_loss = float(loss)

            aux_value = float("nan")
            if aux_batch is not None:
                # The SAME tower the policy uses, so this shapes the representation the policy
                # reads -- not a separate encoder that happens to share a name.
                images, labels = aux_batch()
                encoder = policy.diffusion.rgb_encoder
                encoder = encoder[0] if isinstance(encoder, torch.nn.ModuleList) else encoder
                feats = encoder(images)
                aux = _supervised_contrastive_loss(feats, labels, args.aux_temperature)
                loss = loss + args.aux_weight * aux
                aux_value = float(aux)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 10.0
            )
            optimiser.step()
            step += 1

            if step % args.log_freq == 0 or step == args.steps:
                rate = step / (time.time() - t0)
                entry = {"step": step, "action_loss": action_loss, "aux_loss": aux_value,
                         "total_loss": float(loss)}
                history.append(entry)
                extra = f"  aux {aux_value:6.3f}" if aux_batch is not None else ""
                log(f"step {step:6d}/{args.steps}  action {action_loss:7.4f}{extra}  "
                    f"|g| {float(grad_norm):6.2f}  {rate:.2f} it/s  "
                    f"eta {(args.steps - step) / rate / 3600:.1f}h")

            if step % args.save_freq == 0 or step == args.steps:
                policy.save_pretrained(out_dir / f"checkpoint_{step:06d}")
                (out_dir / "history.json").write_text(
                    json.dumps({"args": vars(args), "history": history}, indent=2, default=str))

    log(f"done: {step} steps, final action loss {history[-1]['action_loss']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
