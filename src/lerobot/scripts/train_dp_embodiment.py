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
from contextlib import nullcontext
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
FPS = 20
EEF_ROOT = Path("/dataset/jiyun/dataset_git/eef_pairs")
CAMERA = "observation.images.robot0_agentview_right"


def log(msg: str) -> None:
    print(f"[dp_embodiment] {msg}", flush=True)


ALL_ROBOTS = [
    "IIWAOmron/pretrain/PnPSinkToCounter/lerobot",
    "PandaOmron/pretrain/PnPSinkToCounter/lerobot",
    "UR5eOmron/pretrain/PnPSinkToCounter/lerobot",
]


def build_policy_dataset(episodes_per_robot: int, chunk: int, heldout_robots=(), horizon=0):
    """The SinkToCounter trees -- one task, several embodiments.

    heldout_robots are excluded here and scored separately, which is what makes "a robot with no
    task demonstrations" a real condition rather than a label.

    horizon > 0 additionally asks the loader for the frame `horizon` steps ahead, so the motion
    auxiliary gets (image_t, image_t+h) out of the same sample instead of a second dataset.
    """
    repos = [r for r in ALL_ROBOTS if r.split("/")[0] not in heldout_robots]
    # DiffusionPolicy asserts the observations carry a time axis of exactly n_obs_steps, so the
    # observation keys need a delta_timestamps entry too -- [0.0] for the single current frame.
    offsets = [0.0] if horizon <= 0 else [0.0, horizon / FPS]
    delta = {
        r: {
            "action": [i / FPS for i in range(chunk)],
            CAMERA: offsets,
            "observation.state": offsets,
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

    dual = args.mode.startswith("dual")
    config = DiffusionConfig(
        n_obs_steps=1,
        horizon=args.chunk,
        n_action_steps=args.chunk,
        crop_shape=None,
        language_conditioned=True,
        use_siglip_encoder=True,
        # main tower = the contrastive one in every dual setting; aux = stock (empty path)
        siglip_encoder_path=(args.tower if args.mode in ("frozen", "finetune", "online") or dual
                             else ""),
        freeze_vision_encoder=(args.mode in ("frozen", "dual_freeze_c", "dual_freeze_both")),
        aux_siglip_encoder_path=("stock" if dual else ""),
        freeze_aux_vision_encoder=(args.mode in ("dual_freeze_s", "dual_freeze_both")),
        aux_branch_dropout=args.aux_dropout,
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


@torch.no_grad()
def evaluate_per_robot(policy, args, device, batches: int = 24):
    """Action loss on each robot separately, including ones excluded from training.

    This is the table experiment 3 exists to fill in: a robot whose task demonstrations the policy
    never saw is only a meaningful condition if it is scored on its own.
    """
    policy.eval()
    heldout = tuple(x for x in args.heldout_robots.split(",") if x)
    rows = {}
    for repo in ALL_ROBOTS:
        name = repo.split("/")[0]
        dataset = build_policy_dataset(0, args.chunk, tuple(r for r in ALL_ROBOTS if r != repo), 0)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=4, drop_last=True)
        losses = []
        for i, batch in enumerate(loader):
            if i >= batches:
                break
            tasks = batch.get("task")
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if tasks is not None:
                batch["task"] = tasks
            if batch[CAMERA].ndim == 4:
                batch[CAMERA] = batch[CAMERA].unsqueeze(1)
            with torch.autocast("cuda", dtype=torch.bfloat16) if args.amp else nullcontext():
                loss, _ = policy.forward(batch)
            losses.append(float(loss))
        rows[name] = {"action_loss": float(np.mean(losses)),
                      "split": "heldout" if name in heldout else "seen"}
        log(f"  {name:<12} {rows[name]['split']:<8} action loss {rows[name]['action_loss']:.4f}")
    policy.train()
    return rows


def build_motion_terms(args, policy, device):
    """The experiment-3 auxiliary: (image_t, image_t+h) -> Cartesian EEF delta.

    Both sources feed ONE head through the policy's OWN encoder, which is the point -- the auxiliary
    has to shape the representation the policy reads, not a parallel one that merely shares a name.

    The two corpora do not agree on quaternion order (eef_pairs is xyzw, the barx trees are wxyz --
    verified, section 9), so the real branch converts. Both express the EEF pose in the robot BASE
    frame, and the camera is base-mounted, so a base-frame displacement means the same thing in both
    and is the only convention either branch uses.
    """
    from lerobot.scripts.motion_data import motion_labels, pose_states
    from lerobot.scripts.train_motion_prediction import MotionHead, build_index, build_table
    from lerobot.scripts.motion_data import load_cache, load_images, sample_pose_pairs

    encoder = policy.diffusion.rgb_encoder
    encoder = encoder[0] if isinstance(encoder, torch.nn.ModuleList) else encoder
    dim = int(encoder.tower.config.hidden_size)
    head = MotionHead(dim).to(device)

    def tokens(images):
        x = torch.nn.functional.interpolate(images, size=(512, 512), mode="bilinear",
                                            align_corners=False) * 2.0 - 1.0
        return encoder.tower(pixel_values=x.to(dtype=encoder.tower.dtype),
                             patch_attention_mask=None).last_hidden_state

    synth = None
    if args.motion_aux in ("synthetic", "both"):
        table = build_table()
        states = pose_states(table)
        index = build_index(table)
        rng = np.random.default_rng(args.seed)
        pairs = sample_pose_pairs(states, 4000, 0.15, rng)
        from lerobot.scripts.motion_data import quat_angle_deg

        angles = quat_angle_deg(motion_labels(states, pairs)[1])
        pairs = pairs[angles <= args.synthetic_max_rot_deg]
        kept = quat_angle_deg(motion_labels(states, pairs)[1])
        # The synthetic poses were sampled to COVER the workspace, so their pairwise rotations are
        # far larger than a real trajectory segment's (60 deg vs 9). Capping narrows the gap but
        # cannot close it, and it costs motion vocabulary: state both numbers rather than let the
        # experiment quietly train on a rotation distribution the real branch never produces.
        log(f"synthetic motion after rot cap {args.synthetic_max_rot_deg} deg: {len(pairs)} pose "
            f"pairs, |d| {np.linalg.norm(motion_labels(states, pairs)[0], axis=1).mean():.3f} m, "
            f"rot {kept.mean():.1f} deg  (real at h={args.motion_horizon}: ~0.09 m, ~9 deg)")
        heldout = [int(x) for x in
                   "0,1,3,8,12,14,23,27,28,33,34,36,42,49".split(",")]
        train_emb = np.array(sorted(set(range(int(table.embodiment.max()) + 1)) - set(heldout)))
        cache = load_cache()
        from lerobot.scripts.train_motion_prediction import make_sampler

        draw = make_sampler(index, pairs, train_emb, rng, allow_gripper_change=True)
        pos_scale = float(np.linalg.norm(motion_labels(states, pairs)[0], axis=1).std())

        def synth():  # noqa: F811
            spec = draw(args.motion_batch)
            p_t = index[spec[:, 0], spec[:, 1], spec[:, 3], spec[:, 5], spec[:, 6]]
            p_h = index[spec[:, 0], spec[:, 2], spec[:, 4], spec[:, 5], spec[:, 6]]
            d_pos, q_rel = motion_labels(states, spec[:, 1:3])
            img_t = (load_images(cache, p_t, device) + 1) / 2
            img_h = (load_images(cache, p_h, device) + 1) / 2
            g = torch.tensor((spec[:, 4] % 2) - (spec[:, 3] % 2) + 1, device=device)
            return img_t, img_h, d_pos / pos_scale, q_rel, g

        log(f"synthetic motion source ready: {len(train_emb)} embodiments, pos scale {pos_scale:.4f}")
    else:
        pos_scale = 0.0338  # the synthetic scale, so the real branch is on the same footing

    def motion_loss(img_t, img_h, d_pos, q_rel, g_label):
        pred_pos, pred_quat, pred_grip = head(tokens(img_t), tokens(img_h))
        target_pos = torch.as_tensor(d_pos, dtype=torch.float32, device=device)
        target_quat = F.normalize(torch.as_tensor(q_rel, dtype=torch.float32, device=device), dim=-1)
        pos = F.mse_loss(pred_pos.float(), target_pos)
        rot = 1.0 - (F.normalize(pred_quat.float(), dim=-1) * target_quat).sum(-1).abs().clamp(max=1).mean()
        grip = F.cross_entropy(pred_grip.float(), g_label) if g_label is not None else 0.0
        return pos + rot + 0.1 * grip

    def real_terms(batch):
        """The loader returned two timesteps; index 0 is t and index 1 is t+h.

        Subsampled to --motion-batch. Using the whole policy batch would push 2x batch_size images
        through the tower with gradients on top of the policy's own forward, which OOMs an 80 GB
        card at batch 48 -- and it is not needed, since this term only has to shape the encoder.
        """
        take = min(args.motion_batch, batch[CAMERA].shape[0])
        images = batch[CAMERA][:take]
        state = batch["observation.state"][:take]
        img_t, img_h = images[:, 0], images[:, 1]
        s_t, s_h = state[:, 0].double().cpu().numpy(), state[:, 1].double().cpu().numpy()
        d_pos = (s_h[:, 7:10] - s_t[:, 7:10]) / pos_scale
        # barx stores wxyz; motion_data is xyzw throughout
        q_t = np.concatenate([s_t[:, 11:14], s_t[:, 10:11]], axis=1)
        q_h = np.concatenate([s_h[:, 11:14], s_h[:, 10:11]], axis=1)
        from lerobot.scripts.motion_data import quat_conj, quat_mul

        return img_t, img_h, d_pos, quat_mul(q_h, quat_conj(q_t)), None

    return head, motion_loss, real_terms, synth


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=[
        "online", "frozen", "finetune", "scratch",
        # dual-tower settings: the contrastive tower and a stock tower side by side. Which half is
        # frozen is the variable -- a single frozen contrastive tower gave the policy nothing
        # (CLAUDE.md 9.6), and the question is whether it helps when it is not the only source.
        "dual_both",        # both trainable
        "dual_freeze_c",    # contrastive FROZEN, stock trainable
        "dual_freeze_s",    # stock frozen, contrastive trainable
        "dual_freeze_both", # both frozen -- the floor for what the pair can contribute
    ])
    ap.add_argument("--aux-dropout", type=float, default=0.0,
                    help="drop the aux branch this often, so the policy cannot silently ignore it")
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
    ap.add_argument("--no-amp", dest="amp", action="store_false",
                    help="run the vision tower in fp32. It is ~4x slower and the features are "
                         "identical (cosine 1.00000), so this exists only to reproduce the first "
                         "round of runs, which predate the autocast.")
    ap.add_argument("--motion-aux", default="none", choices=["none", "real", "synthetic", "both"],
                    help="experiment 3: A=none, B=real, C=both")
    ap.add_argument("--motion-loss-weight", type=float, default=1.0)
    ap.add_argument("--motion-horizon", type=int, default=25,
                    help="steps ahead for the REAL pair. 25 @20fps gives |d|=0.09 m, matching the "
                         "synthetic 0.094 m; measured, not guessed")
    ap.add_argument("--synthetic-max-rot-deg", type=float, default=30.0,
                    help="cap on synthetic rotation so the two motion distributions are comparable")
    ap.add_argument("--motion-batch", type=int, default=16)
    ap.add_argument("--heldout-robots", default="",
                    help="comma-separated robot prefixes excluded from task training, e.g. UR5eOmron")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    heldout_robots = tuple(x for x in args.heldout_robots.split(",") if x)
    horizon = args.motion_horizon if args.motion_aux != "none" else 0
    dataset = build_policy_dataset(0, args.chunk, heldout_robots, horizon)
    if heldout_robots:
        log(f"held out of training entirely: {list(heldout_robots)}")
    log(f"policy corpus: {len(dataset)} frames, {dataset.num_episodes} episodes")

    policy, config = make_policy(args, dataset, device)
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    log(f"mode={args.mode}  amp={'bf16' if args.amp else 'fp32'}  "
        f"trainable {trainable / 1e6:.1f}M / {total / 1e6:.1f}M  "
        f"tower={'pretrained' if args.mode != 'scratch' else 'stock'}  "
        f"frozen={config.freeze_vision_encoder}")

    aux_batch = build_embodiment_batcher(args, device) if args.mode == "online" else None
    motion_head = motion_loss = real_terms = synth_batch = None
    if args.motion_aux != "none":
        motion_head, motion_loss, real_terms, synth_batch = build_motion_terms(args, policy, device)
        log(f"motion auxiliary: {args.motion_aux}, weight {args.motion_loss_weight}, "
            f"real horizon {args.motion_horizon} steps")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0,
    )
    params = [p for p in policy.parameters() if p.requires_grad]
    if motion_head is not None:
        params = params + list(motion_head.parameters())
    optimiser = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-6)

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
            motion_inputs = None
            if args.motion_aux in ("real", "both"):
                motion_inputs = real_terms(batch)
            if args.motion_aux != "none":
                # n_obs_steps is 1; the second timestep exists only for the auxiliary, and feeding
                # it to the policy would silently change the task the policy is being trained on.
                batch = dict(batch)
                batch[CAMERA] = batch[CAMERA][:, :1]
                batch["observation.state"] = batch["observation.state"][:, :1]

            # The step is ~95% vision tower (1021 ms of 1075 at batch 64), and the tower is
            # fp32 weights, so without this H100 tensor cores go unused. bf16 autocast keeps fp32
            # master weights -- this is NOT the bf16-parameter trap in section 8 -- and the tower's
            # features are unchanged: fp32-vs-bf16 feature cosine is 1.00000 at 5 decimals.
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if args.amp else nullcontext()
            with amp:
                loss, parts = policy.forward(batch)
            loss = loss.float()
            action_loss = float(loss)

            motion_value = float("nan")
            if args.motion_aux != "none":
                terms = []
                if motion_inputs is not None:
                    terms.append(motion_loss(*motion_inputs))
                if synth_batch is not None:
                    terms.append(motion_loss(*synth_batch()))
                motion_term = sum(terms) / len(terms)
                loss = loss + args.motion_loss_weight * motion_term
                motion_value = float(motion_term)

            aux_value = float("nan")
            if aux_batch is not None:
                # The SAME tower the policy uses, so this shapes the representation the policy
                # reads -- not a separate encoder that happens to share a name.
                images, labels = aux_batch()
                encoder = policy.diffusion.rgb_encoder
                encoder = encoder[0] if isinstance(encoder, torch.nn.ModuleList) else encoder
                with amp:
                    feats = encoder(images)
                # _supervised_contrastive_loss casts to float32 internally, which matters here for
                # the reason section 3 gives: this loss lives in the cos > 0.99 regime where
                # bfloat16's spacing near 1.0 rounds the gradient away entirely.
                aux = _supervised_contrastive_loss(feats.float(), labels, args.aux_temperature)
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
                         "motion_loss": motion_value, "total_loss": float(loss)}
                history.append(entry)
                extra = f"  aux {aux_value:6.3f}" if aux_batch is not None else ""
                extra += f"  motion {motion_value:6.3f}" if args.motion_aux != "none" else ""
                log(f"step {step:6d}/{args.steps}  action {action_loss:7.4f}{extra}  "
                    f"|g| {float(grad_norm):6.2f}  {rate:.2f} it/s  "
                    f"eta {(args.steps - step) / rate / 3600:.1f}h")

            if step % args.save_freq == 0 or step == args.steps:
                policy.save_pretrained(out_dir / f"checkpoint_{step:06d}")
                (out_dir / "history.json").write_text(
                    json.dumps({"args": vars(args), "history": history}, indent=2, default=str))

    log(f"done: {step} steps, final action loss {history[-1]['action_loss']:.4f}")
    log("per-robot evaluation:")
    per_robot = evaluate_per_robot(policy, args, device)
    (out_dir / "history.json").write_text(json.dumps(
        {"args": vars(args), "history": history, "per_robot": per_robot}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
