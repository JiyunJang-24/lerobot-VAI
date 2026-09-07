#!/usr/bin/env python
"""Experiment 08 — can static synthetic embodiment supervision replace action demonstrations?

Seven settings. The ACTION corpus is byte-identical in all of them; the only thing that changes is
what auxiliary supervision the VLM receives on top.

    A       vanilla            action demonstrations only
    B1      state_real         "Where is the gripper?" on ACTION frames
    B2      pixel_real         "Where is the end effector in the image?" on ACTION frames
    C1      state_synth        the same question on SYNTHETIC frames of many embodiments
    C2      pixel_synth        the same question on SYNTHETIC frames of many embodiments
    D       motion_synth       two synthetic frames -> the Cartesian displacement between them
    LAP     lap                the action chunk described in English, on ACTION frames

B versus C is the experiment's hinge: same objective, same question, same answer format, and the
ONLY difference is whether the frames come from embodiments that also have action demonstrations.
Any gap is attributable to embodiment diversity rather than to the auxiliary task.

BASE-RELATIVE EEF, NOT WORLD. libero_object seats the robot base on the floor while the other three
suites raise it to z=0.912, so the same gripper height reads 0.205 in one suite and 0.994 in
another. A world-frame answer would encode which suite a frame came from as much as where the
gripper is. `observation.eef_base_rel` is used throughout, and it is exactly `eef_pose[:3] -
robot_base_pos` (verified to 0.0).

    python src/lerobot/scripts/train_libero_exp08.py --aux state_synth --tag C1
"""

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.exp08_vqa import Exp08VQATokenizer  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402

LIBERO = Path("/dataset/jiyun/dataset_git/visual_robust_libero")
ACTION_TREES = [
    "IIWA_Robotiq85_spatial_t0", "IIWA_Robotiq85_spatial_t1",
    "Panda_PandaGr_object_t0", "Panda_PandaGr_object_t1",
    "UR5e_Robotiq85_goal_t4", "UR5e_Robotiq85_goal_t8",
    "Panda_Rethink_long_t2", "Panda_Rethink_long_t8",
]
SYNTH_TREES = ["libero_spatial_t0", "libero_object_t0", "libero_goal_t4", "libero_10_t2"]
CAMERA = "observation.image"
AUX = ("none", "state_real", "pixel_real", "state_synth", "pixel_synth", "motion_synth", "lap")


def log(msg: str) -> None:
    print(f"[exp08] {msg}", flush=True)


def build_action_dataset(chunk: int):
    """The eight LIBERO action trees. Identical in every setting -- do not vary this."""
    # MultiLeRobotDataset opens each tree at `root / repo_id` and keys delta_timestamps by the
    # SAME string, so repo_id has to be the bare directory name -- a "libero/" prefix makes the
    # path right and the dictionary lookup wrong.
    # LIBERO runs at 10 fps, not the 20 the RoboCasa corpora use. delta_timestamps must land on
    # exact frame boundaries or the loader rejects the whole dataset, so the rate is read from
    # info.json rather than assumed.
    import json as _json

    fps = _json.loads((LIBERO / "action" / ACTION_TREES[0] / "meta" / "info.json").read_text())["fps"]
    delta = {t: {"action": [i / fps for i in range(chunk)],
                 CAMERA: [0.0], "observation.state": [0.0]} for t in ACTION_TREES}
    return MultiLeRobotDataset(list(ACTION_TREES), root=LIBERO / "action",
                               delta_timestamps=delta, visual_cue_mode="vanilla",
                               use_wrist_cam=False, use_state=True, cache_in_memory=False)


class AuxSource:
    """Yields (images, answer sentences) for whichever auxiliary a setting uses.

    Holds its own dataset handles so the policy corpus is never touched -- the action data must be
    identical across settings, and sharing a loader would let sampling order differ between them.
    """

    def __init__(self, kind: str, tokenizer, device, batch: int, horizon: int, seed: int):
        self.kind = kind
        self.device = device
        self.batch = batch
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        self.packer = Exp08VQATokenizer(
            tokenizer, {"state_real": "state", "state_synth": "state", "pixel_real": "pixel",
                        "pixel_synth": "pixel", "motion_synth": "motion"}[kind])
        synthetic = kind.endswith("_synth")
        root = LIBERO / ("synthetic_randik" if synthetic else "action")
        names = SYNTH_TREES if synthetic else ACTION_TREES
        self.datasets = [LeRobotDataset(n, root=root / n) for n in names]
        self.lengths = [len(d) for d in self.datasets]
        total = sum(self.lengths)
        log(f"aux source '{kind}': {len(self.datasets)} trees, {total} frames "
            f"({'SYNTHETIC embodiments' if synthetic else 'action embodiments only'})")
        # Decode ONCE into RAM. Random access into these parquet-image trees costs more than a
        # training step, and with eight jobs competing it starved every auxiliary run: the two
        # settings with no auxiliary reached 1000 steps while the other five had not logged one.
        # 256x256x3 uint8 x 47k frames is under 10 GB, so the whole thing fits comfortably.
        # One cache per DOMAIN, not per setting. state_* and pixel_* read the same frames and
        # differ only in which answer column they use, so caching per setting decoded the real
        # trees twice and the synthetic trees three times -- 163k decodes for 70k distinct frames.
        domain = "synth" if synthetic else "real"
        cache = Path("/dev/shm") / f"exp08_aux_{domain}.pt"
        if cache.exists():
            log(f"  reusing {cache} ({cache.stat().st_size / 1e9:.1f} GB)")
            # weights_only=False: the blob holds numpy answer arrays alongside the image tensor,
            # and torch.load refuses those under its 2.6 default. This file is written by
            # tools/build_exp08_aux_cache.py in this repo, so there is nothing untrusted in it.
            # mmap is dropped because it requires weights_only.
            blob = torch.load(cache, weights_only=False)
        else:
            log(f"  decoding {total} frames into {cache} ...")
            images = torch.empty(total, 3, 256, 256, dtype=torch.uint8)
            base_rel, pixels, episodes, at = [], [], [], 0
            for d in self.datasets:
                for i in range(len(d)):
                    row = d[i]
                    img = row[CAMERA]
                    images[at] = (img[-1] if img.ndim == 4 else img).mul(255).round().clamp(
                        0, 255).to(torch.uint8)
                    base_rel.append(np.asarray(row["observation.eef_base_rel"], dtype=np.float64))
                    pixels.append(np.asarray(row["eef_pixel"], dtype=np.float64))
                    episodes.append(int(row["episode_index"]))
                    at += 1
                    if at % 5000 == 0:
                        log(f"    {at}/{total}")
            blob = {"images": images, "base_rel": np.stack(base_rel),
                    "pixels": np.stack(pixels), "episodes": np.array(episodes)}
            torch.save(blob, cache)
            log(f"  wrote {cache} ({cache.stat().st_size / 1e9:.1f} GB)")
        self.images = blob["images"]
        self.episodes = blob["episodes"]
        self.answers = blob["pixels"] if kind.startswith("pixel") else blob["base_rel"]
        self.total = len(self.images)

    def __call__(self):
        idx = self.rng.integers(0, self.total, size=self.batch)
        first = self.images[idx].to(self.device, dtype=torch.float32).div_(255.0)
        if self.kind.startswith("state"):
            answers = [self.packer.state_sentence(self.answers[i], np.zeros(3)) for i in idx]
            return [first], self.packer.encode(answers, self.device)
        if self.kind.startswith("pixel"):
            answers = [self.packer.pixel_sentence(self.answers[i], 256, 256) for i in idx]
            return [first], self.packer.encode(answers, self.device)
        # motion: pair with a frame `horizon` later, but only within the same episode -- a pair
        # spanning an episode boundary would be two unrelated scenes with a meaningless label
        nxt = np.minimum(idx + self.horizon, self.total - 1)
        nxt = np.where(self.episodes[nxt] == self.episodes[idx], nxt, idx)
        second = self.images[nxt].to(self.device, dtype=torch.float32).div_(255.0)
        answers = [self.packer.motion_sentence(self.answers[b] - self.answers[a])
                   for a, b in zip(idx, nxt, strict=True)]
        return [first, second], self.packer.encode(answers, self.device)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aux", required=True, choices=AUX)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--aux-batch", type=int, default=8)
    ap.add_argument("--aux-weight", type=float, default=1.0)
    ap.add_argument("--motion-horizon", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--log-freq", type=int, default=250)
    ap.add_argument("--save-freq", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "outputs/exp08"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_action_dataset(args.chunk)
    log(f"action corpus: {len(dataset)} frames over {len(ACTION_TREES)} trees "
        f"(identical in every setting)")
    sample = dataset[0]
    config = SmolVLAConfig(
        chunk_size=args.chunk, n_action_steps=args.chunk, n_obs_steps=1,
        knowledge_insulation=(args.aux == "lap"), ki_objective="lap",
        # SmolVLA defaults to train_expert_only=True, which FREEZES the VLM. Every auxiliary here
        # trains the VLM through the LM head, so with the default they cannot learn at all -- the
        # first attempt sat at CE 11.0, which is ln(49280), exactly chance for this vocabulary.
        # It is set False in EVERY setting, including the vanilla baselines: if only the auxiliary
        # runs unfroze the VLM, the comparison would confound "the auxiliary helped" with "the VLM
        # was trainable", and that is the one thing this experiment must not confound.
        train_expert_only=False,
        # LIBERO's action is [7] = dx dy dz droll dpitch dyaw gripper, not the 12-dim RoboCasa
        # layout the LAP defaults assume. Thresholds are quantiles of THIS corpus (below).
        lap_translation_dims=(0, 1, 2), lap_rotation_dims=(3, 4, 5), lap_gripper_dim=6,
        # measured quantiles of this corpus's |action| (35th / 75th), so the three magnitude
        # words are populated rather than one of them swallowing the data
        lap_translation_thresholds=(0.121, 0.549), lap_rotation_thresholds=(0.0021, 0.0514),
        lap_idle_threshold=0.02, lap_gripper_close_is_positive=True,
    )
    config.input_features = {
        CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=tuple(sample[CAMERA].shape)),
        "observation.state": PolicyFeature(type=FeatureType.STATE,
                                           shape=(sample["observation.state"].shape[-1],)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION,
                                                      shape=(sample["action"].shape[-1],))}
    config.normalization_mapping = {"VISUAL": NormalizationMode.IDENTITY,
                                    "STATE": NormalizationMode.MEAN_STD,
                                    "ACTION": NormalizationMode.MEAN_STD}
    config.device = str(device)
    dataset_stats = dataset._datasets[0].meta.stats
    policy = SmolVLAPolicy(config, dataset_stats=dataset_stats).to(device)
    # As in train_dp_embodiment: save_pretrained omits the normalizer pipelines, and without them
    # a checkpoint silently produces actions on the wrong scale at evaluation time.
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    preprocessor, postprocessor = make_smolvla_pre_post_processors(config,
                                                                   dataset_stats=dataset_stats)
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    if args.aux == "lap":
        # LAP describes the RAW command in words, so it has to undo the action normalisation.
        stats = dataset_stats["action"]
        policy.model.set_action_stats(stats["mean"], stats["std"])
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    log(f"setting {args.tag} (aux={args.aux})  trainable {trainable / 1e6:.0f}M")

    aux = None
    if args.aux not in ("none", "lap"):
        aux = AuxSource(args.aux, tokenizer, device, args.aux_batch, args.motion_horizon,
                        args.seed)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0)
    optimiser = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=1e-6)

    history, step, t0 = [], 0, time.time()
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
            if batch[CAMERA].ndim == 4:
                batch[CAMERA] = batch[CAMERA].unsqueeze(1)
            # SmolVLA reads pre-tokenised language; the dataset only carries the task string.
            tokenised = tokenizer(
                [t if t.endswith("\n") else t + "\n" for t in batch["task"]],
                padding="max_length", padding_side="right",
                max_length=config.tokenizer_max_length, return_tensors="pt")
            batch[OBS_LANGUAGE_TOKENS] = tokenised["input_ids"].to(device)
            batch[OBS_LANGUAGE_ATTENTION_MASK] = tokenised["attention_mask"].to(
                dtype=torch.bool, device=device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = policy.forward(batch)
            loss = loss.float()
            action_loss = float(loss)

            aux_value = float("nan")
            aux_acc = float("nan")
            if aux is not None:
                views, packed = aux()
                masks = [torch.ones(v.shape[0], dtype=torch.bool, device=device) for v in views]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = policy.model.vqa_state_loss(views, masks, packed)
                aux_term = out["token_ce_loss"]
                loss = loss + args.aux_weight * aux_term.float()
                aux_value = float(aux_term)
                aux_acc = float(out.get("token_content_accuracy", out["token_accuracy"]))

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 10.0)
            optimiser.step()
            step += 1

            if step % args.log_freq == 0 or step == args.steps:
                rate = step / (time.time() - t0)
                entry = {"step": step, "action_loss": action_loss, "aux_loss": aux_value,
                         "total": float(loss)}
                if aux is not None:
                    entry["aux_accuracy"] = aux_acc
                if isinstance(parts, dict):
                    for k in ("lap_token_loss", "lap_token_accuracy"):
                        if k in parts:
                            entry[k] = float(parts[k])
                history.append(entry)
                extra = (f"  aux {aux_value:7.4f} acc {aux_acc:.3f}") if aux is not None else ""
                log(f"step {step:6d}/{args.steps}  action {action_loss:7.4f}{extra}  "
                    f"|g| {float(grad):6.2f}  {rate:.2f} it/s  "
                    f"eta {(args.steps - step) / rate / 3600:.1f}h")

            if step % args.save_freq == 0 or step == args.steps:
                ckpt = out_dir / f"checkpoint_{step:06d}"
                policy.save_pretrained(ckpt)
                preprocessor.save_pretrained(ckpt)
                postprocessor.save_pretrained(ckpt)
                (out_dir / "history.json").write_text(json.dumps(
                    {"args": vars(args), "history": history}, indent=2, default=str))
    log(f"done: {step} steps -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
