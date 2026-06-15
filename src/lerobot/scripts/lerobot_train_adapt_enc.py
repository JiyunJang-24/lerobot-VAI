#!/usr/bin/env python

import copy
import dataclasses
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat

import draccus
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from termcolor import colored
from torch import Tensor
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata, MultiLeRobotDataset
from lerobot.datasets.transforms import ImageTransforms
from lerobot.datasets.utils import cycle
from lerobot.datasets.visual_cue_utils import PluckerEmbedder
from lerobot.policies.RMA import AdaptationEncoder, AdaptationEncoderConfig
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import format_big_number, init_logging


@dataclass
class AdaptEncoderTrainConfig:
    sequence_length: int = 10
    image_key: str = "observation.image"
    action_key: str = "action"
    intrinsic_key: str = "intrinsic_matrix"
    extrinsic_key: str = "extrinsic_matrix"
    target_grid_size: int = 16
    visual_backbone: str = "facebook/dinov2-small"
    freeze_visual: bool = True
    hidden_dim: int = 512
    action_feature_dim: int = 128
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_ff_dim: int = 2048
    dropout: float = 0.1
    use_cls_token: bool = True
    dinov2_image_size: int = 224


@dataclass
class AdaptEncoderPipelineConfig(TrainPipelineConfig):
    adapt: AdaptEncoderTrainConfig = field(default_factory=AdaptEncoderTrainConfig)


def _make_sequence_delta_timestamps(
    ds_meta: LeRobotDatasetMetadata,
    cfg: AdaptEncoderTrainConfig,
) -> dict[str, list[float]]:
    indices = list(range(1 - cfg.sequence_length, 1))
    delta_timestamps = {
        cfg.image_key: [idx / ds_meta.fps for idx in indices],
        cfg.action_key: [idx / ds_meta.fps for idx in indices],
    }
    return delta_timestamps


def make_adaptation_dataset(cfg: AdaptEncoderPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if cfg.dataset.repo_id.startswith("["):
        if ";" in cfg.dataset.repo_id:
            repo_ids = cfg.dataset.repo_id.strip("[]").strip("''").split(";")
        else:
            repo_ids = cfg.dataset.repo_id.strip("[]").strip("''").split(",")
        repo_ids = [repo_id.strip().replace("'", "") for repo_id in repo_ids]

        delta_timestamps = {}
        for repo_id in repo_ids:
            ds_meta = LeRobotDatasetMetadata(
                repo_id,
                root=f"{cfg.dataset.root}/{repo_id}",
                revision=cfg.dataset.revision,
            )
            delta_timestamps[repo_id] = _make_sequence_delta_timestamps(ds_meta, cfg.adapt)

        dataset = MultiLeRobotDataset(
            repo_ids,
            root=cfg.dataset.root,
            episodes={},
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
            visual_cue_mode="vanilla",
            use_wrist_cam=cfg.dataset.use_wrist_cam,
            use_state=cfg.dataset.use_state,
            cache_in_memory=cfg.dataset.cache_in_memory,
        )
        logging.info("Multiple datasets were provided: %s", pformat(dataset.repo_id_to_index, indent=2))
        return dataset

    ds_meta = LeRobotDatasetMetadata(cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision)
    delta_timestamps = _make_sequence_delta_timestamps(ds_meta, cfg.adapt)
    return LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        tolerance_s=cfg.tolerance_s,
        cache_in_memory=cfg.dataset.cache_in_memory,
    )


def _get_dataset_meta(dataset: LeRobotDataset | MultiLeRobotDataset):
    if isinstance(dataset, MultiLeRobotDataset):
        ds_meta = copy.copy(dataset._datasets[0].meta)
        ds_meta.episodes = dataset.meta_episodes
        ds_meta.stats = dataset.stats.get("panda", dataset.stats) if isinstance(dataset.stats, dict) else dataset.stats
        return ds_meta
    return dataset.meta


def _episode_table_to_rows(ds_meta) -> list[dict]:
    episodes = ds_meta.episodes
    if hasattr(episodes, "to_pandas"):
        episodes = episodes.to_pandas()
    if hasattr(episodes, "to_dict"):
        return episodes.to_dict("records")
    return list(episodes)


def build_valid_sequence_indices(ds_meta, sequence_length: int) -> list[int]:
    valid_indices = []
    for episode in _episode_table_to_rows(ds_meta):
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        first_valid = start + sequence_length - 1
        if first_valid < end:
            valid_indices.extend(range(first_valid, end))
    return valid_indices


def _last_if_sequence(tensor: Tensor) -> Tensor:
    if tensor.ndim >= 4 and tensor.shape[-2:] in ((3, 3), (4, 4)):
        return tensor[:, -1]
    return tensor


def remove_camera_axis_correction(extrinsics: Tensor) -> Tensor:
    camera_axis_correction = extrinsics.new_tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return extrinsics @ camera_axis_correction


def prepare_batch(
    batch: dict,
    cfg: AdaptEncoderTrainConfig,
    plucker_embedder: PluckerEmbedder,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    images = batch[cfg.image_key].to(device, non_blocking=True)
    actions = batch[cfg.action_key].to(device, non_blocking=True).float()
    intrinsics = _last_if_sequence(batch[cfg.intrinsic_key].to(device, non_blocking=True).float())
    extrinsics = _last_if_sequence(batch[cfg.extrinsic_key].to(device, non_blocking=True).float())

    if images.ndim == 4:
        images = images.unsqueeze(1)
    if actions.ndim == 2:
        actions = actions.unsqueeze(1)
    if images.shape[1] != cfg.sequence_length or actions.shape[1] != cfg.sequence_length:
        raise ValueError(
            f"Expected {cfg.sequence_length} image/action steps, got images={tuple(images.shape)}, "
            f"actions={tuple(actions.shape)}"
        )
    if images.shape[2] > 3:
        images = images[:, :, :3]

    extrinsics = remove_camera_axis_correction(extrinsics)
    plucker = plucker_embedder(intrinsics, extrinsics)["plucker"]
    plucker = plucker.permute(0, 3, 1, 2).contiguous()
    target = plucker.flatten(start_dim=1)
    return images, actions, target


def save_adaptation_checkpoint(
    output_dir: Path,
    step: int,
    cfg: AdaptEncoderPipelineConfig,
    model: AdaptationEncoder,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
) -> None:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    payload = {
        "step": step,
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict(),
        "adapt_config": dataclasses.asdict(cfg.adapt),
    }
    checkpoint_path = checkpoint_dir / f"step_{step:06d}.pt"
    accelerator.save(payload, checkpoint_path)
    accelerator.save(payload, checkpoint_dir / "last.pt")
    with open(output_dir / "train_config.json", "w") as f, draccus.config_type("json"):
        draccus.dump(cfg, f, indent=4)


@parser.wrap()
def train(cfg: AdaptEncoderPipelineConfig, accelerator: Accelerator | None = None):
    cfg.validate()
    cfg.eval_freq = 0
    if cfg.seed is not None:
        set_seed(cfg.seed)

    if accelerator is None:
        accelerator = Accelerator(cpu=cfg.policy.device == "cpu")

    init_logging(accelerator=accelerator)
    is_main_process = accelerator.is_main_process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if is_main_process:
        logging.info("Creating adaptation dataset")
    dataset = make_adaptation_dataset(cfg)
    ds_meta = _get_dataset_meta(dataset)
    valid_indices = build_valid_sequence_indices(ds_meta, cfg.adapt.sequence_length)
    if not valid_indices:
        raise ValueError(
            f"No valid {cfg.adapt.sequence_length}-step windows found. "
            "Check dataset episode lengths and adapt.sequence_length."
        )
    train_dataset = torch.utils.data.Subset(dataset, valid_indices)

    action_shape = ds_meta.features[cfg.adapt.action_key]["shape"]
    action_dim = int(action_shape[-1])
    if cfg.adapt.target_grid_size <= 0 or 256 % cfg.adapt.target_grid_size != 0:
        raise ValueError("adapt.target_grid_size must be a positive divisor of 256")
    plucker_dim = cfg.adapt.target_grid_size * cfg.adapt.target_grid_size * 6

    model = AdaptationEncoder(
        AdaptationEncoderConfig(
            action_dim=action_dim,
            plucker_dim=plucker_dim,
            visual_backbone=cfg.adapt.visual_backbone,
            action_feature_dim=cfg.adapt.action_feature_dim,
            hidden_dim=cfg.adapt.hidden_dim,
            transformer_layers=cfg.adapt.transformer_layers,
            transformer_heads=cfg.adapt.transformer_heads,
            transformer_ff_dim=cfg.adapt.transformer_ff_dim,
            dropout=cfg.adapt.dropout,
            freeze_visual=cfg.adapt.freeze_visual,
            use_cls_token=cfg.adapt.use_cls_token,
            image_size=cfg.adapt.dinov2_image_size,
        )
    )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=cfg.optimizer.lr,
        betas=cfg.optimizer.betas,
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay,
    )

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=not cfg.dataset.streaming,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=cfg.dataloader_prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.dataloader_persistent_workers if cfg.num_workers > 0 else False,
    )

    plucker_embedder = PluckerEmbedder(
        img_size=256,
        patch_size=256 // cfg.adapt.target_grid_size,
        device=device,
    ).to(device)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    dl_iter = cycle(dataloader)

    if is_main_process:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"valid_sequence_windows={len(train_dataset)}")
        logging.info(f"{dataset.num_episodes=}")
        num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"{action_dim=}, {plucker_dim=}, fps={ds_meta.fps}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    train_metrics = {
        "loss": AverageMeter("loss", ":.4f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        cfg.batch_size * accelerator.num_processes,
        len(train_dataset),
        dataset.num_episodes,
        train_metrics,
        accelerator=accelerator,
    )

    model.train()
    for step in tqdm(range(cfg.steps), disable=not is_main_process):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        fetch_done_time = time.perf_counter()
        images, actions, target = prepare_batch(batch, cfg.adapt, plucker_embedder, device)

        with accelerator.autocast():
            pred = model(images, actions)
            loss = F.mse_loss(pred, target)

        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step_id = step + 1
        train_tracker.loss = loss.item()
        train_tracker.grad_norm = grad_norm.item()
        train_tracker.lr = optimizer.param_groups[0]["lr"]
        train_tracker.update_s = time.perf_counter() - fetch_done_time
        train_tracker.dataloading_s = fetch_done_time - start_time
        train_tracker.step()

        if cfg.log_freq > 0 and step_id % cfg.log_freq == 0 and is_main_process:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_logger.log_dict(train_tracker.to_dict(), step_id)
            train_tracker.reset_averages()

        is_saving_step = cfg.save_checkpoint and (step_id % cfg.save_freq == 0 or step_id == cfg.steps)
        if is_saving_step:
            if is_main_process:
                save_adaptation_checkpoint(
                    cfg.output_dir,
                    step_id,
                    cfg,
                    model,
                    optimizer,
                    accelerator,
                )
            accelerator.wait_for_everyone()

    if is_main_process:
        logging.info("End of adaptation encoder training")
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
