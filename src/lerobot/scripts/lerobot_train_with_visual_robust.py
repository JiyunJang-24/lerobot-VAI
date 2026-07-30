#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
import logging
import time
import copy
from contextlib import nullcontext
from pprint import pformat
from typing import Any

from tqdm import tqdm

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)
from lerobot.datasets.lerobot_dataset import (
    MultiLeRobotDataset,
)
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
from collections import defaultdict
import os
from pathlib import Path


POLICY_CAMERA_KEYS = ("observation.image", "observation.wrist_image")


def unwrap_model(model):
    # DDP/FSDP 등으로 감싸진 경우를 대비
    return getattr(model, "module", model)

def dump_trainable_report(model, out_path="trainable_report.txt", topk=50):
    model = unwrap_model(model)

    # 1) 파라미터 리스트 수집
    trainable = []
    frozen = []

    total_params = 0
    trainable_params = 0

    # 큰 단위(최상위 모듈)로 집계: "backbone.xxx" -> "backbone"
    group_stats = defaultdict(lambda: {"trainable": 0, "frozen": 0, "trainable_cnt": 0, "frozen_cnt": 0})

    for name, p in model.named_parameters():
        n = p.numel()
        total_params += n
        group = name.split(".", 1)[0] if "." in name else name  # top-level group

        info = {
            "name": name,
            "shape": tuple(p.shape),
            "numel": n,
            "requires_grad": bool(p.requires_grad),
            "dtype": str(p.dtype),
            "device": str(p.device),
        }

        if p.requires_grad:
            trainable.append(info)
            trainable_params += n
            group_stats[group]["trainable"] += n
            group_stats[group]["trainable_cnt"] += 1
        else:
            frozen.append(info)
            group_stats[group]["frozen"] += n
            group_stats[group]["frozen_cnt"] += 1

    # 2) 콘솔 출력: 큰 단위 요약
    print("=" * 80)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,} ({(trainable_params/total_params*100 if total_params else 0):.2f}%)")
    print(f"Frozen params: {total_params - trainable_params:,}")
    print("-" * 80)
    print("[Top-level module summary]")
    rows = []
    for g, st in group_stats.items():
        rows.append((g, st["trainable"], st["frozen"], st["trainable_cnt"], st["frozen_cnt"]))
    # trainable numel 기준 내림차순
    rows.sort(key=lambda x: x[1], reverse=True)

    for g, tr_n, fr_n, tr_c, fr_c in rows[:topk]:
        total_g = tr_n + fr_n
        pct = (tr_n / total_g * 100) if total_g else 0.0
        print(f"- {g:30s} | trainable {tr_n:12,} ({pct:6.2f}%)  | frozen {fr_n:12,}  | (param cnt: tr {tr_c}, fr {fr_c})")
    if len(rows) > topk:
        print(f"... ({len(rows)-topk} more groups)")
    print("=" * 80)

    # 3) txt 저장
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("=== Trainable / Frozen Parameter Report ===\n")
        f.write(f"Total params: {total_params:,}\n")
        f.write(f"Trainable params: {trainable_params:,} ({(trainable_params/total_params*100 if total_params else 0):.2f}%)\n")
        f.write(f"Frozen params: {total_params - trainable_params:,}\n\n")

        f.write("=== Top-level module summary ===\n")
        for g, tr_n, fr_n, tr_c, fr_c in rows:
            total_g = tr_n + fr_n
            pct = (tr_n / total_g * 100) if total_g else 0.0
            f.write(f"- {g} | trainable {tr_n:,} ({pct:.2f}%) | frozen {fr_n:,} | (param cnt: tr {tr_c}, fr {fr_c})\n")
        f.write("\n")

        def write_param_list(title, plist):
            f.write(f"=== {title} (count={len(plist)}) ===\n")
            for info in plist:
                f.write(
                    f"{info['name']}\tshape={info['shape']}\tnumel={info['numel']:,}\t"
                    f"requires_grad={info['requires_grad']}\tdtype={info['dtype']}\tdevice={info['device']}\n"
                )
            f.write("\n")

        # 이름 전체 덤프
        write_param_list("Trainable parameters", trainable)
        write_param_list("Frozen parameters", frozen)

    print(f"[Saved] {out_path}")


def _get_unwrapped_policy(policy: PreTrainedPolicy, accelerator: Accelerator) -> PreTrainedPolicy:
    return accelerator.unwrap_model(policy, keep_fp32_wrapper=True)


def _select_visual_robust_image_keys(batch, image_prefix: str, max_views=None):
    image_keys = sorted(
        key
        for key, value in batch.items()
        if key.startswith(image_prefix)
        and torch.is_tensor(value)
        and value.ndim >= 4
        and value.shape[-3] == 3
    )

    if max_views is not None and max_views > 0:
        image_keys = image_keys[:max_views]

    return image_keys


def _is_visual_robust_auxiliary_key(key: str) -> bool:
    return (
        key.startswith("observation.image.")
        or key.startswith("observation.wrist_image.")
        or key.startswith("intrinsic_matrix.")
        or key.startswith("extrinsic_matrix.")
    )


def _ensure_policy_camera_features(cfg: TrainPipelineConfig, ds_meta: Any, is_main_process: bool) -> None:
    if not cfg.policy.input_features:
        return

    dataset_policy_features = dataset_to_policy_features(ds_meta.features)
    required_camera_keys = ["observation.image"]
    if cfg.dataset.use_wrist_cam:
        required_camera_keys.append("observation.wrist_image")

    added_camera_keys = []
    for key in required_camera_keys:
        if key in dataset_policy_features and key not in cfg.policy.input_features:
            cfg.policy.input_features[key] = dataset_policy_features[key]
            added_camera_keys.append(key)

    if added_camera_keys and is_main_process:
        logging.info(
            "Added dataset camera feature(s) to existing policy input_features because use_wrist_cam=%s: %s",
            cfg.dataset.use_wrist_cam,
            added_camera_keys,
        )


def _prepare_visual_robust_images(
    policy: PreTrainedPolicy, batch: dict[str, Any], image_keys: list[str], device: torch.device
) -> torch.Tensor:
    images = []

    for key in image_keys:
        img = batch[key].to(device, non_blocking=True)
        img = img[:, -1] if img.ndim == 5 else img
        if policy.config.resize_imgs_with_padding is not None:
            img = resize_with_pad(img, *policy.config.resize_imgs_with_padding, pad_value=0)
        images.append(img * 2.0 - 1.0)

    return torch.stack(images, dim=1)


def _supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    features = F.normalize(features.float(), dim=-1)
    logits = features @ features.T
    logits = logits / temperature

    eye = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~eye
    logits_mask = ~eye

    logits = logits.masked_fill(~logits_mask, torch.finfo(logits.dtype).min)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    positive_counts = positive_mask.sum(dim=1)
    valid = positive_counts > 0
    if not torch.any(valid):
        return features.new_zeros(())

    mean_log_prob_pos = (log_prob * positive_mask).sum(dim=1)[valid] / positive_counts[valid]
    return -mean_log_prob_pos.mean()


def _positive_group_alignment_loss(features: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
    features = F.normalize(features.float(), dim=-1)
    if batch_size < 1 or num_views < 2:
        return features.new_zeros(())

    grouped_features = features.view(batch_size, num_views, -1)
    similarities = grouped_features @ grouped_features.transpose(1, 2)
    eye = torch.eye(num_views, dtype=torch.bool, device=similarities.device)
    return 1.0 - similarities[:, ~eye].mean()


def _gripper_width_values(
    batch: dict[str, Any],
    *,
    batch_size: int,
    device: torch.device,
    state_key: str,
    left_index: int,
    right_index: int,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    if state_key not in batch:
        return None, {}

    state = batch[state_key]
    if not torch.is_tensor(state):
        state = torch.as_tensor(state)
    state = state.to(device=device, dtype=torch.float32, non_blocking=True)
    state = state[:, -1] if state.ndim == 3 else state
    if state.ndim != 2 or state.shape[0] != batch_size:
        return None, {}

    max_index = max(left_index, right_index)
    min_index = min(left_index, right_index)
    if min_index < 0 or max_index >= state.shape[-1]:
        raise ValueError(
            f"Cannot compute gripper width from {state_key} with shape {tuple(state.shape)} "
            f"and indices ({left_index}, {right_index})."
        )

    width = torch.abs(state[:, left_index] - state[:, right_index])
    metrics = {
        "gripper_width_mean": width.detach().mean().float().item(),
        "gripper_width_min": width.detach().min().float().item(),
        "gripper_width_max": width.detach().max().float().item(),
    }
    return width, metrics


def _normalize_gripper_width(
    width: torch.Tensor,
    *,
    width_min: float,
    width_max: float,
) -> torch.Tensor:
    if width_max <= width_min:
        raise ValueError("--dataset.visual_robust_wrist_width_max must be greater than wrist_width_min.")
    return ((width - width_min) / (width_max - width_min)).clamp(0.0, 1.0)


def _gripper_width_bin_labels(
    batch: dict[str, Any],
    *,
    batch_size: int,
    num_views: int,
    device: torch.device,
    state_key: str,
    left_index: int,
    right_index: int,
    bin_size: float,
    width_min: float,
    width_max: float,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    if bin_size <= 0:
        raise ValueError("--dataset.visual_robust_wrist_width_bin_size must be positive.")
    width, metrics = _gripper_width_values(
        batch,
        batch_size=batch_size,
        device=device,
        state_key=state_key,
        left_index=left_index,
        right_index=right_index,
    )
    if width is None:
        return None, {}

    normalized_width = _normalize_gripper_width(width, width_min=width_min, width_max=width_max)
    width_bins = torch.round(width / bin_size).to(torch.long)
    normalized_width_bins = torch.round(normalized_width / bin_size).to(torch.long)
    labels = normalized_width_bins.repeat_interleave(num_views)
    metrics.update({
        "gripper_opening_mean": normalized_width.detach().mean().float().item(),
        "gripper_opening_min": normalized_width.detach().min().float().item(),
        "gripper_opening_max": normalized_width.detach().max().float().item(),
        "gripper_width_bin_size": float(bin_size),
        "gripper_width_unique_bins": float(torch.unique(width_bins).numel()),
        "gripper_opening_unique_bins": float(torch.unique(normalized_width_bins).numel()),
        "gripper_width_normalization_min": float(width_min),
        "gripper_width_normalization_max": float(width_max),
    })
    return labels, metrics


def _continuous_gripper_opening_loss(
    features: torch.Tensor,
    opening: torch.Tensor,
    *,
    num_views: int,
    sigma: float,
) -> torch.Tensor:
    if sigma <= 0:
        raise ValueError("--dataset.visual_robust_wrist_width_sigma must be positive.")
    features = F.normalize(features.float(), dim=-1)
    opening = opening.repeat_interleave(num_views).to(device=features.device, dtype=torch.float32)
    similarities = features @ features.T
    target = torch.exp(-torch.abs(opening[:, None] - opening[None, :]) / sigma)
    eye = torch.eye(similarities.shape[0], dtype=torch.bool, device=similarities.device)
    return F.mse_loss(similarities[~eye], target[~eye])


def _encode_visual_robust_features(
    policy: PreTrainedPolicy,
    batch: dict[str, Any],
    accelerator: Accelerator,
    max_views: int | None,
    encoder_chunk_size: int,
    image_prefix: str,
) -> tuple[torch.Tensor | None, int, int]:
    unwrapped_policy = _get_unwrapped_policy(policy, accelerator)
    if unwrapped_policy.config.type != "smolvla":
        return None, 0, 0

    image_keys = _select_visual_robust_image_keys(batch, image_prefix=image_prefix, max_views=max_views)
    if len(image_keys) < 2:
        return None, len(image_keys), 0

    views = _prepare_visual_robust_images(unwrapped_policy, batch, image_keys, accelerator.device)
    batch_size, num_views = views.shape[:2]
    flat_images = views.flatten(0, 1)

    vision_model = unwrapped_policy.model.vlm_with_expert.get_vlm_model().vision_model
    vision_feature_chunks = []
    for image_chunk in flat_images.split(encoder_chunk_size, dim=0):
        vision_feature_chunks.append(
            vision_model(
                pixel_values=image_chunk.to(dtype=vision_model.dtype),
                patch_attention_mask=None,
            ).last_hidden_state
        )
    vision_features = torch.cat(vision_feature_chunks, dim=0)
    return vision_features.mean(dim=1), batch_size, num_views


def compute_visual_robust_contrastive_loss(
    policy: PreTrainedPolicy,
    batch: dict[str, Any],
    accelerator: Accelerator,
    temperature: float,
    max_views: int | None,
    encoder_chunk_size: int,
    image_prefix: str = "observation.image.",
    metric_prefix: str = "visual_robust",
) -> tuple[torch.Tensor | None, dict[str, float]]:
    pooled_features, batch_size, num_views = _encode_visual_robust_features(
        policy=policy,
        batch=batch,
        accelerator=accelerator,
        max_views=max_views,
        encoder_chunk_size=encoder_chunk_size,
        image_prefix=image_prefix,
    )
    if pooled_features is None:
        return None, {f"{metric_prefix}_num_views": float(batch_size)}

    labels = torch.arange(batch_size, device=pooled_features.device).repeat_interleave(num_views)
    loss = _supervised_contrastive_loss(pooled_features, labels, temperature=temperature)
    metrics = {
        f"{metric_prefix}_contrastive_loss": loss.detach().float().item(),
        f"{metric_prefix}_num_views": float(num_views),
    }
    if "episode_index" in batch:
        metrics[f"{metric_prefix}_episode_unique"] = float(torch.unique(batch["episode_index"]).numel())
    if "dataset_index" in batch:
        metrics[f"{metric_prefix}_dataset_unique"] = float(torch.unique(batch["dataset_index"]).numel())
    return loss, metrics


def compute_visual_robust_alignment_loss(
    policy: PreTrainedPolicy,
    batch: dict[str, Any],
    accelerator: Accelerator,
    max_views: int | None,
    encoder_chunk_size: int,
    image_prefix: str,
    metric_prefix: str,
    alignment_mode: str = "all",
    temperature: float = 0.1,
    width_state_key: str = "observation.state",
    width_left_index: int = 0,
    width_right_index: int = 1,
    width_bin_size: float = 0.01,
    width_min: float = 0.0,
    width_max: float = 0.08,
    width_sigma: float = 0.2,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    pooled_features, batch_size, num_views = _encode_visual_robust_features(
        policy=policy,
        batch=batch,
        accelerator=accelerator,
        max_views=max_views,
        encoder_chunk_size=encoder_chunk_size,
        image_prefix=image_prefix,
    )
    if pooled_features is None:
        return None, {f"{metric_prefix}_num_views": float(num_views)}

    mode = alignment_mode.lower()
    width_metrics = {}
    if mode == "all":
        loss = _positive_group_alignment_loss(pooled_features, batch_size=batch_size, num_views=num_views)
    elif mode in {"width_bins", "gripper_width_bins"}:
        labels, width_metrics = _gripper_width_bin_labels(
            batch,
            batch_size=batch_size,
            num_views=num_views,
            device=pooled_features.device,
            state_key=width_state_key,
            left_index=width_left_index,
            right_index=width_right_index,
            bin_size=width_bin_size,
            width_min=width_min,
            width_max=width_max,
        )
        if labels is None:
            return None, {
                f"{metric_prefix}_num_views": float(num_views),
                f"{metric_prefix}_missing_width_state": 1.0,
            }
        loss = _supervised_contrastive_loss(pooled_features, labels, temperature=temperature)
    elif mode in {"width_continuous", "gripper_width_continuous", "opening_continuous"}:
        width, width_metrics = _gripper_width_values(
            batch,
            batch_size=batch_size,
            device=pooled_features.device,
            state_key=width_state_key,
            left_index=width_left_index,
            right_index=width_right_index,
        )
        if width is None:
            return None, {
                f"{metric_prefix}_num_views": float(num_views),
                f"{metric_prefix}_missing_width_state": 1.0,
            }
        opening = _normalize_gripper_width(width, width_min=width_min, width_max=width_max)
        width_metrics.update({
            "gripper_opening_mean": opening.detach().mean().float().item(),
            "gripper_opening_min": opening.detach().min().float().item(),
            "gripper_opening_max": opening.detach().max().float().item(),
            "gripper_width_normalization_min": float(width_min),
            "gripper_width_normalization_max": float(width_max),
            "gripper_width_sigma": float(width_sigma),
        })
        loss = _continuous_gripper_opening_loss(
            pooled_features,
            opening,
            num_views=num_views,
            sigma=width_sigma,
        )
    else:
        raise ValueError(
            "--dataset.visual_robust_wrist_alignment_mode must be one of "
            f"'all', 'width_bins', or 'width_continuous', got {alignment_mode!r}."
        )

    metrics = {
        f"{metric_prefix}_alignment_loss": loss.detach().float().item(),
        f"{metric_prefix}_alignment_mode_{mode}": 1.0,
        f"{metric_prefix}_num_views": float(num_views),
        f"{metric_prefix}_num_positive_groups": float(batch_size),
        f"{metric_prefix}_num_embeddings": float(pooled_features.shape[0]),
    }
    metrics.update({f"{metric_prefix}_{key}": value for key, value in width_metrics.items()})
    if "episode_index" in batch:
        metrics[f"{metric_prefix}_episode_unique"] = float(torch.unique(batch["episode_index"]).numel())
    if "dataset_index" in batch:
        metrics[f"{metric_prefix}_dataset_unique"] = float(torch.unique(batch["dataset_index"]).numel())
    return loss, metrics


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    visual_robust_batch: Any | None,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    rabc_weights_provider=None,
    visual_robust_contrastive_weight: float = 0.0,
    visual_robust_temperature: float = 0.1,
    visual_robust_max_views: int | None = None,
    visual_robust_wrist_alignment_weight: float = 0.0,
    visual_robust_wrist_alignment_mode: str = "all",
    visual_robust_wrist_alignment_max_views: int | None = None,
    visual_robust_wrist_width_bin_size: float = 0.01,
    visual_robust_wrist_width_temperature: float = 0.1,
    visual_robust_wrist_width_min: float = 0.0,
    visual_robust_wrist_width_max: float = 0.08,
    visual_robust_wrist_width_sigma: float = 0.2,
    visual_robust_wrist_width_state_key: str = "observation.state",
    visual_robust_wrist_width_left_index: int = 0,
    visual_robust_wrist_width_right_index: int = 1,
    visual_robust_encoder_chunk_size: int = 32,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        rabc_weights_provider: Optional RABCWeights instance for sample weighting.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Get RA-BC weights if enabled
    rabc_batch_weights = None
    rabc_batch_stats = None
    if rabc_weights_provider is not None:
        rabc_batch_weights, rabc_batch_stats = rabc_weights_provider.compute_batch_weights(batch)

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        # Use per-sample loss when RA-BC is enabled for proper weighting
        if rabc_batch_weights is not None:
            # Get per-sample losses
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")

            # Apply RA-BC weights: L_RA-BC = Σ(w_i * l_i) / (Σw_i + ε)
            # rabc_batch_weights is already normalized to sum to batch_size
            epsilon = 1e-6
            loss = (per_sample_loss * rabc_batch_weights).sum() / (rabc_batch_weights.sum() + epsilon)
            # Log raw mean weight (before normalization) - this is the meaningful metric
            output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
            output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
            output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
        else:
            loss, output_dict = policy.forward(batch)

        if visual_robust_contrastive_weight > 0:
            contrastive_loss, contrastive_metrics = compute_visual_robust_contrastive_loss(
                policy=policy,
                batch=visual_robust_batch if visual_robust_batch is not None else batch,
                accelerator=accelerator,
                temperature=visual_robust_temperature,
                max_views=visual_robust_max_views,
                encoder_chunk_size=visual_robust_encoder_chunk_size,
                image_prefix="observation.image.",
                metric_prefix="visual_robust",
            )
            if contrastive_loss is not None:
                loss = loss + visual_robust_contrastive_weight * contrastive_loss
                output_dict.update(contrastive_metrics)
                output_dict["visual_robust_contrastive_weight"] = visual_robust_contrastive_weight
                output_dict["loss_with_visual_robust"] = loss.detach().float().item()

        if visual_robust_wrist_alignment_weight > 0:
            wrist_alignment_loss, wrist_alignment_metrics = compute_visual_robust_alignment_loss(
                policy=policy,
                batch=visual_robust_batch if visual_robust_batch is not None else batch,
                accelerator=accelerator,
                max_views=visual_robust_wrist_alignment_max_views,
                encoder_chunk_size=visual_robust_encoder_chunk_size,
                image_prefix="observation.wrist_image.",
                metric_prefix="visual_robust_wrist",
                alignment_mode=visual_robust_wrist_alignment_mode,
                temperature=visual_robust_wrist_width_temperature,
                width_state_key=visual_robust_wrist_width_state_key,
                width_left_index=visual_robust_wrist_width_left_index,
                width_right_index=visual_robust_wrist_width_right_index,
                width_bin_size=visual_robust_wrist_width_bin_size,
                width_min=visual_robust_wrist_width_min,
                width_max=visual_robust_wrist_width_max,
                width_sigma=visual_robust_wrist_width_sigma,
            )
            if wrist_alignment_loss is not None:
                loss = loss + visual_robust_wrist_alignment_weight * wrist_alignment_loss
                output_dict.update(wrist_alignment_metrics)
                output_dict["visual_robust_wrist_alignment_weight"] = visual_robust_wrist_alignment_weight
                output_dict["loss_with_visual_robust"] = loss.detach().float().item()

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def _parse_repo_ids(repo_ids: str) -> list[str]:
    repo_ids = repo_ids.strip()
    if repo_ids.startswith("[") and repo_ids.endswith("]"):
        repo_ids = repo_ids[1:-1]
    return [repo_id.strip().strip("'").strip('"') for repo_id in repo_ids.split(",") if repo_id.strip()]


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    scalar_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            scalar_metrics[key] = float(value)
        elif torch.is_tensor(value) and value.numel() == 1:
            scalar_metrics[key] = value.detach().float().item()
    return scalar_metrics


class SameEpisodeBatchSampler:
    def __init__(self, episodes, batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.episode_indices = []

        for episode in episodes.itertuples(index=False):
            start_idx = int(episode.dataset_from_index)
            end_idx = int(episode.dataset_to_index)
            if end_idx - start_idx >= batch_size:
                self.episode_indices.append(list(range(start_idx, end_idx)))

        if not self.episode_indices:
            raise ValueError(
                "No visual robust episode is long enough for same-episode negative sampling. "
                f"Reduce --dataset.visual_robust_batch_size below {batch_size}."
            )

    def __iter__(self):
        episode_order = torch.randperm(len(self.episode_indices)).tolist() if self.shuffle else range(len(self.episode_indices))
        for episode_idx in episode_order:
            indices = self.episode_indices[episode_idx]
            frame_order = torch.randperm(len(indices)).tolist() if self.shuffle else range(len(indices))
            shuffled_indices = [indices[i] for i in frame_order]
            for start in range(0, len(shuffled_indices) - self.batch_size + 1, self.batch_size):
                yield shuffled_indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return sum(len(indices) // self.batch_size for indices in self.episode_indices)


def make_visual_robust_contrastive_loader(
    *,
    root: str,
    repo_ids: str,
    batch_size: int,
    num_workers: int,
    cache_in_memory: bool,
    video_backend: str | None,
    same_episode_negatives: bool,
) -> torch.utils.data.DataLoader:
    contrastive_repo_ids = _parse_repo_ids(repo_ids)
    if not contrastive_repo_ids:
        raise ValueError("--dataset.visual_robust_repo_id is empty. Contrastive dataset cannot be created.")

    dataset = MultiLeRobotDataset(
        contrastive_repo_ids,
        root=Path(root),
        delta_timestamps={repo_id: None for repo_id in contrastive_repo_ids},
        video_backend=video_backend,
        visual_cue_mode="vanilla",
        use_wrist_cam=False,
        use_state=True,
        cache_in_memory=cache_in_memory,
    )

    batch_sampler = None
    if same_episode_negatives:
        batch_sampler = SameEpisodeBatchSampler(dataset.meta_episodes, batch_size=batch_size, shuffle=True)

    dataloader_kwargs = {
        "dataset": dataset,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "prefetch_factor": 2 if num_workers > 0 else None,
        "persistent_workers": num_workers > 0,
    }
    if batch_sampler is not None:
        dataloader_kwargs["batch_sampler"] = batch_sampler
    else:
        dataloader_kwargs.update({"batch_size": batch_size, "shuffle": True, "drop_last": True})

    return torch.utils.data.DataLoader(**dataloader_kwargs)


def get_default_peft_configuration(policy_type):
    """Build a basic PEFT configuration for the given policy type assuming that we train a policy from a checkpoint."""

    common_projections = "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"

    if policy_type == "smolvla":
        return {
            "target_modules": rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))",
            "modules_to_save": [],
        }
    elif policy_type in ("pi0", "pi05"):
        return {
            "target_modules": rf"(.*\.gemma_expert\..*\.self_attn.(q|v)_proj|model\.({common_projections}))",
            "modules_to_save": [],
        }

    return {"modules_to_save": None}


def wrap_policy_in_peft_model(cfg, policy):
    from peft import PEFT_TYPE_TO_CONFIG_MAPPING, PeftType, get_peft_model

    # Disable all gradients because we'll only train the parameters selected by the PEFT method.
    # Layers that should receive gradients anyway need to be listed in `modules_to_save`.
    for p in policy.parameters():
        p.requires_grad_(False)

    if not cfg.policy.pretrained_path:
        raise ValueError(
            "Training from scratch using PEFT. This is unlikely to yield good results. "
            "Supply a `policy.path` to fine-tune an existing model."
        )

    if cfg.policy.type == "smolvla" and not cfg.policy.load_vlm_weights:
        logging.warning(
            "Training SmolVLA from scratch using PEFT. This is unlikely to yield good results. Set "
            "`load_vlm_weights=True` to fine-tune the existing policy."
        )

    peft_config_policy = get_default_peft_configuration(cfg.policy.type)
    peft_config_cli = dataclasses.asdict(cfg.peft) if cfg.peft else {}
    peft_config_cli["modules_to_save"] = peft_config_cli["full_training_modules"]  # compatibility with PEFT
    peft_method_type = PeftType[peft_config_cli["method_type"].upper()]
    peft_config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_method_type]

    # Handle specific CLI overrides
    for key in ["target_modules", "modules_to_save", "r"]:
        if peft_config_cli[key] is not None:
            peft_config_policy[key] = peft_config_cli[key]

    if "target_modules" not in peft_config_policy:
        raise ValueError(
            f"There is no default `target_modules` value for policy {cfg.policy.type}. Please pass it manually."
        )

    # Init method depends on the used PEFT method, your specific PEFT method
    # might not be considered here, in that case an error is raised.
    if peft_config_cli["init_type"] is not None:
        if peft_method_type == "LORA":
            peft_config_policy["init_lora_weights"] = peft_config_cli["init_type"]
        elif peft_method_type == "MISS":
            peft_config_policy["init_weights"] = peft_config_cli["init_type"]
        else:
            raise ValueError(
                f"Init type {peft_config_cli['init_type']} unknown for PEFT method {peft_method_type}."
            )

    # PEFT uses this attribute to set adapter_config.base_name_or_path which we use for loading the
    # correct base model in `make_policy` since in a PEFT loading setting we only get the path to the
    # adapter, not the base model.
    if policy.config.pretrained_path:
        policy.name_or_path = str(policy.config.pretrained_path)

    # Finally wrap the policy in a PEFT model
    policy = get_peft_model(
        policy,
        peft_config_cls(**peft_config_policy),
    )

    # Make sure that the config is tagged as using PEFT so that the loading code can take the
    # appropriate steps to use the adapter weights and the PEFT config instead of the full model weights.
    policy.config.use_peft = True

    return policy


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when policy.device is set to CPU.
        force_cpu = cfg.policy.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        if is_main_process:
            logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")

    if isinstance(dataset, MultiLeRobotDataset):
        ds_meta = copy.copy(dataset._datasets[0].meta)
        ds_meta.episodes = dataset.meta_episodes
        try:
            ds_meta.stats = dataset.stats['panda']
            print("Not implementing using cross embodiment normalize yet. We use only panda stats")
        except:
            ds_meta.stats = dataset.stats
    else:
        ds_meta = dataset.meta

    visual_robust_policy_excluded_keys = [
        key
        for key in ds_meta.features
        if _is_visual_robust_auxiliary_key(key)
    ]
    if visual_robust_policy_excluded_keys:
        ds_meta.info["features"] = {
            key: value
            for key, value in ds_meta.features.items()
            if key not in visual_robust_policy_excluded_keys
        }
        if hasattr(ds_meta, "stats"):
            ds_meta.stats = {
                key: value for key, value in ds_meta.stats.items() if key not in visual_robust_policy_excluded_keys
            }
        if is_main_process:
            logging.info(
                "Keeping %s visual_robust extra feature keys out of the VLA policy input; "
                "they remain available for contrastive loss.",
                len(visual_robust_policy_excluded_keys),
            )

    if not cfg.dataset.use_wrist_cam:
        wrist_feature_keys = [key for key in ds_meta.features if "wrist" in key]
        ds_meta.info["features"] = {
            key: value for key, value in ds_meta.features.items() if key not in wrist_feature_keys
        }
        if hasattr(ds_meta, "stats"):
            ds_meta.stats = {
                key: value for key, value in ds_meta.stats.items() if key not in wrist_feature_keys
            }
    _ensure_policy_camera_features(cfg, ds_meta, is_main_process)
    dataset.meta = ds_meta
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=ds_meta,
        rename_map=cfg.rename_map,
    )
    policy_image_keys = list(policy.config.image_features)
    if is_main_process:
        logging.info("Policy image features: %s", policy_image_keys)
    if cfg.dataset.use_wrist_cam:
        missing_policy_camera_keys = [
            key for key in POLICY_CAMERA_KEYS if key in ds_meta.features and key not in policy.config.image_features
        ]
        if missing_policy_camera_keys:
            raise ValueError(
                "dataset.use_wrist_cam=true but policy is missing camera input feature(s): "
                f"{missing_policy_camera_keys}. Policy image features: {policy_image_keys}"
            )

    if cfg.peft is not None:
        logging.info("Using PEFT! Wrapping model.")
        policy = wrap_policy_in_peft_model(cfg, policy)

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        # processor_kwargs["dataset_stats"] = dataset.meta.stats
        processor_kwargs["dataset_stats"] = ds_meta.stats

    # For SARM, always provide dataset_meta for progress normalization
    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Load precomputed SARM progress for RA-BC if enabled
    # Generate progress using: src/lerobot/policies/sarm/compute_rabc_weights.py
    rabc_weights = None
    if cfg.use_rabc:
        from lerobot.utils.rabc import RABCWeights

        # Get chunk_size from policy config
        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("Chunk size is not found in policy config")

        head_mode = getattr(cfg, "rabc_head_mode", "sparse")
        logging.info(f"Loading SARM progress for RA-BC from {cfg.rabc_progress_path}")
        logging.info(f"Using chunk_size={chunk_size} from policy config, head_mode={head_mode}")
        rabc_weights = RABCWeights(
            progress_path=cfg.rabc_progress_path,
            chunk_size=chunk_size,
            head_mode=head_mode,
            kappa=getattr(cfg, "rabc_kappa", 0.01),
            epsilon=getattr(cfg, "rabc_epsilon", 1e-6),
            device=device,
        )

    visual_robust_contrastive_weight = cfg.dataset.visual_robust_contrastive_weight
    visual_robust_temperature = cfg.dataset.visual_robust_temperature
    visual_robust_max_views = cfg.dataset.visual_robust_max_views
    visual_robust_wrist_alignment_weight = cfg.dataset.visual_robust_wrist_alignment_weight
    visual_robust_wrist_alignment_mode = cfg.dataset.visual_robust_wrist_alignment_mode
    visual_robust_wrist_alignment_max_views = (
        cfg.dataset.visual_robust_wrist_alignment_max_views
        if cfg.dataset.visual_robust_wrist_alignment_max_views is not None
        else visual_robust_max_views
    )
    visual_robust_wrist_width_bin_size = cfg.dataset.visual_robust_wrist_width_bin_size
    visual_robust_wrist_width_temperature = cfg.dataset.visual_robust_wrist_width_temperature
    visual_robust_wrist_width_min = cfg.dataset.visual_robust_wrist_width_min
    visual_robust_wrist_width_max = cfg.dataset.visual_robust_wrist_width_max
    visual_robust_wrist_width_sigma = cfg.dataset.visual_robust_wrist_width_sigma
    visual_robust_wrist_width_state_key = cfg.dataset.visual_robust_wrist_width_state_key
    visual_robust_wrist_width_left_index = cfg.dataset.visual_robust_wrist_width_left_index
    visual_robust_wrist_width_right_index = cfg.dataset.visual_robust_wrist_width_right_index
    visual_robust_encoder_chunk_size = cfg.dataset.visual_robust_encoder_chunk_size
    visual_robust_dataset_root = cfg.dataset.visual_robust_root
    visual_robust_repo_ids = cfg.dataset.visual_robust_repo_id
    visual_robust_batch_size = cfg.dataset.visual_robust_batch_size or cfg.batch_size
    visual_robust_num_workers = cfg.dataset.visual_robust_num_workers
    visual_robust_cache_in_memory = cfg.dataset.visual_robust_cache_in_memory
    visual_robust_same_episode_negatives = cfg.dataset.visual_robust_same_episode_negatives
    visual_robust_contrastive_enabled = (
        visual_robust_contrastive_weight > 0 or visual_robust_wrist_alignment_weight > 0
    )
    if is_main_process and visual_robust_contrastive_enabled:
        logging.info(
            "Visual robust contrastive learning enabled: front_weight=%s front_temperature=%s "
            "front_max_views=%s wrist_alignment_weight=%s wrist_alignment_mode=%s "
            "wrist_alignment_max_views=%s wrist_width_bin_size=%s wrist_width_temperature=%s "
            "wrist_width_min=%s wrist_width_max=%s wrist_width_sigma=%s "
            "wrist_width_state_key=%s wrist_width_indices=(%s,%s) encoder_chunk_size=%s "
            "dataset_root=%s batch_size=%s same_episode_negatives=%s",
            visual_robust_contrastive_weight,
            visual_robust_temperature,
            visual_robust_max_views,
            visual_robust_wrist_alignment_weight,
            visual_robust_wrist_alignment_mode,
            visual_robust_wrist_alignment_max_views,
            visual_robust_wrist_width_bin_size,
            visual_robust_wrist_width_temperature,
            visual_robust_wrist_width_min,
            visual_robust_wrist_width_max,
            visual_robust_wrist_width_sigma,
            visual_robust_wrist_width_state_key,
            visual_robust_wrist_width_left_index,
            visual_robust_wrist_width_right_index,
            visual_robust_encoder_chunk_size,
            visual_robust_dataset_root,
            visual_robust_batch_size,
            visual_robust_same_episode_negatives,
        )

    visual_robust_loader = None
    visual_robust_iter = None
    if visual_robust_contrastive_enabled:
        if not visual_robust_dataset_root or not visual_robust_repo_ids:
            raise ValueError(
                "Visual robust contrastive learning is enabled, but --dataset.visual_robust_root "
                "or --dataset.visual_robust_repo_id is not set."
            )

        if is_main_process:
            logging.info("Creating visual robust contrastive dataset")
        visual_robust_loader = make_visual_robust_contrastive_loader(
            root=visual_robust_dataset_root,
            repo_ids=visual_robust_repo_ids,
            batch_size=visual_robust_batch_size,
            num_workers=visual_robust_num_workers,
            cache_in_memory=visual_robust_cache_in_memory,
            video_backend=cfg.dataset.video_backend,
            same_episode_negatives=visual_robust_same_episode_negatives,
        )

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        dump_trainable_report(policy, out_path="trainable_report.txt")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=cfg.dataloader_prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.dataloader_persistent_workers if cfg.num_workers > 0 else False,
    )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    if visual_robust_loader is not None:
        visual_robust_loader = accelerator.prepare(visual_robust_loader)
        visual_robust_iter = cycle(visual_robust_loader)
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
        "fetch_s": AverageMeter("fetch_s", ":.3f"),
        "preprocess_s": AverageMeter("prep_s", ":.3f"),
    }

    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    for _ in tqdm(range(step, cfg.steps)):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        visual_robust_batch = next(visual_robust_iter) if visual_robust_iter is not None else None
        fetch_done_time = time.perf_counter()
        batch = preprocessor(batch)
        preprocess_done_time = time.perf_counter()
        train_tracker.fetch_s = fetch_done_time - start_time
        train_tracker.preprocess_s = preprocess_done_time - fetch_done_time
        train_tracker.dataloading_s = preprocess_done_time - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            visual_robust_batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            rabc_weights_provider=rabc_weights,
            visual_robust_contrastive_weight=visual_robust_contrastive_weight,
            visual_robust_temperature=visual_robust_temperature,
            visual_robust_max_views=visual_robust_max_views,
            visual_robust_wrist_alignment_weight=visual_robust_wrist_alignment_weight,
            visual_robust_wrist_alignment_mode=visual_robust_wrist_alignment_mode,
            visual_robust_wrist_alignment_max_views=visual_robust_wrist_alignment_max_views,
            visual_robust_wrist_width_bin_size=visual_robust_wrist_width_bin_size,
            visual_robust_wrist_width_temperature=visual_robust_wrist_width_temperature,
            visual_robust_wrist_width_min=visual_robust_wrist_width_min,
            visual_robust_wrist_width_max=visual_robust_wrist_width_max,
            visual_robust_wrist_width_sigma=visual_robust_wrist_width_sigma,
            visual_robust_wrist_width_state_key=visual_robust_wrist_width_state_key,
            visual_robust_wrist_width_left_index=visual_robust_wrist_width_left_index,
            visual_robust_wrist_width_right_index=visual_robust_wrist_width_right_index,
            visual_robust_encoder_chunk_size=visual_robust_encoder_chunk_size,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            scalar_output_dict = _scalar_metrics(output_dict) if output_dict else {}
            if scalar_output_dict:
                logging.info("extra train metrics: %s", pformat(scalar_output_dict))
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if scalar_output_dict:
                    wandb_log_dict.update(scalar_output_dict)
                # Log RA-BC statistics if enabled
                if rabc_weights is not None:
                    rabc_stats = rabc_weights.get_stats()
                    wandb_log_dict.update(
                        {
                            "rabc_delta_mean": rabc_stats["delta_mean"],
                            "rabc_delta_std": rabc_stats["delta_std"],
                            "rabc_num_frames": rabc_stats["num_frames"],
                        }
                    )
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            if cfg.policy.use_peft:
                unwrapped_policy.push_model_to_hub(cfg, peft_model=unwrapped_policy)
            else:
                unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
