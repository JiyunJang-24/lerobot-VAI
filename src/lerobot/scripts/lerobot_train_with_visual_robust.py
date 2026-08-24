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
import pathlib
import time
import copy
from contextlib import nullcontext
from pprint import pformat
from typing import Any

from tqdm import tqdm

import numpy as np
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
from lerobot.policies.smolvla.vqa_state_text import (
    VQAStateTokenizer,
    build_episode_reference_states,
    lookup_episode_references,
)
from collections import defaultdict
import os
from pathlib import Path


POLICY_CAMERA_KEYS = ("observation.image", "observation.wrist_image")
# fingertip_xyz(3) + quat_xyzw(4) + gripper_close(1), the layout the visual-robust export uses.
EEF_STATE_DIM = 8


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


class VisualRobustProjectionHead(torch.nn.Module):
    """MLP head sitting on top of the VLM connector's output, for the contrastive term only.

    Motivation: with head_mode="none" the contrastive loss is applied straight to the mean-pooled
    vision-backbone output, so all of its invariance pressure lands on the backbone the policy also
    depends on -- push it hard enough and it erases visual detail the policy still needs. The
    SimCLR-style fix is to give the loss its own head to absorb that pressure. Here the head sits
    after the *existing* connector (the adapter whose output is exactly what the VLM consumes), so
    the contrastive term shapes the representation the policy actually reads, one stage removed from
    the backbone.

    Input is the connector output pooled over its tokens, i.e. [B, 960] for SmolVLM2-500M.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        layers: list[torch.nn.Module] = []
        dim = in_dim
        for _ in range(max(num_layers - 1, 0)):
            layers += [torch.nn.Linear(dim, hidden_dim), torch.nn.GELU()]
            dim = hidden_dim
        layers.append(torch.nn.Linear(dim, out_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_visual_robust_head(policy: PreTrainedPolicy, hidden_dim: int, out_dim: int, num_layers: int):
    """Build the head as a STANDALONE module -- deliberately not a submodule of the policy.

    Registering it on the policy looks tidier but breaks under DDP here. The trainer wraps the policy
    with find_unused_parameters=True, and the head is only ever called from the contrastive path,
    never from inside the policy's own forward(). DDP's traversal therefore classes its parameters as
    unused and marks them ready at the start of backward; the real gradient then arrives and marks
    them a second time, which aborts the step with

        Expected to mark a variable ready only once ...
        visual_robust_head.net.4.bias has been marked as ready twice

    Kept separate and passed through accelerator.prepare() on its own, the head gets its own DDP
    wrapper, its forward is a real DDP forward, and gradients sync correctly across ranks. Its
    parameters are added to the optimizer explicitly by the caller.

    Consequence worth knowing: the head is not part of the policy checkpoint. That is fine for
    inference (it is an auxiliary head, discarded) but it does mean a resumed run restarts the head
    from scratch.
    """
    in_dim = policy.model.vlm_with_expert.get_vlm_model().connector.modality_projection.proj.out_features
    head = VisualRobustProjectionHead(in_dim, hidden_dim, out_dim, num_layers)
    head = head.to(dtype=next(policy.parameters()).dtype, device=next(policy.parameters()).device)
    return head, in_dim


def _l2sp_target_module(policy: PreTrainedPolicy, scope: str):
    """The submodule whose drift from its pretrained weights L2-SP penalises."""
    vlm_with_expert = unwrap_model(policy).model.vlm_with_expert
    if scope == "vision":
        return vlm_with_expert.get_vlm_model().vision_model
    if scope == "vlm":
        return vlm_with_expert.vlm
    raise ValueError(f"vision_l2sp_scope must be 'vision' or 'vlm', got {scope!r}")


def load_pretrained_vision_anchor(vlm_model_name: str, device, dtype=torch.float32) -> torch.nn.Module:
    """Load the pretrained SigLIP tower straight from the VLM checkpoint, independent of the policy.

    This is what makes --resume legal for both regularisers. Snapshotting the anchor off the live
    policy only gives the pretrained weights on step 0; on a resumed run it would capture whatever
    the run had already drifted to, silently turning "stay near pretrained" into "stay where you
    happen to be" -- a run that looks perfectly healthy while optimising a different objective.

    Reloading from `vlm_model_name` sidesteps that entirely: the pretrained tower is deterministic,
    so the anchor is identical whether the run starts fresh or resumes at step 30000. Only the vision
    tower is reloaded, so this says nothing about the rest of the VLM.
    """
    from transformers import AutoModelForImageTextToText

    vlm = AutoModelForImageTextToText.from_pretrained(vlm_model_name, dtype=dtype)
    tower = vlm.model.vision_model.to(device=device, dtype=dtype).eval()
    for param in tower.parameters():
        param.requires_grad_(False)
    return tower


def _assert_anchor_matches(anchor: torch.nn.Module, module: torch.nn.Module, what: str) -> None:
    """Fail loudly if the reloaded anchor is not parameter-for-parameter the policy's tower."""
    anchor_shapes = {name: tuple(p.shape) for name, p in anchor.named_parameters()}
    live_shapes = {name: tuple(p.shape) for name, p in module.named_parameters()}
    missing = sorted(set(live_shapes) - set(anchor_shapes))
    mismatched = sorted(n for n in set(live_shapes) & set(anchor_shapes) if live_shapes[n] != anchor_shapes[n])
    if missing or mismatched:
        raise ValueError(
            f"{what} anchor loaded from the pretrained VLM does not match the policy's tower: "
            f"{len(missing)} missing parameter(s) {missing[:5]}, "
            f"{len(mismatched)} shape mismatch(es) {mismatched[:5]}"
        )


def capture_l2sp_reference(policy: PreTrainedPolicy, scope: str) -> tuple[dict[str, torch.Tensor], float]:
    """Snapshot the pretrained weights that training will be pulled back toward.

    Must be called BEFORE the first optimizer step, while the policy still holds exactly what
    `--policy.load_vlm_weights=true` loaded. The caller is responsible for refusing to run when that
    is not what the policy contains (a resumed run, or load_vlm_weights=false), because then this
    snapshot would anchor training to some already-drifted point and the run would look identical
    while regularising toward the wrong target.

    Stored in float32 on the parameters' own device: under mixed precision the parameters themselves
    stay float32, and (w - w0) is a small difference between nearly equal numbers, so keeping the
    reference in bf16 would quantise away most of the signal the penalty is made of. ~535 MB for the
    133.7M-parameter SigLIP tower.

    Returns the snapshot and its squared Frobenius norm, so the per-step diagnostic does not have to
    re-reduce 133.7M constants every step to normalise the drift.
    """
    module = _l2sp_target_module(policy, scope)
    reference = {
        name: param.detach().clone().float()
        for name, param in module.named_parameters()
        if param.requires_grad
    }
    reference_sq_norm = float(sum(tensor.pow(2).sum() for tensor in reference.values()))
    return reference, reference_sq_norm


def compute_l2sp_loss(
    policy: PreTrainedPolicy,
    reference: dict[str, torch.Tensor],
    reference_sq_norm: float,
    scope: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """sum_i (w_i - w0_i)^2 over the scoped parameters, plus drift diagnostics.

    Computed in float32 outside autocast for the same reason the reference is stored in float32.

    The returned loss is a SUM, not a mean: its gradient is 2*(w - w0) per parameter, which is
    independent of how many parameters there are, so the caller's coefficient keeps a fixed meaning
    (a pull-back rate) instead of needing to be rescaled whenever the scope changes. The scalar
    itself is therefore large and hard to read, which is what `relative_drift` -- ||w - w0|| / ||w0||,
    a unitless number -- is for.
    """
    module = _l2sp_target_module(policy, scope)
    device_type = next(module.parameters()).device.type
    with torch.autocast(device_type=device_type, enabled=False):
        terms = [
            (param.float() - reference[name]).pow(2).sum()
            for name, param in module.named_parameters()
            if name in reference
        ]
        squared_drift = torch.stack(terms).sum() if terms else None

    if squared_drift is None:
        return None, {}

    drift = float(squared_drift.detach())
    metrics = {
        "vision_l2sp_loss": drift,
        "vision_l2sp_relative_drift": (drift / reference_sq_norm) ** 0.5 if reference_sq_norm > 0 else 0.0,
    }
    return squared_drift, metrics


class VisualRobustStateHead(torch.nn.Module):
    """Regresses the EEF state from the vision tower's token grid.

    pool="mean" averages the 1024 patch tokens and runs an MLP on the result. It is the cheap option
    and the one that matches every other auxiliary head here, but it is a poor fit for this
    particular target: two of the eight dimensions the head must predict are a 3D POSITION, and a
    mean over all tokens is close to position-blind. (Not entirely -- SigLIP tokens carry positional
    embeddings, so the mean does shift with the gripper -- but the signal is indirect.)

    pool="attn" adds a learned attention pooling alongside the mean: a linear scorer picks which
    tokens matter and the output is their weighted sum. Because the tokens are position-tagged, "which
    tokens does the model attend to" IS the localisation, so this gives the head a direct route to
    the fingertip's whereabouts instead of an indirect one. The mean is concatenated rather than
    replaced so nothing that already worked is lost.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, pool: str = "mean"):
        super().__init__()
        if pool not in ("mean", "attn"):
            raise ValueError(f"pool must be 'mean' or 'attn', got {pool!r}")
        self.pool = pool
        self.score = torch.nn.Linear(in_dim, 1) if pool == "attn" else None
        mlp_in = in_dim * 2 if pool == "attn" else in_dim

        layers: list[torch.nn.Module] = []
        dim = mlp_in
        for _ in range(max(num_layers - 1, 0)):
            layers += [torch.nn.Linear(dim, hidden_dim), torch.nn.GELU()]
            dim = hidden_dim
        layers.append(torch.nn.Linear(dim, out_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [N, L, D] -- the grid, not a pooled vector."""
        pooled = tokens.mean(dim=1)
        if self.pool == "attn":
            weights = torch.softmax(self.score(tokens), dim=1)
            pooled = torch.cat([pooled, (weights * tokens).sum(dim=1)], dim=-1)
        return self.net(pooled)



def build_episode_position_offsets(dataset, state_key: str, pos_slice: slice):
    """Per-episode mean of the position dims, so the regression target can be made robot-relative.

    Measured on the export: the fingertip moves ~0.09-0.15 m in x within an episode, but the episode
    MEAN of x shifts 1.4-2.1 m between episodes (between/within ratio 13-15x) because robocasa parks
    the robot base somewhere different each time. `agentview_right` is robot-mounted, so the image
    looks the same wherever the base is standing -- world-frame position is genuinely not inferable
    from it, and a head asked to predict it can do no better than guess the mean. The first probe did
    exactly that: pos_err plateaued at 1.57 m, which is the between-episode std.

    Subtracting the episode mean turns the target into "where is the fingertip relative to this
    arm's working centre", which is what the camera actually shows. The invariance mechanism is
    untouched: all three renders of a frame belong to the same episode, so they still share one
    target.

    Note this offset is computed over the whole episode, i.e. it peeks at future frames. That is fine
    for a representation-shaping auxiliary term -- nothing here is used at inference -- but it would
    not be acceptable for a predictive target.
    """
    import pandas as pd

    offsets = {}
    subs = getattr(dataset, "_datasets", [dataset])
    for dataset_index, sub in enumerate(subs):
        root = pathlib.Path(sub.root)
        files = sorted(root.glob("data/**/*.parquet"))
        if not files:
            continue
        frame = pd.concat(
            [pd.read_parquet(f, columns=["episode_index", state_key]) for f in files], ignore_index=True
        )
        states = np.stack(frame[state_key].to_numpy())[:, pos_slice]
        episodes = frame["episode_index"].to_numpy()
        for episode in np.unique(episodes):
            offsets[(dataset_index, int(episode))] = torch.as_tensor(
                states[episodes == episode].mean(axis=0), dtype=torch.float32
            )
    if not offsets:
        raise ValueError(f"could not build episode offsets for {state_key}")
    return offsets


def lookup_episode_offsets(offsets, batch, pos_dim: int, device) -> torch.Tensor:
    """Gather the per-episode position offset for each row of the auxiliary batch."""
    episode_index = batch["episode_index"]
    episode_index = episode_index[:, -1] if episode_index.ndim == 2 else episode_index
    dataset_index = batch.get("dataset_index")
    if dataset_index is None:
        dataset_index = torch.zeros_like(episode_index)
    dataset_index = dataset_index[:, -1] if dataset_index.ndim == 2 else dataset_index

    zero = torch.zeros(pos_dim, dtype=torch.float32)
    rows = [
        offsets.get((int(d), int(e)), zero)
        for d, e in zip(dataset_index.tolist(), episode_index.tolist(), strict=False)
    ]
    return torch.stack(rows).to(device)


def canonicalize_quaternion_xyzw(state: torch.Tensor, quat_slice: slice) -> torch.Tensor:
    """Force w >= 0 so a rotation has exactly one representation.

    q and -q are the same rotation, and the export contains both signs (w spans about -0.50..0.97 in
    every tree). Regressing the raw values would ask the head to predict two different targets for
    identical images, putting a noise floor on the loss that no amount of training removes.
    """
    state = state.clone()
    quaternion = state[..., quat_slice]
    flip = (quaternion[..., 3:4] < 0).to(quaternion.dtype) * -2.0 + 1.0
    state[..., quat_slice] = quaternion * flip
    return state


def _mean_within_episode_std(dataset, state_key: str, pos_slice: slice):
    """Average of each episode's own positional std -- the scale of a per-episode-centred target.

    Computed directly rather than as sqrt(pooled^2 - between^2): that identity is exact in theory
    but cancels catastrophically when the between-episode term carries almost all the variance,
    which is the normal case for a single-tree export.
    """
    import pathlib

    import pandas as pd

    stds = []
    for sub in getattr(dataset, "_datasets", [dataset]):
        files = sorted(pathlib.Path(sub.root).glob("data/**/*.parquet"))
        if not files:
            continue
        frame = pd.concat(
            [pd.read_parquet(f, columns=["episode_index", state_key]) for f in files], ignore_index=True
        )
        states = np.stack(frame[state_key].to_numpy())[:, pos_slice]
        episodes = frame["episode_index"].to_numpy()
        for episode in np.unique(episodes):
            rows = states[episodes == episode]
            if len(rows) > 1:
                stds.append(rows.std(axis=0))
    if not stds:
        return None
    return torch.as_tensor(np.mean(np.stack(stds), axis=0), dtype=torch.float32)


def compute_eef_state_normalizer(
    dataset, state_key: str, quat_slice: slice, std_floor: float = 1e-3, episode_offsets=None,
    pos_slice: slice = slice(0, 3)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pooled mean/std of the auxiliary EEF state, with constant dimensions neutralised.

    The std floor is not cosmetic. PandaOmron_TurnOnSinkFaucet never opens its gripper, so that
    dimension has std exactly 0 in that tree; z-scoring it with the usual `std + 1e-8` would map its
    constant value to |z| ~ 1e8 and the auxiliary loss would be nothing but that one dead dimension.
    This is the same trap that put 36% of the action MSE into panda_human's frozen base-motion dims
    earlier in this project, so dimensions whose spread is below the floor are given std 1 and a zero
    weight instead -- they are excluded from the loss rather than silently dominating it.

    Returns (mean, std, dim_weight) where dim_weight is 0 for excluded dimensions and 1 elsewhere.
    """
    states = []
    for sub in getattr(dataset, "_datasets", [dataset]):
        stats = getattr(sub, "meta", None)
        stats = getattr(stats, "stats", None)
        if not stats or state_key not in stats:
            continue
        entry = stats[state_key]
        mean = torch.as_tensor(entry["mean"], dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(entry["std"], dtype=torch.float32).reshape(-1)
        states.append((mean, std))
    if not states:
        raise ValueError(f"no {state_key} stats found on the visual-robust dataset")

    mean = torch.stack([m for m, _ in states]).mean(dim=0)
    # Pool the spread across trees: within-tree variance plus the spread of the tree means, which is
    # what a single normaliser over the concatenated data would see.
    within = torch.stack([s for _, s in states]).pow(2).mean(dim=0)
    between = torch.stack([m for m, _ in states]).var(dim=0, unbiased=False)
    std = (within + between).sqrt()

    dim_weight = (std > std_floor).to(torch.float32)
    std = torch.where(std > std_floor, std, torch.ones_like(std))
    # The quaternion is already unit-norm and sign-canonicalised, so leave its scale alone; z-scoring
    # per component would distort the rotation geometry.
    mean[quat_slice] = 0.0
    std[quat_slice] = 1.0

    # With per-episode centering the position target is a deviation, so its scale is the WITHIN-episode
    # spread, not the pooled one. Using the pooled std here would divide a ~0.1 m signal by ~1.7 m and
    # hand the head a target that is essentially zero.
    if episode_offsets is not None:
        centered = torch.stack(list(episode_offsets.values()))
        mean[pos_slice] = 0.0
        # sqrt(pooled^2 - between^2) is a law-of-total-variance estimate of the within-episode
        # spread, and it degenerates when the two terms are nearly equal -- which is exactly what
        # happens on a single-tree export, where the pooled spread IS the between-episode spread.
        # Measured on the 6-embodiment UR5e tree: pooled x = 1.648, between x = 1.650, so the
        # difference goes negative, clamps to the floor, and the x target ends up divided by 0.001.
        # The EEF loss then reads ~3900 instead of ~2 and swamps everything else in the sum.
        # Averaging each episode's own std has no cancellation in it, so it stays correct whether
        # the normalizer sees one tree or several.
        within = _mean_within_episode_std(dataset, state_key, pos_slice)
        if within is None:
            within = (std[pos_slice].pow(2) - centered.std(dim=0).pow(2)).clamp(min=std_floor**2).sqrt()
        std[pos_slice] = within.clamp(min=std_floor)
    return mean, std, dim_weight


def compute_visual_robust_state_loss(
    policy: PreTrainedPolicy,
    batch,
    accelerator: Accelerator,
    head: torch.nn.Module,
    normalizer: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    episode_offsets=None,
    state_key: str = "observation.state",
    quat_slice: slice = slice(3, 7),
    pos_slice: slice = slice(0, 3),
    grip_slice: slice = slice(7, 8),
    rotation_weight: float = 1.0,
    gripper_weight: float = 1.0,
    max_views: int | None = None,
    random_views: bool = False,
    prefixes: tuple[str, ...] = ("observation.image.",),
    encoder_chunk_size: int = 32,
    head_mode: str = "none",
    projection_head: torch.nn.Module | None = None,
    freeze_backbone: bool = False,
    include_views=None,
):
    """Regress the end-effector state from each embodiment's render of the same frame.

    The auxiliary export stores ONE observation.state per frame and several renders of it, one per
    robot, so every view of a frame carries the identical target. Asking the head to produce that one
    target from three visually different robots is a supervised route to embodiment invariance: the
    only way to win is to locate the fingertip regardless of which arm is holding it.

    Why this is not just the contrastive loss with extra steps: contrastive says "these views should
    be near each other" and is satisfied by any embodiment-invariant code, including a degenerate one
    that throws the scene away. This says "these views should all decode to THIS pose", which pins
    the invariant down to something physical and keeps it informative.
    """
    unwrapped_policy = _get_unwrapped_policy(policy, accelerator)
    if unwrapped_policy.config.type != "smolvla" or state_key not in batch:
        return None, {}

    mean, std, dim_weight = normalizer
    device = accelerator.device
    mean, std, dim_weight = mean.to(device), std.to(device), dim_weight.to(device)

    keys = _select_visual_robust_image_keys(
        batch, image_prefix=prefixes[0], max_views=max_views, random_views=random_views,
        include_views=include_views,
    )
    if not keys:
        return None, {}

    views = _prepare_visual_robust_images(unwrapped_policy, batch, keys, device)
    batch_size, num_views = views.shape[:2]

    tokens = _encode_flat_visual_robust(
        unwrapped_policy,
        views.flatten(0, 1),
        encoder_chunk_size,
        head_mode,
        projection_head,
        freeze_backbone,
        return_tokens=True,
    )
    return visual_robust_state_loss_from_tokens(
        tokens=tokens,
        batch=batch,
        head=head,
        normalizer=(mean, std, dim_weight),
        device=device,
        num_views=num_views,
        episode_offsets=episode_offsets,
        state_key=state_key,
        quat_slice=quat_slice,
        pos_slice=pos_slice,
        grip_slice=grip_slice,
        rotation_weight=rotation_weight,
        gripper_weight=gripper_weight,
    )


def visual_robust_state_loss_from_tokens(
    *,
    tokens: torch.Tensor,
    batch,
    head: torch.nn.Module,
    normalizer: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device,
    num_views: int,
    episode_offsets=None,
    state_key: str = "observation.state",
    quat_slice: slice = slice(3, 7),
    pos_slice: slice = slice(0, 3),
    grip_slice: slice = slice(7, 8),
    rotation_weight: float = 1.0,
    gripper_weight: float = 1.0,
):
    """The EEF-state loss itself, given an already-encoded token grid.

    Split out of compute_visual_robust_state_loss so the standalone encoder pre-training script
    (src/lerobot/scripts/pretrain_siglip_visual_robust.py) runs the identical objective without
    having to build a policy around the tower. Every subtlety this loss encodes -- per-episode
    position centering, the sign-invariant quaternion term, the zero-weighted constant dimensions --
    is worth exactly one implementation.
    """
    mean, std, dim_weight = normalizer
    mean, std, dim_weight = mean.to(device), std.to(device), dim_weight.to(device)
    batch_size = tokens.shape[0] // num_views

    predictions = head(tokens)

    target = batch[state_key]
    target = target[:, -1] if target.ndim == 3 else target
    target = target.to(device=device, dtype=torch.float32)

    # Position is made robot-relative; see build_episode_position_offsets for why world frame is
    # unlearnable from a robot-mounted camera.
    position_target = target[:, pos_slice]
    if episode_offsets is not None:
        position_target = position_target - lookup_episode_offsets(
            episode_offsets, batch, position_target.shape[-1], device
        )
    normalized_position = (position_target - mean[pos_slice]) / std[pos_slice]
    gripper_target = (target[:, grip_slice] - mean[grip_slice]) / std[grip_slice]
    quaternion_target = F.normalize(target[:, quat_slice], dim=-1)

    # Every view of a frame gets the same target -- this repeat IS the invariance pressure.
    normalized_position = normalized_position.repeat_interleave(num_views, dim=0)
    gripper_target = gripper_target.repeat_interleave(num_views, dim=0)
    quaternion_target = quaternion_target.repeat_interleave(num_views, dim=0)

    device_type = device.type
    with torch.autocast(device_type=device_type, enabled=False):
        predictions = predictions.float()
        position_pred = predictions[:, pos_slice]
        quaternion_pred = F.normalize(predictions[:, quat_slice], dim=-1)
        gripper_pred = predictions[:, grip_slice]

        position_loss = (position_pred - normalized_position).pow(2).mean()
        gripper_loss = (gripper_pred - gripper_target).pow(2).mean()
        # Sign-invariant geodesic term. NOT an MSE on the quaternion: q and -q are the same rotation,
        # and 64% of this export's frames sit at |w| < 0.05 -- right on the w=0 boundary -- so any
        # sign convention splits near-identical rotations onto opposite targets. Taking |cos| removes
        # the double cover instead of trying to pick a side.
        cos_half = (quaternion_pred * quaternion_target).sum(-1).abs().clamp(max=1.0)
        rotation_loss = 1.0 - cos_half.mean()

        loss = position_loss + rotation_weight * rotation_loss + gripper_weight * gripper_loss

        # Diagnostics in physical units.
        unnormalized_position = position_pred * std[pos_slice] + mean[pos_slice]
        position_error = (unnormalized_position - (
            position_target.repeat_interleave(num_views, 0) )).norm(dim=-1)
        rotation_error = torch.rad2deg(2.0 * torch.arccos(cos_half))
        # How much the three robots' predictions disagree about the same frame -- the number this
        # loss exists to drive down.
        grouped = unnormalized_position.view(batch_size, num_views, -1)
        view_spread = grouped.std(dim=1).mean() if num_views > 1 else torch.zeros((), device=device)

    metrics = {
        "visual_robust_state_loss": float(loss.detach()),
        "visual_robust_state_pos_err_m": float(position_error.mean().detach()),
        "visual_robust_state_rot_err_deg": float(rotation_error.mean().detach()),
        "visual_robust_state_view_spread_m": float(view_spread.detach()),
        "visual_robust_state_views": float(num_views),
        "visual_robust_state_pos_loss": float(position_loss.detach()),
        "visual_robust_state_rot_loss": float(rotation_loss.detach()),
        "visual_robust_state_grip_loss": float(gripper_loss.detach()),
    }
    return loss, metrics



def compute_policy_state_loss(
    tokens: torch.Tensor,
    batch,
    head: torch.nn.Module,
    normalizer,
    accelerator: Accelerator,
    state_key: str = "observation.state",
    pos_start: int = 7,
    quat_start: int = 10,
    grip_index: int = 14,
    rotation_weight: float = 1.0,
    gripper_weight: float = 1.0,
):
    """The EEF-state head applied to the POLICY batch, reusing the tokens its forward already made.

    No second encode: `tokens` comes from the forward hook, so this term's only cost is the head.

    There is no cross-embodiment pair here -- the policy batch has one render per frame -- so this
    contributes grounding, not invariance. It is the auxiliary batch that supplies the invariance
    pressure; this makes the same head answer the same question on 9x more data so the feature it
    reads has to encode arm pose generally rather than for 324 episodes' worth of scenes.

    Slices default to the layout confirmed on this corpus: base(0:7), eef(7:14), gripper(14:16).
    """
    if state_key not in batch:
        return None, {}
    mean, std = normalizer
    device = tokens.device
    mean, std = mean.to(device), std.to(device)

    target = batch[state_key]
    target = target[:, -1] if target.ndim == 3 else target
    target = target.to(device=device, dtype=torch.float32)
    if target.shape[-1] < grip_index + 1:
        return None, {}

    predictions = head(tokens)
    pos_slice = slice(pos_start, pos_start + 3)
    quat_slice = slice(quat_start, quat_start + 4)

    with torch.autocast(device_type=device.type, enabled=False):
        predictions = predictions.float()
        position_target = (target[:, pos_slice] - mean[:3]) / std[:3]
        gripper_target = (target[:, grip_index : grip_index + 1] - mean[7:8]) / std[7:8]
        quaternion_target = F.normalize(target[:, quat_slice], dim=-1)

        position_loss = (predictions[:, :3] - position_target).pow(2).mean()
        gripper_loss = (predictions[:, 7:8] - gripper_target).pow(2).mean()
        # Sign-invariant, for the same double-cover reason as the auxiliary branch.
        cos_half = (F.normalize(predictions[:, 3:7], dim=-1) * quaternion_target).sum(-1).abs().clamp(max=1.0)
        rotation_loss = 1.0 - cos_half.mean()
        loss = position_loss + rotation_weight * rotation_loss + gripper_weight * gripper_loss

        position_error = ((predictions[:, :3] * std[:3] + mean[:3]) - target[:, pos_slice]).norm(dim=-1)
        rotation_error = torch.rad2deg(2.0 * torch.arccos(cos_half))

    metrics = {
        "policy_state_loss": float(loss.detach()),
        "policy_state_pos_err_m": float(position_error.mean().detach()),
        "policy_state_rot_err_deg": float(rotation_error.mean().detach()),
    }
    return loss, metrics


def build_policy_state_normalizer(dataset, state_key: str, pos_start: int, grip_index: int, std_floor: float = 1e-3):
    """mean/std for the policy corpus's eef pose + gripper, in the head's 8-dim ordering."""
    means, stds = [], []
    for sub in getattr(dataset, "_datasets", [dataset]):
        stats = getattr(getattr(sub, "meta", None), "stats", None)
        if not stats or state_key not in stats:
            continue
        means.append(torch.as_tensor(stats[state_key]["mean"], dtype=torch.float32).reshape(-1))
        stds.append(torch.as_tensor(stats[state_key]["std"], dtype=torch.float32).reshape(-1))
    if not means:
        raise ValueError(f"no {state_key} stats on the policy dataset")
    mean_all = torch.stack(means).mean(0)
    std_all = (torch.stack(stds).pow(2).mean(0) + torch.stack(means).var(0, unbiased=False)).sqrt()

    mean = torch.zeros(EEF_STATE_DIM)
    std = torch.ones(EEF_STATE_DIM)
    mean[:3] = mean_all[pos_start : pos_start + 3]
    std[:3] = std_all[pos_start : pos_start + 3].clamp(min=std_floor)
    mean[7] = mean_all[grip_index]
    std[7] = std_all[grip_index].clamp(min=std_floor)
    return mean, std


def make_frozen_vision_teacher(policy: PreTrainedPolicy, scope: str = "vision") -> torch.nn.Module:
    """A frozen copy of the pretrained vision tower, used as a distillation target.

    Same anchor as L2-SP but a different place to apply it: L2-SP constrains the WEIGHTS, this
    constrains what the tower OUTPUTS on the actual training images. Weights can move a long way
    while the function they compute on this data barely changes (and vice versa), so the two are not
    substitutes.

    Costs one extra no-grad forward of an 86M-parameter tower per step, plus 0.32 GiB for its
    parameters. The student side is free -- see _StudentFeatureCapture.
    """
    teacher = copy.deepcopy(_l2sp_target_module(policy, scope))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


class _StudentFeatureCapture:
    """Steals the trainable tower's output from inside policy.forward() via a forward hook.

    The alternative -- running the student tower a second time on the same images -- would double the
    vision forward AND keep a second copy of its activations for backward, for features the policy
    already computed. The hook keeps the tensor that is already part of the autograd graph, so the
    distillation gradient flows back through the very same activations the policy loss uses, at no
    extra compute.

    `captured` is a list because the tower is called once per camera; every call is distilled.
    """

    def __init__(self, vision_model: torch.nn.Module):
        self.captured: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.enabled = False
        self._handle = vision_model.register_forward_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs, output):
        if not self.enabled:
            return
        pixel_values = kwargs.get("pixel_values")
        if pixel_values is None and args:
            pixel_values = args[0]
        self.captured.append((pixel_values, output.last_hidden_state))

    def arm(self):
        self.captured.clear()
        self.enabled = True

    def disarm(self):
        self.enabled = False

    def remove(self):
        self._handle.remove()


def compute_vision_distill_loss(
    teacher: torch.nn.Module, capture: _StudentFeatureCapture, accelerator: Accelerator
):
    """Mean squared error between the trainable tower's patch tokens and the frozen tower's.

    Deliberately per-TOKEN, not on the mean-pooled feature. Pooling first leaves 1024 tokens' worth of
    structure unconstrained, which is exactly how the alignment objective ended up satisfying itself
    (mean-pooled cosine driven to 0.9999) while the policy's action loss stayed at the baseline's
    0.069 -- the policy consumes the tokens, not their mean, so a pooled constraint never touched
    what it reads.

    The difference is taken in float32 outside autocast: the whole quantity is a small residual
    between two nearly identical activations, and bf16 rounding eats residuals like that.
    """
    if not capture.captured:
        return None, {}

    device_type = next(teacher.parameters()).device.type
    terms = []
    squared_error = 0.0
    squared_target = 0.0
    for pixel_values, student_features in capture.captured:
        with torch.no_grad(), accelerator.autocast():
            teacher_features = teacher(
                pixel_values=pixel_values.to(dtype=next(teacher.parameters()).dtype),
                patch_attention_mask=None,
            ).last_hidden_state
        with torch.autocast(device_type=device_type, enabled=False):
            student = student_features.float()
            target = teacher_features.float()
            difference = student - target
            terms.append(difference.pow(2).mean())
            squared_error += float(difference.pow(2).sum())
            squared_target += float(target.pow(2).sum())

    loss = torch.stack(terms).mean()
    metrics = {
        "vision_distill_loss": float(loss.detach()),
        # ||student - teacher|| / ||teacher||: unitless, so it stays readable whatever the weight is.
        "vision_distill_relative_error": (squared_error / squared_target) ** 0.5 if squared_target else 0.0,
        "vision_distill_views": len(capture.captured),
    }
    return loss, metrics


def _select_visual_robust_image_keys(
    batch, image_prefix: str, max_views=None, random_views: bool = False, include_views=None
):
    """Pick which auxiliary camera keys become the positive group for this step.

    `random_views` matters as soon as the dataset offers more views than max_views can afford.
    Truncating the *sorted* list takes a contiguous alphabetical block, and these keys sort by
    embodiment first: with the background-variation export
    (observation.image.<Emb>.<background>, 3 embodiments x 4 backgrounds = 12 keys), max_views=4
    yields IIWAOmron.{dark,plain,warm,white} -- one single embodiment. The contrastive term would
    then only ever teach background invariance for IIWA and never see a cross-embodiment positive
    pair, silently defeating the whole point of the loss.

    Sampling instead draws a fresh subset every step, so over training every view is used and every
    pair co-occurs, at unchanged memory cost. Sampling is uniform without replacement and makes no
    assumption about how the keys are named; for 4-of-12 the chance of landing on a single-embodiment
    group is 3/495 (~0.6%).
    """
    image_keys = sorted(
        key
        for key, value in batch.items()
        if key.startswith(image_prefix)
        and torch.is_tensor(value)
        and value.ndim >= 4
        and value.shape[-3] == 3
    )

    if include_views:
        # Naming an explicit subset, e.g. "UR5eOmron,PandaOmronPandaGripper,JacoOmron,
        # JacoOmronPandaGripper". Matching is by substring on the key, so it works whatever the
        # export calls its cameras. An unmatched name is an error rather than a silent drop: a
        # typo would otherwise quietly shrink the positive group and look like a weaker result.
        selected = [k for k in image_keys if any(v in k for v in include_views)]
        missing = [v for v in include_views if not any(v in k for k in image_keys)]
        if missing:
            raise ValueError(
                f"visual_robust_include_views named {missing}, which match none of {image_keys}"
            )
        image_keys = selected

    if max_views is not None and max_views > 0 and len(image_keys) > max_views:
        if random_views:
            picked = torch.randperm(len(image_keys))[:max_views].tolist()
            image_keys = [image_keys[i] for i in sorted(picked)]
        else:
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
    """1 - mean cosine similarity within each frame's view group.

    The similarity matmul is forced to float32. Casting the input with .float() is not enough: under
    accelerate's autocast the matmul is promoted straight back to bfloat16, and bfloat16's spacing
    near 1.0 is 2^-8 = 0.0039. This loss lives precisely in the cos > 0.99 regime, so every
    similarity rounds to exactly 1.0, the loss reads exactly 0.0 and -- worse -- its gradient is
    zero everywhere, i.e. the term silently stops training anything. Measured: at cos = 0.99878 the
    float32 value is 1.2e-3 while bfloat16 gives 0.0.
    """
    if batch_size < 1 or num_views < 2:
        return features.float().new_zeros(())

    device_type = features.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        normalized = F.normalize(features.float(), dim=-1)
        grouped_features = normalized.view(batch_size, num_views, -1)
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
    random_views: bool = False,
    head_mode: str = "none",
) -> tuple[torch.Tensor | None, int, int]:
    """Encode the auxiliary views into the vectors the contrastive loss compares.

    head_mode="none" (original): backbone -> mean over its 1024 patch tokens. The loss lands
    directly on the backbone, which is also what the policy uses.

    head_mode="adapter_mlp": backbone (NO GRAD) -> the VLM's own connector -> mean over its 64
    output tokens -> MLP head. The backbone is frozen *for this loss only* -- torch.no_grad() around
    it stops the contrastive gradient there while the policy's own forward pass, which runs
    separately, still trains it normally. So the contrastive term only updates the connector and the
    head, and the representation it shapes is exactly the one handed to the VLM.
    """
    unwrapped_policy = _get_unwrapped_policy(policy, accelerator)
    if unwrapped_policy.config.type != "smolvla":
        return None, 0, 0

    image_keys = _select_visual_robust_image_keys(
        batch, image_prefix=image_prefix, max_views=max_views, random_views=random_views
    )
    if len(image_keys) < 2:
        return None, len(image_keys), 0

    views = _prepare_visual_robust_images(unwrapped_policy, batch, image_keys, accelerator.device)
    batch_size, num_views = views.shape[:2]
    flat_images = views.flatten(0, 1)

    vlm = unwrapped_policy.model.vlm_with_expert.get_vlm_model()
    vision_model = vlm.vision_model
    freeze_backbone = head_mode == "adapter_mlp"

    vision_feature_chunks = []
    for image_chunk in flat_images.split(encoder_chunk_size, dim=0):
        with torch.no_grad() if freeze_backbone else nullcontext():
            hidden = vision_model(
                pixel_values=image_chunk.to(dtype=vision_model.dtype),
                patch_attention_mask=None,
            ).last_hidden_state
        vision_feature_chunks.append(hidden)
    vision_features = torch.cat(vision_feature_chunks, dim=0)

    if not freeze_backbone:
        return vision_features.mean(dim=1), batch_size, num_views

    raise RuntimeError(
        "head_mode='adapter_mlp' is not supported by this single-prefix helper -- the head is a "
        "standalone module now (see make_visual_robust_head) and is not reachable from the policy. "
        "Use compute_visual_robust_contrastive_loss_multi(), which the training loop calls."
    )
    # Unreachable; kept so the original single-prefix path below stays readable.
    adapted = vlm.connector(vision_features.detach())
    return head(adapted.mean(dim=1)), batch_size, num_views


def _encode_flat_visual_robust(
    unwrapped_policy: PreTrainedPolicy,
    flat_images: torch.Tensor,
    encoder_chunk_size: int,
    head_mode: str,
    head: torch.nn.Module | None = None,
    freeze_backbone: bool = True,
    return_tokens: bool = False,
) -> torch.Tensor:
    """Run [N, C, H, W] auxiliary images through the encoder stack once. See
    _encode_visual_robust_features for what each head_mode does.

    return_tokens keeps the [N, L, D] token grid instead of collapsing it. Needed by the
    EEF-state head: regressing a fingertip POSITION from a mean over 1024 tokens throws away
    most of the where-information the task is asking for."""
    vlm = unwrapped_policy.model.vlm_with_expert.get_vlm_model()
    vision_model = vlm.vision_model
    use_head = head_mode == "adapter_mlp"
    # head_mode="none" contrasts the backbone output directly; freezing it there would leave the
    # loss with nothing to train, so the flag only applies when there is a head downstream.
    freeze_backbone = use_head and freeze_backbone

    chunks = []
    for image_chunk in flat_images.split(encoder_chunk_size, dim=0):
        with torch.no_grad() if freeze_backbone else nullcontext():
            chunks.append(
                vision_model(
                    pixel_values=image_chunk.to(dtype=vision_model.dtype),
                    patch_attention_mask=None,
                ).last_hidden_state
            )
    vision_features = torch.cat(chunks, dim=0)

    if not use_head:
        return vision_features if return_tokens else vision_features.mean(dim=1)

    if head is None:
        raise RuntimeError(
            "visual_robust_head_mode='adapter_mlp' but no head was passed in. "
            "make_visual_robust_head() must run before the optimizer is built."
        )
    # detach() only when the backbone is meant to be frozen; otherwise the gradient must reach it.
    features = vision_features.detach() if freeze_backbone else vision_features
    projected = vlm.connector(features)
    return projected if return_tokens else head(projected.mean(dim=1))


def compute_visual_robust_vqa_loss(
    policy: PreTrainedPolicy,
    batch,
    accelerator: Accelerator,
    tokenizer,
    references,
    *,
    prefixes: tuple[str, ...] = ("observation.image.",),
    max_views: int | None = None,
    random_views: bool = False,
    include_views=None,
    max_frames: int | None = None,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Ask the VLM where the gripper is, once per embodiment render, and score the answer.

    Every render of a frame gets the SAME answer, so the only way to be right for all of them is to
    locate the gripper regardless of which arm is holding it -- the same mechanism as the
    EEF-state head, but routed through the LM head so the gradient reaches the whole VLM.

    Each view becomes its own sample rather than being concatenated into one long sequence: the
    question is about one picture, and stacking six robots into a single context would let the model
    answer by averaging them instead of reading each one.
    """
    unwrapped_policy = _get_unwrapped_policy(policy, accelerator)
    if unwrapped_policy.config.type != "smolvla" or "observation.state" not in batch:
        return None, {}

    keys = _select_visual_robust_image_keys(
        batch, image_prefix=prefixes[0], max_views=max_views, random_views=random_views,
        include_views=include_views,
    )
    if not keys:
        return None, {"visual_robust_vqa_views": 0.0}

    device = accelerator.device
    views = _prepare_visual_robust_images(unwrapped_policy, batch, keys, device)  # [B, V, C, H, W]
    batch_size, num_views = views.shape[:2]
    if max_frames is not None and batch_size > max_frames:
        views = views[:max_frames]
        batch_size = max_frames

    state = batch["observation.state"]
    state = state[:, -1] if state.ndim == 3 else state
    state = state[:batch_size].to(device=device, dtype=torch.float32)
    reference = lookup_episode_references(references, batch, device, state.shape[-1])[:batch_size]

    vqa = tokenizer.encode(state, reference)
    # One sample per (frame, view): repeat the labels across views, flatten the views into the batch.
    flat_views = views.flatten(0, 1)
    repeat = lambda t: t.repeat_interleave(num_views, dim=0)  # noqa: E731
    vqa_flat = {
        "question_tokens": repeat(vqa["question_tokens"]),
        "question_masks": repeat(vqa["question_masks"]),
        "tokens": repeat(vqa["tokens"]),
        "pad_masks": repeat(vqa["pad_masks"]),
        "loss_masks": repeat(vqa["loss_masks"]),
        "content_masks": repeat(vqa["content_masks"]),
        "stats": vqa["stats"],
    }
    img_masks = [torch.ones(flat_views.shape[0], dtype=torch.bool, device=device)]

    out = unwrapped_policy.model.vqa_state_loss([flat_views], img_masks, vqa_flat)
    loss = out.pop("token_ce_loss")
    metrics = {
        "visual_robust_vqa_loss": float(loss.detach()),
        "visual_robust_vqa_accuracy": out.get("token_accuracy", 0.0),
        "visual_robust_vqa_content_accuracy": out.get("token_content_accuracy", 0.0),
        "visual_robust_vqa_views": float(num_views),
        "visual_robust_vqa_frames": float(batch_size),
        "visual_robust_vqa_samples": float(flat_views.shape[0]),
        "visual_robust_vqa_answer_tokens": out.get("vqa_answer_tokens_mean", 0.0),
    }
    return loss, metrics


def compute_visual_robust_contrastive_loss_multi(
    policy: PreTrainedPolicy,
    batch: dict[str, Any],
    accelerator: Accelerator,
    temperature: float,
    max_views: int | None,
    encoder_chunk_size: int,
    prefixes: tuple[str, ...],
    random_views: bool = False,
    head_mode: str = "none",
    head: torch.nn.Module | None = None,
    freeze_backbone: bool = True,
    objective: str = "contrastive",
    include_views=None,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Visual-robust loss over several view groups (one per prefix), averaged.

    All groups are encoded in a SINGLE pass rather than one call per prefix. That is not just a
    speed choice: calling the shared connector/head once per group makes their parameters receive
    two separate gradient-ready events in one backward, and DDP's reducer rejects that with
    "Expected to mark a variable ready only once ... visual_robust_head.net.4.bias has been marked
    as ready twice". Encoding the concatenated batch once and slicing the features back out keeps
    every parameter used exactly once per step, so no DDP static-graph workaround is needed.
    """
    unwrapped_policy = _get_unwrapped_policy(policy, accelerator)
    if unwrapped_policy.config.type != "smolvla":
        return None, {}

    groups = []
    for prefix in prefixes:
        keys = _select_visual_robust_image_keys(
            batch, image_prefix=prefix, max_views=max_views, random_views=random_views,
            include_views=include_views,
        )
        if len(keys) < 2:
            continue
        tag = prefix.rstrip(".").rsplit(".", 1)[-1] if prefix != "observation.image." else ""
        groups.append((tag, _prepare_visual_robust_images(unwrapped_policy, batch, keys, accelerator.device)))

    if not groups:
        return None, {"visual_robust_num_views": 0.0}

    features = _encode_flat_visual_robust(
        unwrapped_policy,
        torch.cat([views.flatten(0, 1) for _, views in groups], dim=0),
        encoder_chunk_size,
        head_mode,
        head,
        freeze_backbone,
    )

    losses, metrics = [], {}
    offset = 0
    for tag, views in groups:
        batch_size, num_views = views.shape[:2]
        count = batch_size * num_views
        group_features = features[offset : offset + count]
        offset += count

        if objective == "alignment":
            # Positives only: 1 - mean cosine similarity inside each frame's view group. No negative
            # term, so nothing in this loss opposes every embedding collapsing to one point -- the
            # action loss is the only thing keeping the representation informative.
            group_loss = _positive_group_alignment_loss(group_features, batch_size, num_views)
            loss_key = "alignment_loss"
        elif objective == "contrastive":
            labels = torch.arange(batch_size, device=group_features.device).repeat_interleave(num_views)
            group_loss = _supervised_contrastive_loss(group_features, labels, temperature=temperature)
            loss_key = "contrastive_loss"
        else:
            raise ValueError(
                f"visual_robust_front_objective must be 'contrastive' or 'alignment', got {objective!r}"
            )
        losses.append(group_loss)

        name = f"visual_robust_{tag}" if tag else "visual_robust"
        metrics[f"{name}_{loss_key}"] = group_loss.detach().float().item()
        metrics[f"{name}_num_views"] = float(num_views)
        # Diagnostics. The alignment objective is minimised by every embedding being identical, and
        # _positive_group_alignment_loss also returns exactly 0 when it is handed an empty group, so a
        # reported 0.0 is ambiguous on its own: feat_std distinguishes "collapsed / degenerate" from
        # "genuinely aligned", and batch says whether the group had any samples at all.
        metrics[f"{name}_batch"] = float(batch_size)
        metrics[f"{name}_feat_std"] = group_features.detach().float().std().item()

    if "episode_index" in batch:
        metrics["visual_robust_episode_unique"] = float(torch.unique(batch["episode_index"]).numel())
    metrics["visual_robust_num_groups"] = float(len(losses))
    return sum(losses) / len(losses), metrics


def compute_visual_robust_contrastive_loss(
    policy: PreTrainedPolicy,
    batch: dict[str, Any],
    accelerator: Accelerator,
    temperature: float,
    max_views: int | None,
    encoder_chunk_size: int,
    image_prefix: str = "observation.image.",
    metric_prefix: str = "visual_robust",
    random_views: bool = False,
    head_mode: str = "none",
) -> tuple[torch.Tensor | None, dict[str, float]]:
    pooled_features, batch_size, num_views = _encode_visual_robust_features(
        policy=policy,
        batch=batch,
        accelerator=accelerator,
        max_views=max_views,
        encoder_chunk_size=encoder_chunk_size,
        image_prefix=image_prefix,
        random_views=random_views,
        head_mode=head_mode,
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
    visual_robust_random_views: bool = False,
    visual_robust_front_prefixes: tuple[str, ...] = ("observation.image.",),
    visual_robust_head_mode: str = "none",
    visual_robust_head: torch.nn.Module | None = None,
    visual_robust_freeze_backbone: bool = True,
    visual_robust_front_objective: str = "contrastive",
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
    visual_robust_include_views=None,
    visual_robust_vqa_weight: float = 0.0,
    visual_robust_vqa_tokenizer=None,
    visual_robust_vqa_references=None,
    visual_robust_vqa_batch_size: int | None = None,
    vision_l2sp_weight: float = 0.0,
    vision_l2sp_scope: str = "vision",
    vision_l2sp_reference: dict[str, torch.Tensor] | None = None,
    vision_l2sp_reference_sq_norm: float = 0.0,
    vision_distill_weight: float = 0.0,
    vision_distill_teacher: torch.nn.Module | None = None,
    vision_distill_capture: "_StudentFeatureCapture | None" = None,
    visual_robust_state_weight: float = 0.0,
    visual_robust_state_head: torch.nn.Module | None = None,
    visual_robust_state_normalizer=None,
    visual_robust_state_episode_offsets=None,
    visual_robust_state_policy_weight: float = 0.0,
    policy_state_normalizer=None,
    visual_robust_state_head_policy: torch.nn.Module | None = None,
    policy_state_slices=(7, 10, 14),
    visual_robust_state_key: str = "observation.state",
    visual_robust_state_quat_slice: slice = slice(3, 7),
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
            distill_active = (
                vision_distill_weight > 0
                and vision_distill_teacher is not None
                and vision_distill_capture is not None
            )
            policy_state_active = (
                visual_robust_state_policy_weight > 0
                and visual_robust_state_head_policy is not None
                and policy_state_normalizer is not None
                and vision_distill_capture is not None
            )
            # Armed only around the policy's own forward, so the auxiliary visual-robust encodes
            # below cannot leak into the captured list and get distilled too.
            if distill_active or policy_state_active:
                vision_distill_capture.arm()
            loss, output_dict = policy.forward(batch)
            if distill_active or policy_state_active:
                vision_distill_capture.disarm()
            # Guarded separately: policy_state_active alone arms the same hook, and calling the
            # distillation term then would pass it a teacher that was never built.
            if distill_active:
                distill_loss, distill_metrics = compute_vision_distill_loss(
                    teacher=vision_distill_teacher,
                    capture=vision_distill_capture,
                    accelerator=accelerator,
                )
                if distill_loss is not None:
                    loss = loss + vision_distill_weight * distill_loss
                    output_dict.update(distill_metrics)
                    output_dict["vision_distill_weight"] = vision_distill_weight
                    output_dict["loss_with_distill"] = loss.detach().float().item()

            if policy_state_active and vision_distill_capture.captured:
                # The hook fires once per camera; this corpus is front-camera only, so take the
                # first (and only) capture rather than silently averaging cameras together.
                _, policy_tokens = vision_distill_capture.captured[0]
                policy_state_loss, policy_state_metrics = compute_policy_state_loss(
                    tokens=policy_tokens,
                    batch=batch,
                    head=visual_robust_state_head_policy,
                    normalizer=policy_state_normalizer,
                    accelerator=accelerator,
                    pos_start=policy_state_slices[0],
                    quat_start=policy_state_slices[1],
                    grip_index=policy_state_slices[2],
                )
                if policy_state_loss is not None:
                    loss = loss + visual_robust_state_policy_weight * policy_state_loss
                    output_dict.update(policy_state_metrics)
                    output_dict["policy_state_weight"] = visual_robust_state_policy_weight

            if vision_distill_capture is not None:
                vision_distill_capture.captured.clear()

        if visual_robust_contrastive_weight > 0:
            # One contrastive term per prefix, averaged. With a single prefix this is exactly the
            # old behaviour. With several (e.g. observation.image.left. / observation.image.right.)
            # each camera viewpoint gets its OWN positive group, so left renders are only ever
            # contrasted against other left renders and right against right.
            #
            # This matters because the loss pulls every view of a frame together: put left and right
            # in one group and it would also force the two viewpoints onto the same representation,
            # i.e. train away the viewpoint information instead of only the robot/background nuisance
            # it is meant to remove. Averaging (not summing) keeps the term's scale independent of how
            # many prefixes are configured, so visual_robust_contrastive_weight keeps its meaning.
            vr_batch = visual_robust_batch if visual_robust_batch is not None else batch
            contrastive_loss, contrastive_metrics = compute_visual_robust_contrastive_loss_multi(
                policy=policy,
                batch=vr_batch,
                accelerator=accelerator,
                temperature=visual_robust_temperature,
                max_views=visual_robust_max_views,
                encoder_chunk_size=visual_robust_encoder_chunk_size,
                prefixes=visual_robust_front_prefixes,
                random_views=visual_robust_random_views,
                head_mode=visual_robust_head_mode,
                head=visual_robust_head,
                freeze_backbone=visual_robust_freeze_backbone,
                objective=visual_robust_front_objective,
                include_views=visual_robust_include_views,
            )
            output_dict.update(contrastive_metrics)
            if contrastive_loss is not None:
                loss = loss + visual_robust_contrastive_weight * contrastive_loss
                output_dict[f"visual_robust_{visual_robust_front_objective}_loss"] = (
                    contrastive_loss.detach().float().item()
                )
                output_dict["visual_robust_contrastive_weight"] = visual_robust_contrastive_weight
                output_dict["loss_with_visual_robust"] = loss.detach().float().item()

        if (
            visual_robust_vqa_weight > 0
            and visual_robust_vqa_tokenizer is not None
            and visual_robust_vqa_references is not None
        ):
            vqa_loss, vqa_metrics = compute_visual_robust_vqa_loss(
                policy=policy,
                batch=visual_robust_batch if visual_robust_batch is not None else batch,
                accelerator=accelerator,
                tokenizer=visual_robust_vqa_tokenizer,
                references=visual_robust_vqa_references,
                prefixes=visual_robust_front_prefixes,
                max_views=visual_robust_max_views,
                random_views=visual_robust_random_views,
                max_frames=visual_robust_vqa_batch_size,
                include_views=visual_robust_include_views,
            )
            output_dict.update(vqa_metrics)
            if vqa_loss is not None:
                loss = loss + visual_robust_vqa_weight * vqa_loss
                output_dict["visual_robust_vqa_weight"] = visual_robust_vqa_weight
                output_dict["loss_with_vqa"] = loss.detach().float().item()

        if visual_robust_state_weight > 0 and visual_robust_state_head is not None:
            state_loss, state_metrics = compute_visual_robust_state_loss(
                policy=policy,
                batch=visual_robust_batch if visual_robust_batch is not None else batch,
                accelerator=accelerator,
                head=visual_robust_state_head,
                normalizer=visual_robust_state_normalizer,
                episode_offsets=visual_robust_state_episode_offsets,
                state_key=visual_robust_state_key,
                quat_slice=visual_robust_state_quat_slice,
                max_views=visual_robust_max_views,
                random_views=visual_robust_random_views,
                prefixes=visual_robust_front_prefixes,
                encoder_chunk_size=visual_robust_encoder_chunk_size,
                head_mode=visual_robust_head_mode,
                projection_head=visual_robust_head,
                freeze_backbone=visual_robust_freeze_backbone,
            )
            if state_loss is not None:
                loss = loss + visual_robust_state_weight * state_loss
                output_dict.update(state_metrics)
                output_dict["visual_robust_state_weight"] = visual_robust_state_weight

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

    # L2-SP sits OUTSIDE the autocast block on purpose: it is a function of the weights, not of the
    # batch, and (w - w0) is a difference between nearly equal float32 numbers that bf16 would round
    # away entirely -- the same failure that once pinned the alignment objective's cosine at exactly
    # 1.0 and killed its gradient.
    #
    # It is still added to `loss` before the single accelerator.backward() below, so both terms live
    # in one autograd graph and each parameter's AccumulateGrad node fires exactly once. Backward-ing
    # them separately would make DDP see the vision tower's gradients twice and abort the step with
    # "Expected to mark a variable ready only once".
    if vision_l2sp_weight > 0 and vision_l2sp_reference is not None:
        l2sp_loss, l2sp_metrics = compute_l2sp_loss(
            policy=policy,
            reference=vision_l2sp_reference,
            reference_sq_norm=vision_l2sp_reference_sq_norm,
            scope=vision_l2sp_scope,
        )
        if l2sp_loss is not None:
            loss = loss + vision_l2sp_weight * l2sp_loss
            output_dict.update(l2sp_metrics)
            output_dict["vision_l2sp_weight"] = vision_l2sp_weight
            output_dict["loss_with_l2sp"] = loss.detach().float().item()

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

    # The LAP knowledge-insulation objective describes the action chunk in words ("move forward
    # moderately"), which only means anything in raw command units -- but the policy is handed
    # mean/std-normalized actions. Give it the stats so it can undo that. Must happen before
    # `accelerator.prepare` wraps the policy.
    if getattr(cfg.policy, "knowledge_insulation", False) and getattr(cfg.policy, "ki_objective", "") == "lap":
        action_stats = ds_meta.stats["action"]
        policy.model.set_action_stats(action_stats["mean"], action_stats["std"])
        if is_main_process:
            logging.info(
                "LAP objective: action stats attached (mean[:12]=%s)",
                np.round(np.asarray(action_stats["mean"])[:12], 3).tolist(),
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

    # Built before the optimizer so its parameters can be added to it, but deliberately NOT
    # registered on the policy -- see make_visual_robust_head() for why that breaks under DDP.
    visual_robust_head = None
    if cfg.dataset.visual_robust_head_mode == "adapter_mlp":
        visual_robust_head, head_in_dim = make_visual_robust_head(
            policy,
            hidden_dim=cfg.dataset.visual_robust_head_hidden_dim,
            out_dim=cfg.dataset.visual_robust_head_output_dim,
            num_layers=cfg.dataset.visual_robust_head_layers,
        )
        if is_main_process:
            logging.info(
                "Visual robust head: adapter_mlp  connector_out=%d -> hidden=%d x %d layers -> %d "
                "(%s params). Backbone %s by the contrastive loss; connector + head are trained.",
                head_in_dim,
                cfg.dataset.visual_robust_head_hidden_dim,
                cfg.dataset.visual_robust_head_layers,
                cfg.dataset.visual_robust_head_output_dim,
                format_big_number(sum(p.numel() for p in visual_robust_head.parameters())),
                "FROZEN" if cfg.dataset.visual_robust_freeze_backbone else "TRAINED",
            )

    # Same standalone-module treatment as visual_robust_head, for the same DDP reason: it is only
    # ever called outside the policy's forward.
    visual_robust_state_head = None
    state_in_dim = (
        policy.model.vlm_with_expert.get_vlm_model().connector.modality_projection.proj.out_features
        if cfg.dataset.visual_robust_head_mode == "adapter_mlp"
        else policy.model.vlm_with_expert.get_vlm_model().vision_model.config.hidden_size
    )
    if cfg.dataset.visual_robust_state_weight > 0:
        visual_robust_state_head = VisualRobustStateHead(
            state_in_dim,
            cfg.dataset.visual_robust_state_head_hidden_dim,
            EEF_STATE_DIM,
            cfg.dataset.visual_robust_state_head_layers,
            pool=cfg.dataset.visual_robust_state_pool,
        ).to(dtype=next(policy.parameters()).dtype, device=next(policy.parameters()).device)
        if is_main_process:
            logging.info(
                "Visual robust EEF-state head: %d (pool=%s) -> hidden %d x %d layers -> %d (%s params)",
                state_in_dim,
                cfg.dataset.visual_robust_state_pool,
                cfg.dataset.visual_robust_state_head_hidden_dim,
                cfg.dataset.visual_robust_state_head_layers,
                EEF_STATE_DIM,
                format_big_number(sum(p.numel() for p in visual_robust_state_head.parameters())),
            )

    # A SECOND head, not the auxiliary one reused. The two branches regress the same physical
    # quantity but in different frames -- the auxiliary export's fingertip is world-frame and gets
    # per-episode centering, while the policy corpus's eef pose is already base-relative -- so they
    # are z-scored against different means. One shared head has to satisfy both at once and cannot:
    # measured at step 200 it left policy_state_loss at 6.6 against the auxiliary branch's ~2.5.
    # Separate heads let each own its convention while both still shape the same visual backbone,
    # which is where the benefit actually lives.
    visual_robust_state_head_policy = None
    if cfg.dataset.visual_robust_state_policy_weight > 0:
        visual_robust_state_head_policy = VisualRobustStateHead(
            state_in_dim,
            cfg.dataset.visual_robust_state_head_hidden_dim,
            EEF_STATE_DIM,
            cfg.dataset.visual_robust_state_head_layers,
            pool=cfg.dataset.visual_robust_state_pool,
        ).to(dtype=next(policy.parameters()).dtype, device=next(policy.parameters()).device)

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    if visual_robust_head is not None:
        optimizer.add_param_group({"params": list(visual_robust_head.parameters())})
    if visual_robust_state_head is not None:
        optimizer.add_param_group({"params": list(visual_robust_state_head.parameters())})
    if visual_robust_state_head_policy is not None:
        optimizer.add_param_group({"params": list(visual_robust_state_head_policy.parameters())})

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
    visual_robust_random_views = cfg.dataset.visual_robust_random_views
    visual_robust_head_mode = cfg.dataset.visual_robust_head_mode
    visual_robust_freeze_backbone = cfg.dataset.visual_robust_freeze_backbone
    visual_robust_front_objective = cfg.dataset.visual_robust_front_objective
    visual_robust_front_prefixes = tuple(
        p.strip() for p in (cfg.dataset.visual_robust_front_prefixes or "observation.image.").split(",") if p.strip()
    )
    visual_robust_include_views = tuple(
        v.strip() for v in (cfg.dataset.visual_robust_include_views or "").split(",") if v.strip()
    ) or None
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
    vision_l2sp_weight = cfg.dataset.vision_l2sp_weight
    vision_l2sp_scope = cfg.dataset.vision_l2sp_scope
    vision_distill_weight = cfg.dataset.vision_distill_weight
    visual_robust_state_weight = cfg.dataset.visual_robust_state_weight
    visual_robust_state_policy_weight = cfg.dataset.visual_robust_state_policy_weight
    visual_robust_state_key = cfg.dataset.visual_robust_state_key
    visual_robust_state_quat_slice = slice(
        cfg.dataset.visual_robust_state_quat_start, cfg.dataset.visual_robust_state_quat_start + 4
    )
    visual_robust_contrastive_enabled = (
        visual_robust_contrastive_weight > 0
        or visual_robust_wrist_alignment_weight > 0
        or visual_robust_state_weight > 0
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
    if visual_robust_contrastive_enabled or cfg.dataset.visual_robust_vqa_weight > 0:
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
    if visual_robust_head is not None:
        # Its own DDP wrapper: the head's forward then IS a DDP forward, so its gradients are
        # all-reduced across ranks like the policy's.
        visual_robust_head = accelerator.prepare(visual_robust_head)
    if visual_robust_state_head is not None:
        visual_robust_state_head = accelerator.prepare(visual_robust_state_head)
    if visual_robust_state_head_policy is not None:
        visual_robust_state_head_policy = accelerator.prepare(visual_robust_state_head_policy)

    # Built from the AUXILIARY dataset's stats, not the policy dataset's: the target lives only in
    # the visual-robust export, and its scale differs per tree.
    policy_state_normalizer = None
    if visual_robust_state_policy_weight > 0:
        policy_state_normalizer = build_policy_state_normalizer(
            dataset,
            cfg.dataset.visual_robust_state_key,
            cfg.dataset.visual_robust_state_policy_pos_start,
            cfg.dataset.visual_robust_state_policy_grip_index,
        )
        if is_main_process:
            logging.info(
                "Policy-batch EEF-state term: weight %s, eef slice %d:%d, gripper %d",
                visual_robust_state_policy_weight,
                cfg.dataset.visual_robust_state_policy_pos_start,
                cfg.dataset.visual_robust_state_policy_pos_start + 7,
                cfg.dataset.visual_robust_state_policy_grip_index,
            )

    visual_robust_vqa_weight = cfg.dataset.visual_robust_vqa_weight
    visual_robust_vqa_tokenizer = None
    visual_robust_vqa_references = None
    if visual_robust_vqa_weight > 0:
        if visual_robust_loader is None:
            raise ValueError(
                "visual_robust_vqa_weight > 0 requires the auxiliary dataset: set "
                "--dataset.visual_robust_root and --dataset.visual_robust_repo_id."
            )
        visual_robust_vqa_tokenizer = VQAStateTokenizer(
            _get_unwrapped_policy(policy, accelerator).model.vlm_with_expert.processor.tokenizer,
            position_resolution_cm=cfg.dataset.visual_robust_vqa_position_resolution_cm,
            yaw_resolution_deg=cfg.dataset.visual_robust_vqa_yaw_resolution_deg,
        )
        visual_robust_vqa_references = build_episode_reference_states(visual_robust_loader.dataset)
        if is_main_process:
            logging.info(
                "VQA objective: %d episode reference frames, %.0f cm / %.0f deg resolution",
                len(visual_robust_vqa_references),
                cfg.dataset.visual_robust_vqa_position_resolution_cm,
                cfg.dataset.visual_robust_vqa_yaw_resolution_deg,
            )
            example = visual_robust_vqa_tokenizer.describe(12.0, -3.0, 84.0, 15.0, 1.0)
            logging.info('  example answer: "%s"', example)

    visual_robust_state_normalizer = None
    visual_robust_state_offsets = None
    if visual_robust_state_weight > 0:
        if visual_robust_loader is None:
            raise ValueError(
                "visual_robust_state_weight > 0 requires the auxiliary dataset: set "
                "--dataset.visual_robust_root and --dataset.visual_robust_repo_id."
            )
        visual_robust_state_offsets = build_episode_position_offsets(
            visual_robust_loader.dataset, visual_robust_state_key, slice(0, 3)
        )
        visual_robust_state_normalizer = compute_eef_state_normalizer(
            visual_robust_loader.dataset,
            visual_robust_state_key,
            visual_robust_state_quat_slice,
            episode_offsets=visual_robust_state_offsets,
        )
        if is_main_process:
            mean, std, dim_weight = visual_robust_state_normalizer
            dropped = (dim_weight == 0).nonzero().flatten().tolist()
            logging.info(
                "EEF-state target: %d/%d dimensions used%s",
                int(dim_weight.sum()),
                dim_weight.numel(),
                f", dropped as constant: {dropped}" if dropped else "",
            )
            logging.info("  mean %s", torch.round(mean, decimals=3).tolist())
            logging.info("  std  %s", torch.round(std, decimals=3).tolist())
            logging.info(
                "  position target is per-episode centered over %d episodes",
                len(visual_robust_state_offsets),
            )
    if visual_robust_loader is not None:
        visual_robust_loader = accelerator.prepare(visual_robust_loader)
        visual_robust_iter = cycle(visual_robust_loader)
    dl_iter = cycle(dataloader)

    # L2-SP anchor. Taken here -- after prepare(), before the first step -- so the snapshot is exactly
    # the pretrained tower the run starts from.
    vision_l2sp_reference = None
    vision_l2sp_reference_sq_norm = 0.0
    if vision_l2sp_weight > 0:
        if not cfg.policy.load_vlm_weights:
            raise ValueError(
                "vision_l2sp_weight > 0 requires --policy.load_vlm_weights=true: with randomly "
                "initialised VLM weights there is no pretrained point to regularise toward, and the "
                "penalty would silently anchor training to noise."
            )
        if cfg.resume:
            # Resumed weights are already drifted, so the anchor must come from the pretrained VLM
            # rather than from the live policy. Reloading gives the identical anchor a fresh run
            # would have used, which is what keeps a resumed run comparable to an unbroken one.
            if vision_l2sp_scope != "vision":
                raise ValueError(
                    f"--resume with vision_l2sp_scope={vision_l2sp_scope!r} is not supported; only "
                    "'vision' can be reloaded from the pretrained VLM as an anchor."
                )
            live = _l2sp_target_module(policy, vision_l2sp_scope)
            anchor = load_pretrained_vision_anchor(
                cfg.policy.vlm_model_name, next(live.parameters()).device
            )
            _assert_anchor_matches(anchor, live, "L2-SP")
            vision_l2sp_reference = {
                name: param.detach().clone().float() for name, param in anchor.named_parameters()
            }
            vision_l2sp_reference_sq_norm = float(
                sum(t.pow(2).sum() for t in vision_l2sp_reference.values())
            )
            del anchor
            if is_main_process:
                logging.info("L2-SP: resumed run, anchor reloaded from %s", cfg.policy.vlm_model_name)
        else:
            vision_l2sp_reference, vision_l2sp_reference_sq_norm = capture_l2sp_reference(
                policy, vision_l2sp_scope
            )
        if is_main_process:
            n_ref = sum(t.numel() for t in vision_l2sp_reference.values())
            logging.info(
                "L2-SP: anchoring %s (%s params, %.2f GiB fp32 reference) with weight %s",
                vision_l2sp_scope,
                format_big_number(n_ref),
                n_ref * 4 / 1024**3,
                vision_l2sp_weight,
            )

    # Feature-space anchor. Same "must be the pretrained tower" requirement as L2-SP, for the same
    # reason: a teacher copied from already-drifted weights would distil toward the wrong function.
    vision_distill_teacher = None
    vision_distill_capture = None
    if vision_distill_weight > 0 or visual_robust_state_policy_weight > 0:
        if not cfg.policy.load_vlm_weights:
            raise ValueError(
                "vision_distill_weight > 0 requires --policy.load_vlm_weights=true: there would "
                "otherwise be no pretrained tower to distil from."
            )
        live = _l2sp_target_module(policy, vision_l2sp_scope)
        if cfg.resume and vision_distill_weight > 0:
            # Same reasoning as L2-SP above: deepcopying the live tower on a resumed run would make
            # the "teacher" a copy of the drifted student, and the distillation term would collapse
            # to ~0 while appearing to work.
            if vision_l2sp_scope != "vision":
                raise ValueError(
                    f"--resume with vision_l2sp_scope={vision_l2sp_scope!r} is not supported for "
                    "feature distillation; only 'vision' can be reloaded from the pretrained VLM."
                )
            vision_distill_teacher = load_pretrained_vision_anchor(
                cfg.policy.vlm_model_name, next(live.parameters()).device
            )
            _assert_anchor_matches(vision_distill_teacher, live, "distillation")
            if is_main_process:
                logging.info(
                    "Feature distillation: resumed run, teacher reloaded from %s",
                    cfg.policy.vlm_model_name,
                )
        elif vision_distill_weight > 0:
            vision_distill_teacher = make_frozen_vision_teacher(policy, vision_l2sp_scope)
        vision_distill_capture = _StudentFeatureCapture(live)
        # The capture hook is shared: the policy-batch EEF-state term arms it too, and in that case
        # no teacher is built.
        if is_main_process and vision_distill_teacher is not None:
            n_teacher = sum(p.numel() for p in vision_distill_teacher.parameters())
            logging.info(
                "Feature distillation: frozen teacher (%s params, %.2f GiB) with weight %s",
                format_big_number(n_teacher),
                n_teacher * 4 / 1024**3,
                vision_distill_weight,
            )

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
            visual_robust_random_views=visual_robust_random_views,
            visual_robust_front_prefixes=visual_robust_front_prefixes,
            visual_robust_head_mode=visual_robust_head_mode,
            visual_robust_head=visual_robust_head,
            visual_robust_freeze_backbone=visual_robust_freeze_backbone,
            visual_robust_front_objective=visual_robust_front_objective,
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
            visual_robust_include_views=visual_robust_include_views,
            visual_robust_vqa_weight=visual_robust_vqa_weight,
            visual_robust_vqa_tokenizer=visual_robust_vqa_tokenizer,
            visual_robust_vqa_references=visual_robust_vqa_references,
            visual_robust_vqa_batch_size=cfg.dataset.visual_robust_vqa_batch_size,
            vision_l2sp_weight=vision_l2sp_weight,
            vision_l2sp_scope=vision_l2sp_scope,
            vision_l2sp_reference=vision_l2sp_reference,
            vision_l2sp_reference_sq_norm=vision_l2sp_reference_sq_norm,
            vision_distill_weight=vision_distill_weight,
            vision_distill_teacher=vision_distill_teacher,
            vision_distill_capture=vision_distill_capture,
            visual_robust_state_weight=visual_robust_state_weight,
            visual_robust_state_head=visual_robust_state_head,
            visual_robust_state_normalizer=visual_robust_state_normalizer,
            visual_robust_state_episode_offsets=visual_robust_state_offsets,
            visual_robust_state_policy_weight=visual_robust_state_policy_weight,
            policy_state_normalizer=policy_state_normalizer,
            visual_robust_state_head_policy=visual_robust_state_head_policy,
            policy_state_slices=(
                cfg.dataset.visual_robust_state_policy_pos_start,
                cfg.dataset.visual_robust_state_policy_quat_start,
                cfg.dataset.visual_robust_state_policy_grip_index,
            ),
            visual_robust_state_key=visual_robust_state_key,
            visual_robust_state_quat_slice=visual_robust_state_quat_slice,
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
