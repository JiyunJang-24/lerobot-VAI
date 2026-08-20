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

from dataclasses import dataclass, field

from lerobot.datasets.transforms import ImageTransformsConfig
from lerobot.datasets.video_utils import get_safe_default_codec


@dataclass
class DatasetConfig:
    # You may provide a list of datasets here. `train.py` creates them all and concatenates them. Note: only data
    # keys common between the datasets are kept. Each dataset gets and additional transform that inserts the
    # "dataset_index" into the returned item. The index mapping is made according to the order in which the
    # datasets are provided.
    repo_id: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | None = None
    episodes: list[int] | None = None
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = field(default_factory=get_safe_default_codec)
    streaming: bool = False
    cache_in_memory: bool = False
    use_wrist_cam: bool = True
    use_state: bool = True
    visual_robust_repo_id: str | None = None
    visual_robust_root: str | None = None
    visual_robust_contrastive_weight: float = 0.0
    visual_robust_temperature: float = 0.1
    visual_robust_max_views: int | None = None
    # When the auxiliary dataset offers more views than visual_robust_max_views, draw a fresh random
    # subset each step instead of always taking the first ones alphabetically. Required for the
    # background-variation export, whose keys sort by embodiment: the first 4 of 12 are all IIWA, so
    # the contrastive term would never see a cross-embodiment positive pair.
    visual_robust_random_views: bool = False
    # Comma-separated batch-key prefixes, one contrastive group each (averaged). Default (None) means
    # a single "observation.image." group. Set e.g.
    # "observation.image.left.,observation.image.right." to contrast left renders only against other
    # left renders and right against right, instead of collapsing both viewpoints into one group.
    visual_robust_front_prefixes: str | None = None
    # "none" (default) applies the contrastive loss straight to the mean-pooled vision backbone
    # output, so its invariance pressure lands on the backbone the policy shares. "adapter_mlp"
    # instead freezes the backbone *for this loss only* (the policy still trains it), routes the
    # features through the VLM's existing connector -- whose output is what the VLM itself consumes
    # -- pools those tokens, and contrasts an MLP head on top, so the head absorbs the pressure.
    # "contrastive" (default): supervised contrastive -- pulls a frame's views together AND pushes
    # different frames apart. "alignment": positives only, 1 - mean cosine similarity within each
    # frame's view group, with no negative term at all.
    visual_robust_front_objective: str = "contrastive"
    visual_robust_head_mode: str = "none"
    # Only consulted when head_mode="adapter_mlp" (head_mode="none" contrasts the backbone output
    # directly, so freezing it there would leave the loss nothing to train). True keeps the
    # contrastive gradient off the backbone so only connector+head absorb it; False lets it flow all
    # the way through, which costs ~17x more memory per auxiliary sample because the backbone's
    # activations for those views must be kept for backward -- measured on this box at batch 48:
    # 0.19 GB vs 3.16 GB per visual_robust_batch_size unit.
    visual_robust_freeze_backbone: bool = True
    visual_robust_head_hidden_dim: int = 1024
    visual_robust_head_output_dim: int = 256
    visual_robust_head_layers: int = 3
    visual_robust_wrist_alignment_weight: float = 0.0
    visual_robust_wrist_alignment_mode: str = "all"
    visual_robust_wrist_alignment_max_views: int | None = None
    visual_robust_wrist_width_bin_size: float = 0.01
    visual_robust_wrist_width_temperature: float = 0.1
    visual_robust_wrist_width_min: float = 0.0
    visual_robust_wrist_width_max: float = 0.08
    visual_robust_wrist_width_sigma: float = 0.2
    visual_robust_wrist_width_state_key: str = "observation.state"
    visual_robust_wrist_width_left_index: int = 0
    visual_robust_wrist_width_right_index: int = 1
    visual_robust_encoder_chunk_size: int = 32
    visual_robust_batch_size: int | None = None
    visual_robust_num_workers: int = 0
    visual_robust_cache_in_memory: bool = True
    visual_robust_same_episode_negatives: bool = True
    # L2-SP: keep a frozen copy of the PRETRAINED vision tower and penalise how far the trainable
    # tower drifts from it, sum_i (w_i - w0_i)^2, added to the policy loss with this coefficient.
    # 0.0 (default) disables it and costs nothing.
    #
    # Motivation, measured on this project's barx front-only runs: fine-tuning the tower makes its
    # positive/negative gap WORSE than the pretrained initialisation (-0.138 centered vs -0.050),
    # i.e. policy training actively teaches the encoder to separate embodiments. Freezing the tower
    # outright prevents that but also stops it adapting to the robot domain at all. This is the
    # middle ground: the tower still trains, but is pulled back toward its pretrained weights.
    #
    # The gradient contribution is 2*weight*(w - w0) per parameter, which does not grow with the
    # parameter count, so `weight` behaves like a weight-decay rate toward the pretrained point
    # rather than toward zero. The reported loss value does scale with parameter count -- read
    # `vision_l2sp_relative_drift` instead when judging whether the pull is doing anything.
    vision_l2sp_weight: float = 0.0
    # "vision" penalises only the SigLIP tower. "vlm" also covers the language model, which the
    # policy fine-tunes too.
    vision_l2sp_scope: str = "vision"
    # Feature-space counterpart of vision_l2sp_weight: run a frozen copy of the PRETRAINED tower on
    # the same images and penalise the mean squared difference between its patch tokens and the
    # trainable tower's. 0.0 (default) disables it.
    #
    # Not a substitute for the weight-space penalty: weights can travel a long way while the function
    # computed on this particular data barely moves, and vice versa. This one constrains the tower's
    # behaviour where it is actually used; L2-SP constrains where it sits in parameter space.
    #
    # Costs one extra no-grad forward of the 86.4M-parameter tower per step plus 0.32 GiB for its
    # weights. The trainable side is captured by a hook from the policy's own forward, so it adds no
    # student-side compute. Read `vision_distill_relative_error` to judge the pull.
    vision_distill_weight: float = 0.0
    # EEF-state regression on the visual-robust views: an MLP on the pooled visual feature predicts
    # the frame's end-effector state (fingertip xyz + quat_xyzw + gripper). 0.0 disables it.
    #
    # The auxiliary export stores ONE observation.state per frame and one render per robot, so all
    # views of a frame share a target. Forcing three visually different arms to decode to the same
    # pose is a supervised path to embodiment invariance -- and unlike the contrastive/alignment
    # terms, the invariant it produces is pinned to something physical, so it cannot be satisfied by
    # a degenerate representation that discards the scene.
    visual_robust_state_weight: float = 0.0
    visual_robust_state_head_hidden_dim: int = 512
    visual_robust_state_head_layers: int = 2
    visual_robust_state_key: str = "observation.state"
    # Layout of that state vector: xyz(0:3), quat_xyzw(3:7), gripper(7).
    visual_robust_state_quat_start: int = 3
    # "mean" pools the vision tower's tokens by averaging; "attn" adds learned attention pooling
    # beside it. Position regression from a mean over 1024 tokens is close to position-blind, so
    # "attn" gives the head a direct route to WHERE the fingertip is.
    visual_robust_state_pool: str = "mean"
    # Same EEF-state regression, but on the POLICY batch as well as the auxiliary one. 0.0 disables.
    #
    # The auxiliary export has ~324 episodes; the policy corpus has 2900. Training the head on both
    # grounds the visual features on ~9x more data. It costs no extra vision compute: the policy's own
    # forward already encodes those images, and the tokens are taken from it with the same hook the
    # distillation term uses.
    #
    # Different layout, verified from the variance structure: the policy state is
    # base_pose(0:7) + eef_pose(7:14) + gripper(14:16). Only the eef part is used -- the base xyz has
    # a between/within-episode variance ratio of 1910x, i.e. it is the robot's parking spot, which a
    # robot-mounted camera cannot see. Unlike the auxiliary export's world-frame fingertip, this eef
    # pose is already base-relative (ratio 0.1-0.5), so it needs no per-episode centering.
    visual_robust_state_policy_weight: float = 0.0

    # VQA on the visual-robust renders: ask the VLM where the gripper is and score the answer with
    # the LM head. Unlike the state-regression head above, this trains the WHOLE VLM rather than the
    # vision tower alone, which is what knowledge insulation needs once its own token objective has
    # saturated. See policies/smolvla/vqa_state_text.py for what the answer says and which frame
    # each quantity is measured in.
    visual_robust_vqa_weight: float = 0.0
    visual_robust_vqa_position_resolution_cm: float = 1.0
    visual_robust_vqa_yaw_resolution_deg: float = 5.0
    # Frames per VQA step. Each frame contributes one sample PER embodiment render, so the real
    # sample count is this times the number of views.
    visual_robust_vqa_batch_size: int = 8
    visual_robust_state_policy_pos_start: int = 7
    visual_robust_state_policy_quat_start: int = 10
    visual_robust_state_policy_grip_index: int = 14

@dataclass
class WandBConfig:
    enable: bool = False
    # Set to true to disable saving an artifact despite training.save_checkpoint=True
    disable_artifact: bool = False
    project: str = "lerobot"
    entity: str | None = None
    notes: str | None = None
    run_id: str | None = None
    mode: str | None = None  # Allowed values: 'online', 'offline' 'disabled'. Defaults to 'online'


@dataclass
class EvalConfig:
    n_episodes: int = 50
    # `batch_size` specifies the number of environments to use in a gym.vector.VectorEnv.
    batch_size: int = 50
    # `use_async_envs` specifies whether to use asynchronous environments (multiprocessing).
    use_async_envs: bool = False

    def __post_init__(self) -> None:
        if self.batch_size > self.n_episodes:
            raise ValueError(
                "The eval batch size is greater than the number of eval episodes "
                f"({self.batch_size} > {self.n_episodes}). As a result, {self.batch_size} "
                f"eval environments will be instantiated, but only {self.n_episodes} will be used. "
                "This might significantly slow down evaluation. To fix this, you should update your command "
                f"to increase the number of episodes to match the batch size (e.g. `eval.n_episodes={self.batch_size}`), "
                f"or lower the batch size (e.g. `eval.batch_size={self.n_episodes}`)."
            )


@dataclass
class PeftConfig:
    # PEFT offers many fine-tuning methods, layer adapters being the most common and currently also the most
    # effective methods so we'll focus on those in this high-level config interface.

    # Either a string (module name suffix or 'all-linear'), a list of module name suffixes or a regular expression
    # describing module names to target with the configured PEFT method. Some policies have a default value for this
    # so that you don't *have* to choose which layers to adapt but it might still be worthwhile depending on your case.
    target_modules: list[str] | str | None = None

    # Names/suffixes of modules to fully fine-tune and store alongside adapter weights. Useful for layers that are
    # not part of a pre-trained model (e.g., action state projections). Depending on the policy this defaults to layers
    # that are newly created in pre-trained policies. If you're fine-tuning an already trained policy you might want
    # to set this to `[]`. Corresponds to PEFT's `modules_to_save`.
    full_training_modules: list[str] | None = None

    # The PEFT (adapter) method to apply to the policy. Needs to be a valid PEFT type.
    method_type: str = "LORA"

    # Adapter initialization method. Look at the specific PEFT adapter documentation for defaults.
    init_type: str | None = None

    # We expect that all PEFT adapters are in some way doing rank-decomposition therefore this parameter specifies
    # the rank used for the adapter. In general a higher rank means more trainable parameters and closer to full
    # fine-tuning.
    r: int = 16
