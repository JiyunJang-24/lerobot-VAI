#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import math
from collections import deque
from typing import TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.fast_action_tokenizer import FASTActionTokenizer
from lerobot.policies.smolvla.language_action_text import LanguageActionTokenizer
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.policies.utils import (
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.utils import get_safe_dtype
from lerobot.datasets.visual_cue_utils import (
    remove_extrinsic_camera_axis_correction,
    _rescale_make_motion_basis_axis_rgb_tensor_cam_to_world,
    _get_motion_dynamics_basis,
    _make_motion_basis_axis_rgb_tensor_cam_to_world,
    _make_motion_basis_wrist_axis_rgb_tensor_cam_to_world,
    save_rgb_image,
    PluckerEmbedder,
)
import einops

class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SmolVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "smolvla"

    def __init__(
        self,
        config: SmolVLAConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()
        if self.config.visual_cue_mode == "plucker_concat" or self.config.visual_cue_mode == "plucker":
            self.image_size = 256
            self.plucker_embedder = PluckerEmbedder(img_size=self.image_size, device='cuda')

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Lets create processor if the config provided
        # If RTC is not enabled - we still can track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # In case of calling init_rtc_processor after the model is created
            # We need to set the rtc_processor to the model
            # During the normal initialization process the model is not created yet
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)
        images, img_masks = self.prepare_images(batch)
        #TODO WS: check state is normalized ? and state is same as training ?
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    def _generate_visual_cue(self, images, state):
        pass

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """

        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        if self.config.visual_cue_mode != "vanilla":
            batch = self._get_visual_cues(batch)
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss.

        Args:
            batch: Training batch containing observations and actions.
            noise: Optional noise tensor for flow matching.
            time: Optional time tensor for flow matching.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])
        images, img_masks = self.prepare_images(batch)
        #TODO WS: check state is normalized ?
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")
        loss_dict = {}
        fast_postfix = self.prepare_postfix(batch, actions_is_pad)
        if fast_postfix is not None and reduction == "none":
            raise NotImplementedError(
                "RA-BC per-sample weighting and knowledge insulation are not combined yet: the "
                "token cross-entropy is a scalar over tokens, not a per-sample loss."
            )
        losses, aux = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, time, fast_postfix
        )
        loss_dict["losses_after_forward"] = losses.clone()
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone()

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.mean()
            loss_dict["flow_matching_loss"] = loss.item()
            if aux:
                ce = aux.pop("token_ce_loss")
                loss = loss + self.config.ki_token_loss_weight * ce
                loss_dict["token_ce_loss"] = ce.item()
                loss_dict.update(aux)
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def prepare_postfix(self, batch, actions_is_pad=None):
        """Turn the action chunk into the LM postfix the VLM is trained to predict."""
        if not self.config.knowledge_insulation:
            return None
        # Tokenize the true action dimensions, not the 32-d zero padding: padding the chunk out
        # roughly triples the token count for no added information.
        action_dim = self.config.action_feature.shape[0]
        raw_actions = batch[ACTION][..., :action_dim]
        if self.config.ki_objective == "lap":
            # The sentence describes physical directions, so it has to be built from the raw
            # command, not the mean/std-normalized one. Summing normalized actions over 50 steps
            # adds 50*mean/std to every axis -- on this corpus that offset is ~30 units, an order
            # of magnitude larger than the motion itself, so every label would point the same way.
            raw_actions = self.model.unnormalize_actions(raw_actions)
        postfix = self.model.postfix_tokenizer.encode(raw_actions)
        if actions_is_pad is not None:
            # A chunk that runs past the end of its episode is padded with repeated frames; it
            # would be tokenized as if it were real. Drop those samples from the token loss (the
            # flow-matching term masks them per timestep, which a single token stream cannot do).
            has_pad = actions_is_pad.reshape(actions_is_pad.shape[0], -1).any(dim=1)
            postfix["loss_masks"] = postfix["loss_masks"] & (~has_pad)[:, None]
        return postfix


    def _get_visual_cues(self, item):
        try:
            extrinsic_matrix = item['observation.cam_info']['extrinsic_matrix'].to(torch.float32).squeeze(0)
            extrinsic_matrix = remove_extrinsic_camera_axis_correction(extrinsic_matrix)

            intrinsic_matrix = item['observation.cam_info']['intrinsic_matrix'].to(torch.float32).squeeze(0)
            robot_state = item['observation.cam_info']['unnormalized_state'] # gripper qpos (2), eef pos (3), eef quat (4),
            img = item['observation.image'] # S * C * H * W

            if "basis" in self.config.visual_cue_mode:
                with torch.no_grad():
                    if self.config.visual_cue_mode == "basis_rescale" or self.config.visual_cue_mode == "basis_rescale_concat":
                        axis_tensor, origin_xy = _rescale_make_motion_basis_axis_rgb_tensor_cam_to_world(
                            rgb_tensor=img,                  # (B, 3,H,W)
                            cam_to_world=extrinsic_matrix,                  # cam_pose = cam_to_world (고정)
                            intrinsic_matrix=intrinsic_matrix,
                            robot_eef_abs_poses=robot_state[:, -7:],  # eef pose (B, 7)
                            origin_robot=True,
                            origin_fallback="pp",
                            arrow_len=60,
                            return_overlay=False,
                        )
                        try:
                            wrist_img = item['observation.wrist_image']
                            wrist_intrinsic_matrix = item['observation.cam_info']['wrist_intrinsic_matrix'].to(torch.float32).squeeze(0)
                            wrist_extrinsic_matrix = item['observation.cam_info']['wrist_extrinsic_matrix'].to(torch.float32).squeeze(0)
                            wrist_plucker_extrinsic_matrix = remove_extrinsic_camera_axis_correction(wrist_extrinsic_matrix)
                            wrist_axis_tensor, wrist_origin_xy = _rescale_make_motion_basis_axis_rgb_tensor_cam_to_world(
                                rgb_tensor=wrist_img,
                                cam_to_world=wrist_plucker_extrinsic_matrix,
                                intrinsic_matrix=wrist_intrinsic_matrix,
                                robot_eef_abs_poses=robot_state[:, -7:],
                                origin_robot=True,
                                origin_fallback="pp",
                                arrow_len=60,
                                return_overlay=False,
                            )
                            # save_rgb_image(wrist_axis_tensor[0], "tmp_dir/wrist_scaled_axis_tensor.png")
                            item['observation.wrist_image'] = torch.cat([wrist_img, wrist_axis_tensor], dim=1)
                        except:
                            pass
                    else:
                        motion_dynamics_basis = _get_motion_dynamics_basis(intrinsic_matrix, cam_to_world=extrinsic_matrix).reshape(-1)
                        axis_tensor, origin_xy = _make_motion_basis_axis_rgb_tensor_cam_to_world(
                            rgb_tensor=img,                  # (B, 3,H,W)
                            motion_dynamics_basis=motion_dynamics_basis,
                            cam_to_world=extrinsic_matrix,                  # cam_pose = cam_to_world (고정)
                            intrinsic_matrix=intrinsic_matrix,
                            robot_eef_abs_poses=robot_state[:, -7:],  # eef pose (B, 7)
                            origin_robot=True,
                            origin_fallback="pp",
                            arrow_len=60,
                            return_overlay=False,
                        ) # (B, 3, H, W)
                        try:
                            wrist_img = item['observation.wrist_image']
                            wrist_intrinsic_matrix = item['observation.cam_info']['wrist_intrinsic_matrix'].to(torch.float32).squeeze(0)
                            wrist_extrinsic_matrix = item['observation.cam_info']['wrist_extrinsic_matrix'].to(torch.float32).squeeze(0)
                            wrist_plucker_extrinsic_matrix = remove_extrinsic_camera_axis_correction(wrist_extrinsic_matrix)
                            wrist_axis_tensor, wrist_origin_xy = _make_motion_basis_wrist_axis_rgb_tensor_cam_to_world(
                                rgb_tensor=wrist_img,
                                cam_to_world=wrist_plucker_extrinsic_matrix,
                                intrinsic_matrix=wrist_intrinsic_matrix,
                                robot_eef_abs_poses=robot_state[:, -7:],
                                origin_robot=True,
                                origin_fallback="pp",
                                arrow_len=60,
                                return_overlay=False,
                            )
                            # save_rgb_image(wrist_axis_tensor[0], "tmp_dir/wrist_non_scaled_axis_tensor.png")
                            item['observation.wrist_image'] = torch.cat([wrist_img, wrist_axis_tensor], dim=1)
                        except:
                            pass
                item['observation.image'] = torch.cat([img, axis_tensor], dim=1)
                # save_rgb_image(axis_tensor[0], "tmp_dir/axis_tensor.png")
                # save_rgb_image(item['observation.image'][0], "tmp_dir/robot_image.png")
            elif self.config.visual_cue_mode == "plucker_concat" or self.config.visual_cue_mode == "plucker":
                intrinsic_tensor = intrinsic_matrix.unsqueeze(0).expand(img.shape[0], -1, -1).cuda()
                extrinsic_tensor = extrinsic_matrix.unsqueeze(0).expand(img.shape[0], -1, -1).cuda()
                plucker_data = self.plucker_embedder(intrinsic_tensor, extrinsic_tensor)
                plucker_tensor = einops.rearrange(plucker_data['plucker'], 's h w c -> s c h w')
                item['observation.image'] = torch.cat([img, plucker_tensor], dim=1)
                try:
                    wrist_img = item['observation.wrist_image']
                    wrist_intrinsic_matrix = item['observation.cam_info']['wrist_intrinsic_matrix'].to(torch.float32).squeeze(0)
                    wrist_extrinsic_matrix = item['observation.cam_info']['wrist_extrinsic_matrix'].to(torch.float32).squeeze(0)
                    wrist_plucker_extrinsic_matrix = remove_extrinsic_camera_axis_correction(wrist_extrinsic_matrix)
                    wrist_intrinsic_tensor = wrist_intrinsic_matrix.unsqueeze(0).expand(wrist_img.shape[0], -1, -1).cuda()
                    wrist_extrinsic_tensor = wrist_extrinsic_matrix.unsqueeze(0).expand(wrist_img.shape[0], -1, -1).cuda()
                    wrist_plucker_data = self.plucker_embedder(wrist_intrinsic_tensor, wrist_extrinsic_tensor)
                    wrist_plucker_tensor = einops.rearrange(wrist_plucker_data['plucker'], 's h w c -> s c h w')
                    # save_rgb_image(wrist_axis_tensor[0], "tmp_dir/wrist_non_scaled_axis_tensor.png")
                    item['observation.wrist_image'] = torch.cat([wrist_img, wrist_plucker_tensor], dim=1)
                except:
                    pass
        except Exception as e:
            print(e)
        return item

def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLAConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device if self.config.device is not None else "auto",
            visual_cue_mode=self.config.visual_cue_mode,
            knowledge_insulation=self.config.knowledge_insulation,
            vision_encoder_path=self.config.vision_encoder_path,
            aux_vision_encoder_path=self.config.aux_vision_encoder_path,
        )
        self.postfix_tokenizer = None
        if self.config.knowledge_insulation:
            if self.config.ki_objective == "fast":
                self.postfix_tokenizer = FASTActionTokenizer(
                    text_tokenizer=self.vlm_with_expert.processor.tokenizer,
                    vocab_size=self.vlm_with_expert.config.text_config.vocab_size,
                    tokenizer_path=self.config.ki_fast_tokenizer_path,
                    skip_tokens=self.config.ki_fast_skip_tokens,
                    max_tokens=self.config.ki_max_tokens,
                )
            else:
                self.postfix_tokenizer = LanguageActionTokenizer(
                    text_tokenizer=self.vlm_with_expert.processor.tokenizer,
                    translation_dims=self.config.lap_translation_dims,
                    rotation_dims=self.config.lap_rotation_dims,
                    gripper_dim=self.config.lap_gripper_dim,
                    gripper_close_is_positive=self.config.lap_gripper_close_is_positive,
                    include_rotation=self.config.lap_include_rotation,
                    style=self.config.lap_style,
                    translation_thresholds=self.config.lap_translation_thresholds,
                    rotation_thresholds=self.config.lap_rotation_thresholds,
                    idle_threshold=self.config.lap_idle_threshold,
                    cm_per_unit=self.config.lap_cm_per_unit,
                    max_tokens=self.config.ki_max_tokens,
                )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def set_action_stats(self, mean, std):
        """Give the model the action mean/std so the LAP objective can undo normalization.

        Kept as plain tensors rather than buffers: they are training-side metadata for building
        labels, and registering them would change every checkpoint's state_dict.
        """
        self._action_mean = torch.as_tensor(mean, dtype=torch.float32)
        self._action_std = torch.as_tensor(std, dtype=torch.float32)

    def unnormalize_actions(self, actions: Tensor) -> Tensor:
        mean = getattr(self, "_action_mean", None)
        std = getattr(self, "_action_std", None)
        if mean is None or std is None:
            raise RuntimeError(
                "The LAP objective needs the action mean/std to rebuild the raw command before "
                "describing it in words. Call `policy.model.set_action_stats(...)` after building "
                "the policy (the training script does this from `ds_meta.stats['action']`)."
            )
        dim = actions.shape[-1]
        mean = mean[:dim].to(device=actions.device, dtype=actions.dtype)
        std = std[:dim].to(device=actions.device, dtype=actions.dtype)
        return actions * std + mean

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state: torch.Tensor = None,
        postfix_tokens: torch.Tensor | None = None,
        postfix_pad_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        if state is not None:
            state_emb = self.state_proj(state)
            state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
            embs.append(state_emb)
            bsize = state_emb.shape[0]
            device = state_emb.device

            states_seq_len = state_emb.shape[1]
            state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1] * (states_seq_len)
        else:
            # The VQA objective asks the VLM where the gripper is; feeding the state in would be
            # handing it the answer.
            bsize = lang_emb.shape[0]
            device = lang_emb.device
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        if postfix_tokens is not None:
            # Knowledge insulation: the FAST-tokenized action chunk, appended to the VLM stream as
            # ordinary language tokens. `att_masks` 1 per token makes the block causal, so token t
            # is predicted from everything before it and never from itself. The action expert is
            # cut off from this block by `forward()` (it holds the ground-truth actions).
            postfix_emb = self.vlm_with_expert.embed_language_tokens(postfix_tokens)
            postfix_emb = postfix_emb * math.sqrt(postfix_emb.shape[-1])
            embs = torch.cat([embs, postfix_emb.to(dtype=embs.dtype)], dim=1)
            pad_masks = torch.cat([pad_masks, postfix_pad_masks], dim=1)
            att_masks = torch.cat(
                [att_masks, torch.ones_like(postfix_pad_masks, dtype=att_masks.dtype)], dim=1
            )

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
        fast_postfix: dict | None = None,
    ) -> tuple[Tensor, dict]:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors).

        With `fast_postfix` (knowledge insulation) the returned dict also carries the postfix
        token cross-entropy, which is the only gradient the VLM receives.
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        postfix_tokens = fast_postfix["tokens"] if fast_postfix is not None else None
        postfix_pad_masks = fast_postfix["pad_masks"] if fast_postfix is not None else None
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
            postfix_tokens=postfix_tokens,
            postfix_pad_masks=postfix_pad_masks,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        if fast_postfix is not None:
            prefix_len = prefix_pad_masks.shape[1]
            postfix_len = postfix_tokens.shape[1]
            core_len = prefix_len - postfix_len

            # The action expert must not see the ground-truth action tokens. The cumulative
            # `att_masks` rule would let it (the postfix sits before the suffix), so the block is
            # cleared explicitly. This also covers the cross-attention layers, which slice their
            # expert mask out of this same tensor.
            att_2d_masks[:, prefix_len:, core_len:prefix_len] = False

            # Give the suffix the position ids it would have had without a postfix, so the action
            # expert's RoPE geometry is bit-identical to a run without knowledge insulation. The
            # postfix and the suffix therefore share a position range, which is harmless: the two
            # blocks never attend to each other.
            core_offset = prefix_pad_masks[:, :core_len].sum(dim=1, keepdim=True)
            suffix_position_ids = core_offset + torch.cumsum(suffix_pad_masks, dim=1) - 1
            position_ids = torch.cat([position_ids[:, :prefix_len], suffix_position_ids], dim=1)

        (prefix_out, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")

        aux = {}
        if fast_postfix is not None:
            aux.update(self.postfix_token_loss(prefix_out, fast_postfix, core_len, prefix_len))
        return losses, aux

    def postfix_token_loss(self, prefix_out, fast_postfix, core_len, prefix_len) -> dict:
        """Next-token cross-entropy over the postfix — the VLM's own training signal."""
        loss_masks = fast_postfix["loss_masks"]
        # Hidden state at t predicts token t+1, so the postfix targets are read off the positions
        # shifted one to the left. Position `core_len - 1` is the state token, always unpadded.
        hidden = prefix_out[:, core_len - 1 : prefix_len - 1]
        # Only the supervised positions reach the LM head: the head is (hidden x 49280) and
        # materialising logits for the prompt and the padding as well would triple the memory.
        selected = hidden[loss_masks]
        targets = fast_postfix["tokens"][loss_masks]
        empty = selected.numel() == 0
        if empty:
            # Every sample in this batch was dropped (all chunks ran past their episode end).
            # Keep the LM head in the graph with a zeroed loss rather than returning NaN.
            selected = hidden[:, 0]
            targets = fast_postfix["tokens"][:, 0]
        logits = self.vlm_with_expert.lm_logits(selected).float()
        ce = F.cross_entropy(logits, targets)
        if empty:
            ce = ce * 0.0
        with torch.no_grad():
            correct = logits.argmax(dim=-1) == targets
            accuracy = correct.float().mean()
            content_accuracy = None
            content_masks = fast_postfix.get("content_masks")
            if content_masks is not None and not empty:
                # Restricted to the direction and magnitude words: the rest of a LAP sentence is
                # scaffolding the model learns immediately, and averaging it in hides the signal.
                content = content_masks[loss_masks]
                if content.any():
                    content_accuracy = correct[content].float().mean().item()
        return {
            "token_ce_loss": ce,
            "token_accuracy": accuracy.item(),
            **({} if content_accuracy is None else {"token_content_accuracy": content_accuracy}),
            "postfix_len": float(fast_postfix["tokens"].shape[1]),
            **fast_postfix["stats"],
        }

    def vqa_state_loss(self, images, img_masks, vqa: dict) -> dict:
        """Answer a question about the gripper from the image alone, through the whole VLM.

        Unlike every other auxiliary term here, which stops at the vision tower, this runs
        vision -> connector -> text layers -> lm_head, so the gradient reaches all of it. Under
        knowledge insulation that is the point: the flow-matching loss is cut off from the VLM and
        the LAP token objective saturates, so without something like this the VLM stops learning.

        The action expert is not involved at all -- `inputs_embeds=[prefix, None]` with
        `fill_kv_cache=True` keeps every layer on the plain self-attention path.
        """
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            vqa["question_tokens"],
            vqa["question_masks"],
            state=None,
            postfix_tokens=vqa["tokens"],
            postfix_pad_masks=vqa["pad_masks"],
        )
        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        (prefix_out, _), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            fill_kv_cache=True,
        )
        prefix_len = prefix_embs.shape[1]
        core_len = prefix_len - vqa["tokens"].shape[1]
        return self.postfix_token_loss(prefix_out, vqa, core_len, prefix_len)

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        num_steps = self.config.num_steps
        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t
