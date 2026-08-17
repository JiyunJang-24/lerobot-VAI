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

"""LAP-style language actions: describe an action chunk as an English sentence.

The alternative to `fast_action_tokenizer.py`. Where FAST compresses the chunk into ~170 opaque
ids, this collapses it into one sentence of ~20 ordinary English tokens:

    "move forward moderately, move right slightly, close gripper"

The idea is LAP's (`lihzha/lap`, `policies/transforms/action_text.py`): a pretrained VLM already
knows what "move forward" means, so supervising it in language reuses that prior instead of
teaching it a new code from scratch.

Two things are deliberately NOT copied from LAP:

* **No centimetres by default.** LAP labels DROID, whose actions are Cartesian velocities with a
  known scale. This corpus's actions are OSC_POSE servo targets in [-1, 1]: the same command
  produces a different displacement on each arm (measured achieved motion per summed action unit:
  1.04 cm on ur5e, 0.43-0.63 on iiwa, 0.32-0.65 on panda, whose z axis does not correlate with the
  command at all). A fixed cm scale would therefore print a different lie for each embodiment —
  in a cross-embodiment study, exactly the wrong error to introduce. Magnitudes are bucketed in
  action units instead, against corpus percentiles. `lap_cm_per_unit > 0` restores centimetres for
  anyone who wants them.
* **The gripper sign is flipped.** Here +1 closes (robosuite `GRIP`), and the sign flips to +1 at
  the grasp and back to -1 at the release in every PnP episode. LAP's code reads >= 0.5 as "open".
"""

import numpy as np
import torch

_AXIS_WORDS = (
    ("move forward", "move back"),  # +x, -x
    ("move left", "move right"),  # +y, -y
    ("move up", "move down"),  # +z, -z
)
# Direction and magnitude — the part of the sentence a model could get wrong while still
# producing fluent output. Encoded with a leading space too, since BPE splits " forward" and
# "forward" into different ids.
_CONTENT_WORDS = tuple(
    variant
    for word in (
        "forward", "back", "left", "right", "up", "down",
        "slightly", "moderately", "lot", "open", "close",
    )
    for variant in (word, " " + word)
)

_ROTATION_WORDS = (
    ("tilt left", "tilt right"),  # +roll, -roll
    ("tilt back", "tilt forward"),  # +pitch, -pitch
    ("rotate counterclockwise", "rotate clockwise"),  # +yaw, -yaw
)


class LanguageActionTokenizer:
    """Encodes action chunks into the LM postfix as an English sentence."""

    def __init__(
        self,
        text_tokenizer,
        translation_dims=(5, 6, 7),
        rotation_dims=(8, 9, 10),
        gripper_dim: int = 11,
        gripper_close_is_positive: bool = True,
        include_rotation: bool = True,
        style: str = "rough",
        translation_thresholds=(4.7, 10.7),
        rotation_thresholds=(0.33, 1.10),
        idle_threshold: float = 0.25,
        cm_per_unit: float = 0.0,
        max_tokens: int = 256,
    ):
        self.tokenizer = text_tokenizer
        self.translation_dims = tuple(translation_dims)
        self.rotation_dims = tuple(rotation_dims)
        self.gripper_dim = int(gripper_dim)
        self.gripper_close_is_positive = bool(gripper_close_is_positive)
        self.include_rotation = bool(include_rotation)
        self.style = style
        self.translation_thresholds = tuple(translation_thresholds)
        self.rotation_thresholds = tuple(rotation_thresholds)
        self.idle_threshold = float(idle_threshold)
        self.cm_per_unit = float(cm_per_unit)
        self.max_tokens = int(max_tokens)

        self.prompt_ids = list(text_tokenizer.encode("Action: ", add_special_tokens=False))
        self.eos_id = int(text_tokenizer.eos_token_id)
        pad_id = text_tokenizer.pad_token_id
        self.pad_id = int(pad_id) if pad_id is not None else 0

        # Most of a 16-token sentence is scaffolding ("move", ", ") that the model nails in a few
        # hundred steps, so plain token accuracy saturates near 0.85 while saying nothing about
        # whether the VLM knows where the arm is going. These are the tokens that carry the answer.
        self._content_ids = {
            token_id
            for word in _CONTENT_WORDS
            for token_id in text_tokenizer.encode(word, add_special_tokens=False)
        }

    def _magnitude(self, value: float, thresholds) -> str:
        if self.style == "numeric":
            if self.cm_per_unit > 0:
                return f"{value * self.cm_per_unit:.0f} cm"
            return f"{value:.1f}"
        if value < thresholds[0]:
            return "slightly"
        if value < thresholds[1]:
            return "moderately"
        return "a lot"

    def describe(self, chunk: np.ndarray) -> str:
        """One action chunk (steps, action_dim) -> one sentence.

        The chunk is summed over time first, exactly as LAP does: the sentence describes where the
        arm is asked to go over the whole horizon, not what it does at each step.
        """
        parts = []
        totals = chunk.sum(axis=0)

        for axis, dim in enumerate(self.translation_dims):
            value = float(totals[dim])
            if abs(value) < self.idle_threshold:
                continue
            word = _AXIS_WORDS[axis][0 if value > 0 else 1]
            parts.append(f"{word} {self._magnitude(abs(value), self.translation_thresholds)}")

        if self.include_rotation:
            for axis, dim in enumerate(self.rotation_dims):
                value = float(totals[dim])
                if abs(value) < self.idle_threshold:
                    continue
                word = _ROTATION_WORDS[axis][0 if value > 0 else 1]
                parts.append(f"{word} {self._magnitude(abs(value), self.rotation_thresholds)}")

        if not parts:
            parts.append("hold position")

        # The gripper is a state, not a delta: what matters is where the chunk leaves it.
        gripper = float(chunk[-1, self.gripper_dim])
        closing = gripper > 0 if self.gripper_close_is_positive else gripper <= 0
        parts.append("close gripper" if closing else "open gripper")
        return ", ".join(parts)

    @torch.no_grad()
    def encode(self, actions: torch.Tensor) -> dict:
        """Build the postfix token block for a batch of action chunks.

        Mirrors `FASTActionTokenizer.encode` so the two objectives are interchangeable downstream.
        """
        if actions.ndim != 3:
            raise ValueError(f"(batch, chunk, action_dim) expected, got {tuple(actions.shape)}")
        needed = max((*self.translation_dims, *self.rotation_dims, self.gripper_dim)) + 1
        if actions.shape[-1] < needed:
            raise ValueError(
                f"LAP action layout needs at least {needed} action dimensions, got {actions.shape[-1]}. "
                "Set --policy.lap_translation_dims/--policy.lap_rotation_dims/--policy.lap_gripper_dim "
                "for this corpus."
            )

        device = actions.device
        chunks = actions.detach().to(dtype=torch.float32, device="cpu").numpy()
        sentences = [self.describe(chunk) for chunk in chunks]
        encoded = [self.tokenizer.encode(s, add_special_tokens=False) for s in sentences]

        n_prompt = len(self.prompt_ids)
        budget = max(1, self.max_tokens - n_prompt - 1)
        lengths = [len(t) for t in encoded]
        num_truncated = sum(1 for length in lengths if length > budget)
        max_len = n_prompt + min(max(lengths), budget) + 1

        bsize = len(encoded)
        tokens = np.full((bsize, max_len), self.pad_id, dtype=np.int64)
        pad_masks = np.zeros((bsize, max_len), dtype=bool)
        loss_masks = np.zeros((bsize, max_len), dtype=bool)
        content_masks = np.zeros((bsize, max_len), dtype=bool)

        for i, ids in enumerate(encoded):
            ids = np.asarray(ids[:budget], dtype=np.int64)
            end = n_prompt + ids.size
            tokens[i, :n_prompt] = self.prompt_ids
            tokens[i, n_prompt:end] = ids
            tokens[i, end] = self.eos_id
            pad_masks[i, : end + 1] = True
            loss_masks[i, n_prompt : end + 1] = True
            content_masks[i, n_prompt:end] = np.isin(ids, list(self._content_ids))

        return {
            "tokens": torch.from_numpy(tokens).to(device=device),
            "pad_masks": torch.from_numpy(pad_masks).to(device=device),
            "loss_masks": torch.from_numpy(loss_masks).to(device=device),
            "content_masks": torch.from_numpy(content_masks).to(device=device),
            "sentences": sentences,
            "stats": {
                "lap_sentence_tokens_mean": float(np.mean(lengths)),
                "lap_sentence_tokens_max": float(np.max(lengths)),
                "lap_truncated_frac": num_truncated / max(1, bsize),
                "lap_hold_position_frac": float(np.mean([s.startswith("hold position") for s in sentences])),
            },
        }
