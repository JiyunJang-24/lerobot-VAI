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

"""Ask the VLM where the gripper is, in words, on the visual-robust renders.

The visual-robust export draws the same instant with several different robots and stores ONE
`observation.state` for all of them, so a question about the gripper has one answer regardless of
which arm is in the picture. Answering it through the LM head is a supervised route to embodiment
invariance that runs through the WHOLE VLM -- vision tower, connector, text layers, lm_head --
rather than through a regression head bolted onto the tower. Under knowledge insulation that
matters: the flow-matching gradient never reaches the VLM, and the LAP token objective saturates by
~15k steps, after which this is the only thing still teaching it.

    Q: Where is the gripper?
    A: 12 cm forward, 3 cm right of the start, 84 cm high, turned 15 degrees left, closed

WHICH FRAME EACH QUANTITY IS IN, and why it is not uniform (measured on this export, 108 episodes):

    axis          between-episode std   within-episode std   ratio
    fingertip x        165.0 cm               11.6 cm        14.3   -> relative to the first frame
    fingertip y         20.8 cm               19.1 cm         1.1   -> relative to the first frame
    fingertip z          4.1 cm               14.9 cm         0.28  -> absolute
    yaw                 86.3 deg              12.4 deg        6.9   -> relative to the first frame
    gripper              0.1                   1.0            0.11  -> absolute

`robot0_agentview_right` is mounted on the robot base, so the picture is identical wherever
robocasa parked that base. Anything whose variance is dominated by the between-episode term is
therefore invisible to the camera, and a model asked for it can only answer the mean -- x would
plateau at 165 cm of error and yaw at 86 degrees. This project already paid for that lesson once
(CLAUDE.md section 1: a head asked for world-frame position stalled at 1.57 m).

The reference is the episode's FIRST frame, not its mean: it is causal, and it matches what the
sentence literally says ("of the start").
"""

import numpy as np
import torch


def quaternion_xyzw_to_yaw_deg(quat: np.ndarray) -> np.ndarray:
    """Yaw in degrees from xyzw quaternions, shape (..., 4)."""
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True).clip(min=1e-8)
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return np.degrees(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def wrap_deg(angle: np.ndarray) -> np.ndarray:
    """Fold an angle difference into (-180, 180]."""
    return (angle + 180.0) % 360.0 - 180.0


class VQAStateTokenizer:
    """Builds the question prefix and the answer postfix for the gripper-state question."""

    QUESTION = "Where is the gripper?"

    def __init__(
        self,
        text_tokenizer,
        position_resolution_cm: float = 1.0,
        yaw_resolution_deg: float = 5.0,
        gripper_close_is_positive: bool = True,
        max_tokens: int = 64,
    ):
        self.tokenizer = text_tokenizer
        self.position_resolution_cm = float(position_resolution_cm)
        self.yaw_resolution_deg = float(yaw_resolution_deg)
        self.gripper_close_is_positive = bool(gripper_close_is_positive)
        self.max_tokens = int(max_tokens)

        self.question_ids = list(text_tokenizer.encode(self.QUESTION + " ", add_special_tokens=False))
        self.prompt_ids = list(text_tokenizer.encode("Answer: ", add_special_tokens=False))
        self.eos_id = int(text_tokenizer.eos_token_id)
        pad_id = text_tokenizer.pad_token_id
        self.pad_id = int(pad_id) if pad_id is not None else 0

        # Only the numbers and the direction words are worth scoring; "cm", "of the start" and the
        # commas are scaffolding the model gets right immediately and would otherwise inflate the
        # accuracy metric the same way it did for the LAP sentence.
        content = ["forward", "back", "left", "right", "high", "degrees", "closed", "open"]
        content += [str(n) for n in range(0, 200)]
        self._content_ids = {
            tid
            for word in content
            for variant in (word, " " + word)
            for tid in text_tokenizer.encode(variant, add_special_tokens=False)
        }

    def _round(self, value: float, resolution: float) -> int:
        return int(round(value / resolution) * resolution)

    def describe(self, dx_cm: float, dy_cm: float, z_cm: float, dyaw_deg: float, gripper: float) -> str:
        """One frame's state -> one sentence. dx/dy/dyaw are already relative to the first frame."""
        dx = self._round(dx_cm, self.position_resolution_cm)
        dy = self._round(dy_cm, self.position_resolution_cm)
        z = self._round(z_cm, self.position_resolution_cm)
        dyaw = self._round(dyaw_deg, self.yaw_resolution_deg)

        parts = [
            f"{abs(dx)} cm {'forward' if dx >= 0 else 'back'}",
            f"{abs(dy)} cm {'left' if dy >= 0 else 'right'} of the start",
            f"{z} cm high",
        ]
        if dyaw == 0:
            parts.append("turned 0 degrees")
        else:
            parts.append(f"turned {abs(dyaw)} degrees {'left' if dyaw > 0 else 'right'}")
        closing = gripper > 0.5 if self.gripper_close_is_positive else gripper <= 0.5
        parts.append("closed" if closing else "open")
        return ", ".join(parts)

    def sentences(self, states: np.ndarray, references: np.ndarray) -> list[str]:
        """states/references: (N, 8) rows of fingertip_xyz + quat_xyzw + gripper."""
        dx = (states[:, 0] - references[:, 0]) * 100.0
        dy = (states[:, 1] - references[:, 1]) * 100.0
        z = states[:, 2] * 100.0
        dyaw = wrap_deg(
            quaternion_xyzw_to_yaw_deg(states[:, 3:7]) - quaternion_xyzw_to_yaw_deg(references[:, 3:7])
        )
        return [
            self.describe(float(a), float(b), float(c), float(d), float(g))
            for a, b, c, d, g in zip(dx, dy, z, dyaw, states[:, 7], strict=True)
        ]

    @torch.no_grad()
    def encode(self, states: torch.Tensor, references: torch.Tensor) -> dict:
        """Question tokens, answer postfix tokens and masks for a batch of frames."""
        s = states.detach().to(dtype=torch.float32, device="cpu").numpy()
        r = references.detach().to(dtype=torch.float32, device="cpu").numpy()
        texts = self.sentences(s, r)
        encoded = [self.tokenizer.encode(t, add_special_tokens=False) for t in texts]

        n_prompt = len(self.prompt_ids)
        budget = max(1, self.max_tokens - n_prompt - 1)
        lengths = [len(e) for e in encoded]
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

        device = states.device
        question = torch.tensor(self.question_ids, dtype=torch.long, device=device)
        return {
            "question_tokens": question[None, :].expand(bsize, -1),
            "question_masks": torch.ones(bsize, len(self.question_ids), dtype=torch.bool, device=device),
            "tokens": torch.from_numpy(tokens).to(device=device),
            "pad_masks": torch.from_numpy(pad_masks).to(device=device),
            "loss_masks": torch.from_numpy(loss_masks).to(device=device),
            "content_masks": torch.from_numpy(content_masks).to(device=device),
            "sentences": texts,
            "stats": {
                "vqa_answer_tokens_mean": float(np.mean(lengths)),
                "vqa_answer_tokens_max": float(np.max(lengths)),
            },
        }


def build_episode_reference_states(dataset, state_key: str = "observation.state"):
    """First frame of every episode, keyed (dataset_index, episode_index).

    First frame rather than episode mean: the mean peeks at the future, and "of the start" in the
    sentence should mean what it says.
    """
    import pathlib

    import pandas as pd

    references = {}
    subs = getattr(dataset, "_datasets", [dataset])
    for dataset_index, sub in enumerate(subs):
        files = sorted(pathlib.Path(sub.root).glob("data/**/*.parquet"))
        if not files:
            continue
        frame = pd.concat(
            [pd.read_parquet(f, columns=["episode_index", "frame_index", state_key]) for f in files],
            ignore_index=True,
        )
        first = frame.sort_values("frame_index").groupby("episode_index", sort=False).first()
        for episode, row in first.iterrows():
            references[(dataset_index, int(episode))] = torch.as_tensor(
                np.asarray(row[state_key], dtype=np.float32), dtype=torch.float32
            )
    if not references:
        raise ValueError(f"could not build episode reference states for {state_key}")
    return references


def lookup_episode_references(references, batch, device, state_dim: int = 8) -> torch.Tensor:
    """Gather each row's episode reference state, mirroring lookup_episode_offsets."""
    episode_index = batch["episode_index"]
    episode_index = episode_index[:, -1] if episode_index.ndim == 2 else episode_index
    dataset_index = batch.get("dataset_index")
    if dataset_index is None:
        dataset_index = torch.zeros_like(episode_index)
    dataset_index = dataset_index[:, -1] if dataset_index.ndim == 2 else dataset_index

    zero = torch.zeros(state_dim, dtype=torch.float32)
    rows = [
        references.get((int(d), int(e)), zero)
        for d, e in zip(dataset_index.tolist(), episode_index.tolist(), strict=False)
    ]
    return torch.stack(rows).to(device=device)
