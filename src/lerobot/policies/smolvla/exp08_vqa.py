"""Experiment 08 auxiliary objectives: EEF state, EEF pixel, and two-frame motion, as VQA.

All three are the same shape -- image(s) + a question -> a short English answer scored by
cross-entropy through the LM head -- so they share one packer and differ only in how the answer
sentence is written. That matters for the comparison the experiment is built on: A/B/C/D must
differ ONLY in what the auxiliary sees, never in how it is trained.

The answers are quantised into words rather than left as numbers on purpose. A previous round of
this project measured that the direction of a motion is recoverable from images while its exact
magnitude is not, and that a model asked for both learns the direction and pins the magnitude to a
constant. Words with explicit resolution make what is being asked for legible in the label itself.
"""

from __future__ import annotations

import numpy as np
import torch

QUESTION = {
    "state": "Where is the gripper?",
    "pixel": "Where is the end effector in the image?",
    "motion": "How did the end effector move between these two observations?",
}
PROMPT = "Answer: "

# Content words a scorer should check: the claim, not the scaffolding.
_CONTENT = {
    "left", "right", "up", "down", "forward", "backward", "near", "far",
    "open", "closed", "still", "centre", "top", "bottom", "middle",
}


def _round_to(value: float, resolution: float) -> int:
    return int(np.sign(value) * np.floor(abs(value) / resolution + 0.5) * resolution)


class Exp08VQATokenizer:
    """Packs (question, answer) into the postfix format `SmolVLAPolicy.vqa_state_loss` expects."""

    def __init__(self, text_tokenizer, kind: str, max_tokens: int = 64):
        if kind not in QUESTION:
            raise ValueError(f"kind must be one of {sorted(QUESTION)}, got {kind!r}")
        self.tokenizer = text_tokenizer
        self.kind = kind
        self.max_tokens = int(max_tokens)
        self.question_ids = list(text_tokenizer.encode(QUESTION[kind] + " ",
                                                       add_special_tokens=False))
        self.prompt_ids = list(text_tokenizer.encode(PROMPT, add_special_tokens=False))
        self.eos_id = text_tokenizer.eos_token_id
        self.pad_id = text_tokenizer.pad_token_id or self.eos_id
        self._content_ids = {
            i for w in _CONTENT
            for i in text_tokenizer.encode(" " + w, add_special_tokens=False)
        }

    # ---- answer writers -------------------------------------------------------------------
    def state_sentence(self, eef_pose: np.ndarray, base: np.ndarray) -> str:
        """EEF position relative to the robot base, in centimetres, plus gripper opening.

        Base-relative rather than world: the base sits at a different place in each LIBERO suite
        (metadata/base_positions.json), so a world-frame answer would encode which suite the frame
        came from as much as where the gripper is.
        """
        d = (eef_pose[:3] - base[:3]) * 100.0
        parts = [
            f"{_round_to(d[0], 2)} cm forward" if d[0] >= 0 else f"{_round_to(-d[0], 2)} cm backward",
            f"{_round_to(d[1], 2)} cm left" if d[1] >= 0 else f"{_round_to(-d[1], 2)} cm right",
            f"{_round_to(d[2], 2)} cm up" if d[2] >= 0 else f"{_round_to(-d[2], 2)} cm down",
        ]
        return ", ".join(parts)

    def pixel_sentence(self, pixel: np.ndarray, width: int, height: int) -> str:
        """Where in the frame, as a coarse grid cell plus the raw pixel.

        Both, because the grid word is what a coarse reader can get right while the numbers keep
        the target precise -- and reporting only the grid would make the task trivially guessable
        from the strong central bias these datasets have.
        """
        u, v = float(pixel[0]), float(pixel[1])
        col = ["left", "centre", "right"][min(2, max(0, int(u / max(width, 1) * 3)))]
        row = ["top", "middle", "bottom"][min(2, max(0, int(v / max(height, 1) * 3)))]
        return f"{row} {col}, at pixel {int(round(u))} {int(round(v))}"

    def motion_sentence(self, delta: np.ndarray) -> str:
        """Cartesian displacement between two observations, in centimetres per axis.

        Axes below 1 cm are omitted rather than reported as zero: naming every axis every time is
        exactly the failure mode measured earlier, where a model asserted motion on a stationary
        axis 79-91% of the time and its recall stopped meaning anything.
        """
        d = np.asarray(delta, dtype=np.float64)[:3] * 100.0
        words = [("forward", "backward"), ("left", "right"), ("up", "down")]
        parts = [f"{_round_to(abs(v), 1)} cm {words[i][0 if v >= 0 else 1]}"
                 for i, v in enumerate(d) if abs(v) >= 1.0]
        return ", ".join(parts) if parts else "did not move"

    # ---- packing --------------------------------------------------------------------------
    def encode(self, texts: list[str], device) -> dict:
        encoded = [self.tokenizer.encode(t, add_special_tokens=False) for t in texts]
        n_prompt = len(self.prompt_ids)
        budget = max(1, self.max_tokens - n_prompt - 1)
        lengths = [len(e) for e in encoded]
        max_len = n_prompt + min(max(lengths), budget) + 1

        size = len(encoded)
        tokens = np.full((size, max_len), self.pad_id, dtype=np.int64)
        pad_masks = np.zeros((size, max_len), dtype=bool)
        loss_masks = np.zeros((size, max_len), dtype=bool)
        content_masks = np.zeros((size, max_len), dtype=bool)
        for i, ids in enumerate(encoded):
            ids = np.asarray(ids[:budget], dtype=np.int64)
            end = n_prompt + ids.size
            tokens[i, :n_prompt] = self.prompt_ids
            tokens[i, n_prompt:end] = ids
            tokens[i, end] = self.eos_id
            pad_masks[i, : end + 1] = True
            loss_masks[i, n_prompt : end + 1] = True
            content_masks[i, n_prompt:end] = np.isin(ids, list(self._content_ids))

        question = torch.tensor(self.question_ids, dtype=torch.long, device=device)
        return {
            "question_tokens": question[None, :].expand(size, -1),
            "question_masks": torch.ones(size, len(self.question_ids), dtype=torch.bool,
                                         device=device),
            "tokens": torch.from_numpy(tokens).to(device=device),
            "pad_masks": torch.from_numpy(pad_masks).to(device=device),
            "loss_masks": torch.from_numpy(loss_masks).to(device=device),
            "content_masks": torch.from_numpy(content_masks).to(device=device),
            "sentences": texts,
            "stats": {"vqa_answer_tokens_mean": float(np.mean(lengths)),
                      "vqa_answer_tokens_max": float(np.max(lengths))},
        }
