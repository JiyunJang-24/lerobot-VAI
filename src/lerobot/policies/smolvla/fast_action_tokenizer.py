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

"""FAST action tokenizer used by the knowledge-insulation objective.

Knowledge insulation (pi_0.5) trains the VLM to *predict* the action chunk as discrete tokens
while the flow-matching gradient is stopped before it can reach the VLM. This module builds the
LM postfix carrying those tokens.

Layout follows openpi's `src/openpi/models/tokenizer.py::FASTTokenizer`:

    postfix = "Action: " + <FAST tokens> + <eos>

FAST ids are mapped into the *tail* of the LM vocabulary, skipping the last `skip_tokens`
entries which are reserved special tokens:

    lm_id = vocab_size - 1 - skip_tokens - fast_id

`scipy` is a hard requirement of the remote-code FAST processor (it uses the DCT); without it
`AutoProcessor.from_pretrained` raises ImportError.
"""

import numpy as np
import torch


class FASTActionTokenizer:
    """Encodes action chunks into the LM postfix, and decodes them back for verification."""

    def __init__(
        self,
        text_tokenizer,
        vocab_size: int,
        tokenizer_path: str = "physical-intelligence/fast",
        skip_tokens: int = 128,
        max_tokens: int = 256,
    ):
        from transformers import AutoProcessor

        self.fast = AutoProcessor.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.vocab_size = int(vocab_size)
        self.skip_tokens = int(skip_tokens)
        self.max_tokens = int(max_tokens)

        self.prompt_ids = list(text_tokenizer.encode("Action: ", add_special_tokens=False))
        self.eos_id = int(text_tokenizer.eos_token_id)
        # Padding positions are masked out of both attention and the loss, so any in-range id works.
        pad_id = text_tokenizer.pad_token_id
        self.pad_id = int(pad_id) if pad_id is not None else 0

        self.fast_vocab_size = int(getattr(self.fast, "vocab_size", 2048))
        lowest_lm_id = self.vocab_size - 1 - self.skip_tokens - (self.fast_vocab_size - 1)
        if lowest_lm_id < 0:
            raise ValueError(
                f"LM vocabulary ({self.vocab_size}) is too small to host {self.fast_vocab_size} FAST "
                f"tokens after skipping {self.skip_tokens} special tokens."
            )
        # Contiguous slice of the LM vocabulary that action tokens live in, low id first.
        self.action_id_min = lowest_lm_id
        self.action_id_max = self.vocab_size - 1 - self.skip_tokens

    def fast_to_lm(self, fast_ids: np.ndarray) -> np.ndarray:
        return self.vocab_size - 1 - self.skip_tokens - fast_ids

    def lm_to_fast(self, lm_ids: np.ndarray) -> np.ndarray:
        return self.vocab_size - 1 - self.skip_tokens - lm_ids

    @torch.no_grad()
    def encode(self, actions: torch.Tensor) -> dict:
        """Build the postfix token block for a batch of action chunks.

        Args:
            actions: (B, chunk, action_dim) — the *unpadded*, normalized actions. Feeding the
                32-dim zero-padded vector instead roughly triples the token count for nothing.

        Returns a dict with, all on `actions.device`:
            tokens:    (B, L) int64, "Action: " + FAST ids mapped into the LM vocab + eos
            pad_masks: (B, L) bool, True on real tokens
            loss_masks:(B, L) bool, True on the tokens the CE loss is computed over
                       (the FAST ids and the eos — not the "Action: " prompt, which is an input)
            stats:     python floats for logging
        """
        if actions.ndim != 3:
            raise ValueError(f"(batch, chunk, action_dim) expected, got {tuple(actions.shape)}")

        device = actions.device
        chunks = actions.detach().to(dtype=torch.float32, device="cpu").numpy()
        fast_tokens = self.fast(chunks)

        n_prompt = len(self.prompt_ids)
        # +1 for the eos. The cap bounds the sequence the VLM has to attend over, and with it the
        # size of the logits tensor, which is (B, L, vocab) and by far the biggest thing here.
        budget = max(1, self.max_tokens - n_prompt - 1)

        lengths = [len(t) for t in fast_tokens]
        num_truncated = sum(1 for length in lengths if length > budget)
        kept = [min(length, budget) for length in lengths]
        max_len = n_prompt + max(kept) + 1

        bsize = len(fast_tokens)
        tokens = np.full((bsize, max_len), self.pad_id, dtype=np.int64)
        pad_masks = np.zeros((bsize, max_len), dtype=bool)
        loss_masks = np.zeros((bsize, max_len), dtype=bool)

        for i, toks in enumerate(fast_tokens):
            toks = np.asarray(toks, dtype=np.int64)[:budget]
            if toks.size and (toks.min() < 0 or toks.max() >= self.fast_vocab_size):
                raise ValueError(f"FAST token id out of range [0, {self.fast_vocab_size}): {toks.max()}")
            end = n_prompt + toks.size
            tokens[i, :n_prompt] = self.prompt_ids
            tokens[i, n_prompt:end] = self.fast_to_lm(toks)
            tokens[i, end] = self.eos_id
            pad_masks[i, : end + 1] = True
            loss_masks[i, n_prompt : end + 1] = True

        return {
            "tokens": torch.from_numpy(tokens).to(device=device),
            "pad_masks": torch.from_numpy(pad_masks).to(device=device),
            "loss_masks": torch.from_numpy(loss_masks).to(device=device),
            "stats": {
                "fast_tokens_mean": float(np.mean(lengths)),
                "fast_tokens_max": float(np.max(lengths)),
                "fast_truncated_frac": num_truncated / max(1, bsize),
            },
        }

    def decode(self, tokens: torch.Tensor, loss_masks: torch.Tensor, chunk_size: int, action_dim: int):
        """Inverse of `encode` — used by the smoke test to prove the mapping round-trips."""
        fast_tokens = []
        for row, mask in zip(tokens.cpu().numpy(), loss_masks.cpu().numpy(), strict=True):
            ids = row[mask]
            ids = ids[ids != self.eos_id]
            fast_tokens.append(self.lm_to_fast(ids).tolist())
        return self.fast.decode(fast_tokens, time_horizon=chunk_size, action_dim=action_dim)
