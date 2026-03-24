from __future__ import annotations

from typing import TypedDict

import attr
import numpy as np
import torch

from proteingen.generative_modeling import (
    TransitionModel,
    MaskedModelLogitFormatter,
)
from esm.models.esm3 import ESM3 as _ESM3
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.utils import generation


# TODO[pi] implement ESM3IF as a structure-conditioned TransitionModel
class ESM3IF(TransitionModel):
    OUTPUT_DIM = 33

    class StructureCondition(TypedDict):
        coords_RAX: np.array

    def __init__(self, esm3_checkpoint: str = "esm3-open"):
        tokenizer = EsmSequenceTokenizer()
        logit_formatter = MaskedModelLogitFormatter(tokenizer, ESM3IF.OUTPUT_DIM)
        # Load on CPU first to avoid accidental CUDA OOM at construction time.
        esm3 = (
            _ESM3.from_pretrained(esm3_checkpoint, device=torch.device("cpu"))
            .float()
            .eval()
        )
        super().__init__(
            model=esm3, tokenizer=tokenizer, logit_formatter=logit_formatter
        )

    def preprocess_observations(self, observations: StructureCondition) -> dict:
        """Encode structure once (expensive). Cached by set_condition_."""
        coords = observations["coords_RAX"]
        encoded = self.model.encode(ESMProtein(coordinates=coords))
        return {"encoded_tokens": encoded}

    def collate_observations(self, seq_SP: torch.LongTensor, observations: dict):
        """Replicate cached encoded tokens for the batch."""
        template = observations["encoded_tokens"]
        batch_size = seq_SP.shape[0]
        return {"input_tokens": [template] * batch_size}

    def forward(self, seq_SP, input_tokens):
        """Forward pass of ESM3 as inverse folding model.

        Args:
            model: ESM3 model
            xt: partially masked sequence tokens (B, D)
            input_tokens: list of ESMProteinTensor (batch templates with structure)

        Returns:
            logits (B, D, S)
        """

        tokenizers = self.model.tokenizers
        sampled_tokens = [attr.evolve(tokens) for tokens in input_tokens]
        device = sampled_tokens[0].device

        sequence_lengths = [len(tokens) for tokens in sampled_tokens]
        batched_tokens = generation._stack_protein_tensors(
            sampled_tokens, sequence_lengths, tokenizers, device
        )

        xt = seq_SP
        xt_copy = xt.clone()
        xt_copy[..., 0] = tokenizers.sequence.cls_token_id
        xt_copy[..., -1] = tokenizers.sequence.eos_token_id
        setattr(batched_tokens, "sequence", xt_copy)

        forward_output = self.model.logits(batched_tokens, LogitsConfig(sequence=True))
        logits = forward_output.logits.sequence[
            ..., : self.model.tokenizers.sequence.vocab_size
        ]
        return logits
