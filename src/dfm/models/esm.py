import attr
import torch
from torch import nn
from torch.nn import functional as F
from dfm.generative_modeling import (
    TransitionModelWithEmbedding,
    TransitionModel,
    MaskedModelLogitFormatter,
)
from esm.utils import generation
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.models.esmc import ESMC as _ESMC
from esm.models.esm3 import ESM3 as _ESM3
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from typing import TypedDict
import numpy as np


class ESMC(TransitionModelWithEmbedding):
    """ESM-C masked language model as a TransitionModelWithEmbedding.

    Tensor Index Legend:
        S: sequence index in batch
        P: position index in sequence
        T: token/vocab dimension (OUTPUT_DIM = 64)
        D: embedding dimension (EMB_DIM = 960)
    """

    OUTPUT_DIM = 64
    EMB_DIM = 960

    def __init__(self, esmc_checkpoint: str = "esmc_300m"):
        tokenizer = EsmSequenceTokenizer()
        logit_formatter = MaskedModelLogitFormatter(tokenizer, ESMC.OUTPUT_DIM)
        esmc = _ESMC.from_pretrained(esmc_checkpoint, device=torch.device("cpu")).eval()
        super().__init__(
            model=esmc, tokenizer=tokenizer, logit_formatter=logit_formatter
        )

    def differentiable_embedding(self, ohe_seq_SPT: torch.FloatTensor) -> torch.FloatTensor:
        if ohe_seq_SPT.shape[-1] < self.OUTPUT_DIM:
            ohe_seq_SPT = F.pad(ohe_seq_SPT, (0, self.OUTPUT_DIM - ohe_seq_SPT.shape[-1]))
        x_SPD = ohe_seq_SPT @ self.model.embed.weight
        sequence_id = ohe_seq_SPT.argmax(-1) != self.tokenizer.pad_token_id
        x_SPD, _, _ = self.model.transformer(x_SPD, sequence_id=sequence_id)
        return x_SPD

    def embedding_to_outputs(self, embedding_SPD: torch.FloatTensor) -> torch.FloatTensor:
        return self.model.sequence_head(embedding_SPD)

    def format_raw_to_logits(
        self, raw_output, seq_SP: torch.LongTensor, **kwargs
    ) -> torch.FloatTensor:
        logits_SPT = raw_output.sequence_logits.float()
        return self.logit_formatter(logits_SPT, seq_SP)




# TODO[pi] implement ESM3IF as a structure-conditioned TransitionModel
class ESM3IF(TransitionModel):
    OUTPUT_DIM = 33

    class StructureCondition(TypedDict):
        coords_RAX: np.array

    def __init__(self, esm3_checkpoint: str = "esm3-open"):
        tokenizer = EsmSequenceTokenizer()
        logit_formatter = MaskedModelLogitFormatter(tokenizer, ESM3IF.OUTPUT_DIM)
        esmc = _ESM3.from_pretrained(esm3_checkpoint).eval()
        super().__init__(
            model=esmc, tokenizer=tokenizer, logit_formatter=logit_formatter
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
