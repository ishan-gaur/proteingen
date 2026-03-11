import attr
import torch
from torch import nn
from dfm.generative_modeling import (
    TransitionModel,
    MaskedModelLogitFormatter,
    LogitFormatter,
)
from dfm.predictive_modeling import PreTrainedEmbeddingModel
from esm.utils import generation
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.models.esmc import ESMC as _ESMC
from esm.models.esmc import ESMCOutput
from esm.models.esm3 import ESM3 as _ESM3
from esm.models.esm3 import ESMOutput
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from transformers import PreTrainedTokenizerBase
from typing import TypedDict
import numpy as np


class ESMC(TransitionModel, PreTrainedEmbeddingModel):
    """ESM-C masked language model wrapped as a TransitionModel.

    Also serves as a frozen embedding extractor (via ``PreTrainedEmbeddingModel``).
    Use ``embed(seq_SP)`` for mean-pooled embeddings from token IDs, or
    ``forward_ohe(ohe_seq_SPT)`` for a differentiable path through the
    embedding layer (for TAG guidance).

    Tensor Index Legend:
        S: sequence index in batch
        P: position index in sequence
        T: token/vocab dimension
        D: embedding dimension (960)
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

    def forward(self, seq_SP: torch.LongTensor, **kwargs) -> ESMCOutput:
        """Standard forward pass from token IDs (for TransitionModel use)."""
        return self.model(seq_SP, **kwargs)

    def forward_ohe(
        self, ohe_seq_SPT: torch.FloatTensor
    ) -> ESMCOutput:
        """Differentiable forward pass from one-hot encoded input.

        Uses ``ohe @ embedding_weights`` instead of ``embedding[token_ids]``
        so gradients flow through the embedding step (needed for TAG).

        Returns the same ``ESMCOutput`` as ``forward()``.
        """
        x_SPD = ohe_seq_SPT @ self.model.embed.weight  # (S, P, D)
        seq_SP = ohe_seq_SPT.argmax(dim=-1)
        sequence_id = seq_SP != self.tokenizer.pad_token_id

        x, _, hiddens = self.model.transformer(x_SPD, sequence_id=sequence_id)
        hiddens = torch.stack(hiddens, dim=0)
        sequence_logits = self.model.sequence_head(x)
        return ESMCOutput(
            sequence_logits=sequence_logits, embeddings=x, hidden_states=hiddens
        )

    def format_raw_to_logits(
        self, model_output: ESMCOutput, seq_SP: torch.LongTensor
    ) -> torch.FloatTensor:
        logits_SPT = model_output.sequence_logits.float()
        logits_SPT = self.logit_formatter(logits_SPT, seq_SP)
        return logits_SPT




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
