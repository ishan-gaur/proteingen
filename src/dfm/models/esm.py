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


class ESMCEmbedding(PreTrainedEmbeddingModel):
    """Frozen ESMC encoder that produces mean-pooled sequence embeddings.

    Wraps ESMC to extract per-token embeddings from the transformer trunk,
    then mean-pools over non-special positions to produce a single vector
    per sequence.  All ESMC parameters are frozen — this class is intended
    as a fixed feature extractor for downstream probes (e.g. ``LinearProbe``).

    Tensor Index Legend:
        S: batch (sequence) index
        P: position index in sequence
        D: embedding dimension (960 for esmc_300m)
        T: token/vocab dimension (for one-hot inputs)
    """

    EMB_DIM = 960

    def __init__(self, esmc_checkpoint: str = "esmc_300m"):
        super().__init__()
        self.tokenizer = EsmSequenceTokenizer()
        self._esmc = _ESMC.from_pretrained(
            esmc_checkpoint, device=torch.device("cpu")
        ).eval()
        # Freeze all parameters — pure feature extractor
        for p in self._esmc.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, seq_SP: torch.LongTensor) -> torch.FloatTensor:
        """Extract mean-pooled embeddings from token IDs.

        Args:
            seq_SP: Token IDs, shape (S, P).  Should include CLS/EOS
                framing tokens as produced by ``EsmSequenceTokenizer``.

        Returns:
            Mean-pooled embeddings, shape (S, D).
        """
        output: ESMCOutput = self._esmc(seq_SP)
        embeddings_SPD = output.embeddings  # (S, P, D)

        # Mask out special tokens (CLS, EOS, PAD) before pooling
        pad_id = self.tokenizer.pad_token_id
        cls_id = self.tokenizer.cls_token_id
        eos_id = self.tokenizer.eos_token_id
        special = {pad_id, cls_id, eos_id}
        mask_SP = torch.ones_like(seq_SP, dtype=torch.bool)
        for sid in special:
            if sid is not None:
                mask_SP = mask_SP & (seq_SP != sid)

        # Mean pool over non-special positions
        mask_SP1 = mask_SP.unsqueeze(-1).float()  # (S, P, 1)
        pooled_SD = (embeddings_SPD * mask_SP1).sum(dim=1) / mask_SP1.sum(dim=1).clamp(
            min=1
        )
        return pooled_SD

    @torch.no_grad()
    def forward_ohe(
        self, ohe_seq_SPT: torch.FloatTensor
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Extract log probs and embeddings from one-hot encoded input.

        Converts one-hot to token IDs via argmax, runs the model, and
        returns both sequence logits (as log probs) and pooled embeddings.

        Args:
            ohe_seq_SPT: One-hot encoded tokens, shape (S, P, T).

        Returns:
            Tuple of (log_probs (S, P, T), pooled_embeddings (S, D)).
        """
        seq_SP = ohe_seq_SPT.argmax(dim=-1)
        output: ESMCOutput = self._esmc(seq_SP)
        log_probs = torch.log_softmax(output.sequence_logits.float(), dim=-1)

        # Pool embeddings the same way as forward()
        embeddings_SPD = output.embeddings
        pad_id = self.tokenizer.pad_token_id
        cls_id = self.tokenizer.cls_token_id
        eos_id = self.tokenizer.eos_token_id
        special = {pad_id, cls_id, eos_id}
        mask_SP = torch.ones(seq_SP.shape, dtype=torch.bool, device=seq_SP.device)
        for sid in special:
            if sid is not None:
                mask_SP = mask_SP & (seq_SP != sid)
        mask_SP1 = mask_SP.unsqueeze(-1).float()
        pooled_SD = (embeddings_SPD * mask_SP1).sum(dim=1) / mask_SP1.sum(dim=1).clamp(
            min=1
        )

        return log_probs, pooled_SD

    def tokenize(self, sequences: list[str]) -> torch.LongTensor:
        """Tokenize raw AA strings to token IDs (with CLS/EOS framing).

        Convenience method for use in data pipelines.

        Args:
            sequences: List of amino acid strings.

        Returns:
            Token IDs tensor, shape (S, P).
        """
        encoded = self.tokenizer(sequences, padding=True, return_tensors="pt")
        return encoded["input_ids"]


class ESMC(TransitionModel):
    """ESM-C masked language model wrapped as a TransitionModel.

    Tensor Index Legend:
        S: sequence index in batch
        P: position index in sequence
        T: token/vocab dimension
    """

    OUTPUT_DIM = 64

    def __init__(self, esmc_checkpoint: str = "esmc_300m"):
        tokenizer = EsmSequenceTokenizer()
        logit_formatter = MaskedModelLogitFormatter(tokenizer, ESMC.OUTPUT_DIM)
        esmc = _ESMC.from_pretrained(esmc_checkpoint).eval()
        super().__init__(
            model=esmc, tokenizer=tokenizer, logit_formatter=logit_formatter
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
