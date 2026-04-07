from __future__ import annotations

import torch
from torch.nn import functional as F

from proteingen.generative_modeling import (
    GenerativeModelWithEmbedding,
    MaskedModelLogitFormatter,
)
from esm.models.esmc import ESMC as _ESMC
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer


class ESMC(GenerativeModelWithEmbedding):
    """ESM-C masked language model as a GenerativeModelWithEmbedding.

    Tensor Index Legend:
        S: sequence index in batch
        P: position index in sequence
        T: token/vocab dimension (OUTPUT_DIM = 64)
        D: embedding dimension (EMB_DIM = 960 for 300m, 1152 for 600m)
    """

    OUTPUT_DIM = 64

    def __init__(self, esmc_checkpoint: str = "esmc_300m"):
        self._esmc_checkpoint = esmc_checkpoint
        tokenizer = EsmSequenceTokenizer()
        logit_formatter = MaskedModelLogitFormatter(tokenizer, ESMC.OUTPUT_DIM)
        esmc = _ESMC.from_pretrained(esmc_checkpoint, device=torch.device("cpu")).eval()
        self.EMB_DIM = esmc.embed.weight.shape[1]  # 960 for 300m, 1152 for 600m
        super().__init__(
            model=esmc, tokenizer=tokenizer, logit_formatter=logit_formatter
        )

    def _save_args(self) -> dict:
        return {"esmc_checkpoint": self._esmc_checkpoint}

    def differentiable_embedding(
        self, ohe_seq_SPT: torch.FloatTensor
    ) -> torch.FloatTensor:
        if ohe_seq_SPT.shape[-1] < self.OUTPUT_DIM:
            ohe_seq_SPT = F.pad(
                ohe_seq_SPT, (0, self.OUTPUT_DIM - ohe_seq_SPT.shape[-1])
            )
        x_SPD = ohe_seq_SPT @ self.model.embed.weight
        sequence_id = ohe_seq_SPT.argmax(-1) != self.tokenizer.pad_token_id
        x_SPD, _, _ = self.model.transformer(x_SPD, sequence_id=sequence_id)
        return x_SPD

    def embedding_to_outputs(
        self, embedding_SPD: torch.FloatTensor
    ) -> torch.FloatTensor:
        return self.model.sequence_head(embedding_SPD)

    def format_raw_to_logits(
        self, raw_output, seq_SP: torch.LongTensor, **kwargs
    ) -> torch.FloatTensor:
        logits_SPT = raw_output.sequence_logits.float()
        return self.logit_formatter(logits_SPT, seq_SP)
