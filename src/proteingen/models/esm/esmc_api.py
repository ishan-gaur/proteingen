from __future__ import annotations

import torch
from torch import nn

from proteingen.generative_modeling import (
    TransitionModel,
    MaskedModelLogitFormatter,
)
from esm.sdk.api import (
    ESMProteinTensor,
    ESMProteinError,
    LogitsConfig,
)
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer


class ESMCAPI(TransitionModel):
    """ESM-C model accessed via the EvolutionaryScale Forge API.

    Wraps ``ESMCForgeInferenceClient`` to provide the same ``get_log_probs``
    interface as the local ``ESMC`` wrapper. No local weights are needed —
    inference happens remotely. Useful for accessing larger ESMC variants
    (e.g. ``esmc-6b-2024-12``) that cannot be run locally.

    Limitations vs local ``ESMC``:
        - No gradient access (no ``embed``, no TAG guidance)
        - No LoRA / fine-tuning
        - No checkpointing
        - Batched inference loops sequentially over the API

    Example::

        import os
        model = ESMCAPI("esmc-6b-2024-12", token=os.environ["ESM_API_KEY"])
        log_probs = model.get_log_probs_from_string(["ACDEF"])

    Tensor Index Legend:
        S: sequence index in batch
        P: position index in sequence
        T: token/vocab dimension (OUTPUT_DIM = 64)
    """

    OUTPUT_DIM = 64

    def __init__(
        self,
        model_name: str,
        token: str,
        url: str = "https://forge.evolutionaryscale.ai",
    ):
        from esm.sdk.forge import ESMCForgeInferenceClient

        tokenizer = EsmSequenceTokenizer()
        logit_formatter = MaskedModelLogitFormatter(tokenizer, ESMCAPI.OUTPUT_DIM)
        # TransitionModel expects an nn.Module for self.model; use an empty module
        # since inference is remote. We override forward() to use self.client.
        super().__init__(
            model=nn.Module(), tokenizer=tokenizer, logit_formatter=logit_formatter
        )
        self.client = ESMCForgeInferenceClient(model=model_name, url=url, token=token)
        self._model_name = model_name

    @property
    def device(self) -> torch.device:
        # API returns CPU tensors; no local parameters to track
        return torch.device("cpu")

    def _call_logits(self, seq_1d: torch.LongTensor) -> torch.FloatTensor:
        """Call the Forge logits endpoint for a single (unpadded) sequence.

        Args:
            seq_1d: 1-D token IDs (L,) including CLS/EOS, no padding.

        Returns:
            Logits tensor of shape (L, OUTPUT_DIM).
        """
        input_tensor = ESMProteinTensor(sequence=seq_1d)
        output = self.client.logits(input_tensor, LogitsConfig(sequence=True))
        if isinstance(output, ESMProteinError):
            raise RuntimeError(
                f"Forge API error ({output.error_code}): {output.error_msg}"
            )
        assert output.logits is not None and output.logits.sequence is not None
        logits = output.logits.sequence.float()
        # logits is (1, L, 64) from the API; squeeze batch dim
        if logits.dim() == 3 and logits.shape[0] == 1:
            logits = logits.squeeze(0)
        return logits  # (L, OUTPUT_DIM)

    def forward(self, seq_SP: torch.LongTensor, **kwargs) -> torch.FloatTensor:
        """Forward pass via the Forge API.

        Loops over the batch dimension, stripping padding from each sequence
        before sending to the API, then re-pads results to the max length.

        Returns:
            Logits tensor of shape (S, P, OUTPUT_DIM).
        """
        S, P = seq_SP.shape
        pad_id = self.tokenizer.pad_token_id

        all_logits = []
        for i in range(S):
            seq = seq_SP[i]
            non_pad = seq != pad_id
            seq_unpadded = seq[non_pad]

            logits = self._call_logits(seq_unpadded)
            L_actual = logits.shape[0]

            # Pad back to max sequence length
            if L_actual < P:
                padding = torch.zeros(
                    P - L_actual, logits.shape[-1], dtype=logits.dtype
                )
                logits = torch.cat([logits, padding], dim=0)
            all_logits.append(logits)

        return torch.stack(all_logits)  # (S, P, OUTPUT_DIM)

    # ── Unsupported operations ───────────────────────────────────────────

    def _save_args(self) -> dict:
        raise NotImplementedError("API models don't support checkpointing")

    def apply_lora(self, **kwargs) -> None:
        raise NotImplementedError("LoRA is not supported for API models")

    def save_lora(self, path) -> None:
        raise NotImplementedError("LoRA is not supported for API models")

    def load_lora(self, path) -> None:
        raise NotImplementedError("LoRA is not supported for API models")
