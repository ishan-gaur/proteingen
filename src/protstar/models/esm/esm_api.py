from __future__ import annotations

import numpy as np
import torch
from torch import nn

from protstar.generative_modeling import (
    GenerativeModel,
    MaskedModelLogitFormatter,
)
from esm.sdk.api import (
    ESMProtein,
    ESMProteinTensor,
    ESMProteinError,
    LogitsConfig,
)
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer


class ESMForgeAPI(GenerativeModel):
    """ESM model accessed via the EvolutionaryScale Forge API.

    Wraps a Forge inference client to provide the same ``get_log_probs``
    interface as the local ESM wrappers. No local weights needed — inference
    happens remotely.

    Automatically selects the right client (ESM3 vs ESMC) based on the
    model name. Structure conditioning is supported for ESM3 models only.

    Limitations vs local models:
        - No gradient access (no ``embed``, no TAG guidance)
        - No LoRA / fine-tuning
        - No checkpointing
        - Batched inference loops sequentially over the API

    Example::

        import os
        model = ESMForgeAPI("esmc-6b-2024-12", token=os.environ["FORGE_TOKEN"])
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
        tokenizer = EsmSequenceTokenizer()
        logit_formatter = MaskedModelLogitFormatter(tokenizer, self.OUTPUT_DIM)
        # GenerativeModel expects an nn.Module for self.model; use an empty module
        # since inference is remote. We override forward() to use self.client.
        super().__init__(
            model=nn.Module(), tokenizer=tokenizer, logit_formatter=logit_formatter
        )
        self._model_name = model_name
        self._is_esm3 = model_name.startswith("esm3")

        if self._is_esm3:
            from esm.sdk.forge import ESM3ForgeInferenceClient

            self.client = ESM3ForgeInferenceClient(
                model=model_name, url=url, token=token
            )
        else:
            from esm.sdk.forge import ESMCForgeInferenceClient

            self.client = ESMCForgeInferenceClient(
                model=model_name, url=url, token=token
            )

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def _call_logits(
        self, seq_1d: torch.LongTensor, **tensor_kwargs
    ) -> torch.FloatTensor:
        """Call the Forge logits endpoint for a single (unpadded) sequence.

        Args:
            seq_1d: 1-D token IDs (L,) including CLS/EOS, no padding.
            **tensor_kwargs: Extra ESMProteinTensor fields (e.g. structure, coordinates).

        Returns:
            Logits tensor of shape (L, OUTPUT_DIM).
        """
        input_tensor = ESMProteinTensor(sequence=seq_1d, **tensor_kwargs)
        output = self.client.logits(input_tensor, LogitsConfig(sequence=True))
        if isinstance(output, ESMProteinError):
            raise RuntimeError(
                f"Forge API error ({output.error_code}): {output.error_msg}"
            )
        assert output.logits is not None and output.logits.sequence is not None
        logits = output.logits.sequence.float()
        if logits.dim() == 3 and logits.shape[0] == 1:
            logits = logits.squeeze(0)
        return logits

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

            tensor_kwargs = {}
            if "structure_tokens" in kwargs:
                tensor_kwargs["structure"] = kwargs["structure_tokens"][i]
            if "coordinates" in kwargs:
                tensor_kwargs["coordinates"] = kwargs["coordinates"][i]

            logits = self._call_logits(seq_unpadded, **tensor_kwargs)
            L_actual = logits.shape[0]

            if L_actual < P:
                padding = torch.zeros(
                    P - L_actual, logits.shape[-1], dtype=logits.dtype
                )
                logits = torch.cat([logits, padding], dim=0)
            all_logits.append(logits)

        return torch.stack(all_logits)

    # ── Structure conditioning (ESM3 only) ───────────────────────────────

    def preprocess_observations(self, observations: dict) -> dict:
        """Encode structure via the Forge API (remote VQ-VAE).

        Only supported for ESM3 models.

        Args:
            observations: {"coords_RAX": (L, 37, 3) tensor or np.array}

        Returns:
            {"structure_tokens": (L+2,), "coordinates": (L+2, 37, 3)} with BOS/EOS.
        """
        assert self._is_esm3, (
            f"Structure conditioning not supported for ESMC model '{self._model_name}'"
        )
        coords = observations["coords_RAX"]
        if isinstance(coords, torch.Tensor):
            coords = coords.float()
        elif isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords).float()
        protein = ESMProtein(coordinates=coords)
        encoded = self.client.encode(protein)
        if isinstance(encoded, ESMProteinError):
            raise RuntimeError(
                f"Forge encode error ({encoded.error_code}): {encoded.error_msg}"
            )
        return {
            "structure_tokens": encoded.structure,
            "coordinates": encoded.coordinates,
        }

    def collate_observations(
        self, seq_SP: torch.LongTensor, observations: dict
    ) -> dict:
        B = seq_SP.shape[0]
        return {
            "structure_tokens": observations["structure_tokens"]
            .unsqueeze(0)
            .expand(B, -1),
            "coordinates": observations["coordinates"]
            .unsqueeze(0)
            .expand(B, -1, -1, -1),
        }

    # ── Unsupported operations ───────────────────────────────────────────

    def _save_args(self) -> dict:
        raise NotImplementedError("API models don't support checkpointing")

    def apply_lora(self, **kwargs) -> None:
        raise NotImplementedError("LoRA is not supported for API models")

    def save_lora(self, path) -> None:
        raise NotImplementedError("LoRA is not supported for API models")

    def load_lora(self, path) -> None:
        raise NotImplementedError("LoRA is not supported for API models")
