from __future__ import annotations

from functools import partial

import einops
import numpy as np
import torch
from torch.nn import functional as F

from ...generative_modeling import (
    GenerativeModelWithEmbedding,
    MaskedModelLogitFormatter,
)
from esm.models.esm3 import ESM3 as _ESM3
from esm.sdk.api import ESMProtein
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.utils.constants import esm3 as ESM3_CONSTANTS
from esm.utils.misc import rbf
from esm.utils.structure.affine3d import build_affine3d_from_coordinates


class ESM3(GenerativeModelWithEmbedding):
    """ESM3 masked language model as a GenerativeModelWithEmbedding.

    Non-sequence tracks (structure, ss8, sasa, function, residue) default to
    padding values. Only the sequence embedding is differentiable.

    Structure conditioning: pass atom37 coordinates via ``set_condition_()``
    or ``conditioned_on()``. The structure VQ-VAE encodes them once; the
    resulting structure tokens and coordinates are used in both the
    differentiable embedding path and the full forward path.

    Example::

        model = ESM3()
        coords_RAX, wt_seq = pdb_to_atom37_and_seq("1abc.pdb")
        with model.conditioned_on({"coords_RAX": coords_RAX}):
            log_probs = model.get_log_probs(seq_SP)

    Tensor Index Legend:
        S: sequence index in batch
        P: position index in sequence
        T: token/vocab dimension (OUTPUT_DIM = 64)
        D: embedding dimension (EMB_DIM = 1536)
    """

    OUTPUT_DIM = 64

    def __init__(self, esm3_checkpoint: str = "esm3-open"):
        self._esm3_checkpoint = esm3_checkpoint
        tokenizer = EsmSequenceTokenizer()
        logit_formatter = MaskedModelLogitFormatter(tokenizer, ESM3.OUTPUT_DIM)
        esm3 = (
            _ESM3.from_pretrained(esm3_checkpoint, device=torch.device("cpu"))
            .float()
            .eval()
        )
        self.EMB_DIM = esm3.encoder.sequence_embed.weight.shape[1]  # 1536
        super().__init__(
            model=esm3, tokenizer=tokenizer, logit_formatter=logit_formatter
        )

    def _save_args(self) -> dict:
        return {"esm3_checkpoint": self._esm3_checkpoint}

    def preprocess_observations(self, observations: dict) -> dict:
        """Encode structure once via VQ-VAE (expensive).

        Args:
            observations: {"coords_RAX": (L, 37, 3) tensor or np.array}

        Returns:
            {"structure_tokens": (L+2,), "coordinates": (L+2, 37, 3)} with BOS/EOS padding.
        """
        coords = observations["coords_RAX"]
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords).float()
        protein = ESMProtein(coordinates=coords)
        with torch.no_grad():
            encoded = self.model.encode(protein)
        return {
            "structure_tokens": encoded.structure,  # (L+2,) with BOS/EOS
            "coordinates": encoded.coordinates,  # (L+2, 37, 3) with BOS/EOS
        }

    def collate_observations(
        self, seq_SP: torch.LongTensor, observations: dict
    ) -> dict:
        """Tile cached structure to match batch size."""
        B = seq_SP.shape[0]
        return {
            "structure_tokens": observations["structure_tokens"]
            .unsqueeze(0)
            .expand(B, -1),
            "coordinates": observations["coordinates"]
            .unsqueeze(0)
            .expand(B, -1, -1, -1),
        }

    def _non_sequence_embedding(
        self, seq_SP: torch.LongTensor, structure_tokens: torch.LongTensor | None = None
    ) -> torch.Tensor:
        """Non-sequence track embeddings.

        If structure_tokens is provided (from conditioning), uses them directly.
        Otherwise defaults to STRUCTURE_MASK_TOKEN with special-position overrides.
        """
        C = ESM3_CONSTANTS
        B, L = seq_SP.shape
        device = seq_SP.device

        if structure_tokens is None:
            # Default: mask everywhere, override at special sequence positions
            structure_tokens = torch.full(
                (B, L), C.STRUCTURE_MASK_TOKEN, dtype=torch.long, device=device
            )
            structure_tokens.masked_fill_(
                seq_SP == C.SEQUENCE_BOS_TOKEN, C.STRUCTURE_BOS_TOKEN
            )
            structure_tokens.masked_fill_(
                seq_SP == C.SEQUENCE_PAD_TOKEN, C.STRUCTURE_PAD_TOKEN
            )
            structure_tokens.masked_fill_(
                seq_SP == C.SEQUENCE_EOS_TOKEN, C.STRUCTURE_EOS_TOKEN
            )
            structure_tokens.masked_fill_(
                seq_SP == C.SEQUENCE_CHAINBREAK_TOKEN, C.STRUCTURE_CHAINBREAK_TOKEN
            )

        ss8_tokens = torch.full(
            (B, L), C.SS8_PAD_TOKEN, dtype=torch.long, device=device
        )
        sasa_tokens = torch.full(
            (B, L), C.SASA_PAD_TOKEN, dtype=torch.long, device=device
        )
        average_plddt = torch.ones((B, L), dtype=torch.float, device=device)
        per_res_plddt = torch.zeros((B, L), dtype=torch.float, device=device)
        function_tokens = torch.full(
            (B, L, 8), C.INTERPRO_PAD_TOKEN, dtype=torch.long, device=device
        )
        residue_annotation_tokens = torch.full(
            (B, L, 16), C.RESIDUE_PAD_TOKEN, dtype=torch.long, device=device
        )

        rbf_16_fn = partial(rbf, v_min=0.0, v_max=1.0, n_bins=16)
        enc = self.model.encoder

        plddt_embed = enc.plddt_projection(rbf_16_fn(average_plddt))
        structure_per_res_plddt = enc.structure_per_res_plddt_projection(
            rbf_16_fn(per_res_plddt)
        )
        structure_embed = enc.structure_tokens_embed(structure_tokens)
        ss8_embed = enc.ss8_embed(ss8_tokens)
        sasa_embed = enc.sasa_embed(sasa_tokens)
        function_embed = torch.cat(
            [fn(t) for fn, t in zip(enc.function_embed, function_tokens.unbind(-1))], -1
        )
        residue_embed = enc.residue_embed(
            einops.rearrange(residue_annotation_tokens, "B L N -> (B L) N")
        )
        residue_embed = einops.rearrange(residue_embed, "(B L) D -> B L D", B=B, L=L)

        return (
            plddt_embed
            + structure_per_res_plddt
            + structure_embed
            + ss8_embed
            + sasa_embed
            + function_embed
            + residue_embed
        )

    def differentiable_embedding(
        self, ohe_seq_SPT: torch.FloatTensor
    ) -> torch.FloatTensor:
        if ohe_seq_SPT.shape[-1] < self.OUTPUT_DIM:
            ohe_seq_SPT = F.pad(
                ohe_seq_SPT, (0, self.OUTPUT_DIM - ohe_seq_SPT.shape[-1])
            )

        B, L, _ = ohe_seq_SPT.shape
        device = ohe_seq_SPT.device
        seq_SP = ohe_seq_SPT.argmax(-1)

        # Differentiable sequence embedding
        seq_embed_SPD = ohe_seq_SPT @ self.model.encoder.sequence_embed.weight

        # Non-sequence track embeddings — use structure conditioning if available
        structure_tokens = None
        coordinates = None
        if self.observations is not None:
            obs = self.collate_observations(seq_SP, self.observations)
            structure_tokens = obs["structure_tokens"].to(device)
            coordinates = obs["coordinates"].to(device)

        with torch.no_grad():
            non_seq_embed_SPD = self._non_sequence_embedding(seq_SP, structure_tokens)

        x_SPD = seq_embed_SPD + non_seq_embed_SPD

        # Build affine for geometric attention
        sequence_id = seq_SP != self.tokenizer.pad_token_id
        if coordinates is not None:
            # structure_coords is (B, L, 37, 3) — ESM3 transformer expects (B, L, 3, 3)
            affine, affine_mask = build_affine3d_from_coordinates(
                coordinates[..., :3, :]
            )
        else:
            coords = torch.full((B, L, 3, 3), float("nan"), device=device)
            affine, affine_mask = build_affine3d_from_coordinates(coords)

        x_SPD, _, _ = self.model.transformer(
            x_SPD, sequence_id=sequence_id, affine=affine, affine_mask=affine_mask
        )
        return x_SPD

    def embedding_to_outputs(
        self, embedding_SPD: torch.FloatTensor
    ) -> torch.FloatTensor:
        return self.model.output_heads.sequence_head(embedding_SPD)

    def forward(self, seq_SP: torch.LongTensor, **kwargs) -> torch.FloatTensor:
        # ESM3's forward uses keyword-only args; pass through conditioning if provided
        fwd_kwargs = {"sequence_tokens": seq_SP}
        if "structure_tokens" in kwargs:
            fwd_kwargs["structure_tokens"] = kwargs["structure_tokens"]
        if "coordinates" in kwargs:
            fwd_kwargs["structure_coords"] = kwargs["coordinates"]
        return self.model(**fwd_kwargs)

    def format_raw_to_logits(
        self, raw_output, seq_SP: torch.LongTensor, **kwargs
    ) -> torch.FloatTensor:
        logits_SPT = raw_output.sequence_logits.float()
        return self.logit_formatter(logits_SPT, seq_SP)
