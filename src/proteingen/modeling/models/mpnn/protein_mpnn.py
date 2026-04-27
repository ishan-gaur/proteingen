from __future__ import annotations

import torch
import torch.utils.checkpoint
from typing import TypedDict

from ...generative_modeling import (
    GenerativeModelWithEmbedding,
    LogitFormatter,
    MPNNTokenizer,
)
from mpnn.model.mpnn import ProteinMPNN as _ProteinMPNN
from mpnn.model.layers.message_passing import cat_neighbors_nodes, gather_nodes
from mpnn.utils.weights import load_legacy_weights
from foundry.inference_engines.checkpoint_registry import REGISTERED_CHECKPOINTS


class _BlockUNKLogitFormatter(LogitFormatter):
    """Sets the UNK column to -inf so UNK is never sampled.

    Unlike MaskedModelLogitFormatter, does NOT force delta at non-mask
    input positions. With conditional_minus_self decoding, every position
    gets a meaningful conditional distribution — forcing delta would
    destroy that signal and make sequence scoring useless.
    """

    def __init__(self, unk_idx: int):
        self.unk_idx = unk_idx

    def __call__(
        self, logits: torch.Tensor, input_ids: torch.LongTensor
    ) -> torch.FloatTensor:
        logits = logits.float().clone()
        logits[..., self.unk_idx] = float("-inf")
        return logits


class _MPNNCondition(TypedDict):
    """Internal tensor dict produced by _encode_structure."""

    X: torch.Tensor  # (L, 37, 3) atom coordinates
    X_m: torch.Tensor  # (L, 37) atom existence mask (bool)
    R_idx: torch.Tensor  # (L,) residue indices within each chain (int)
    chain_labels: torch.Tensor  # (L,) chain IDs as integers (int)
    residue_mask: torch.Tensor  # (L,) valid residues (bool)


class ProteinMPNN(GenerativeModelWithEmbedding):
    """ProteinMPNN structure-conditioned sequence design model.

    Wraps Foundry's ProteinMPNN as a GenerativeModelWithEmbedding for use
    with proteingen's sampling, guidance, and probe infrastructure.

    Structure conditioning is **required** — call ``set_condition_()`` or
    ``conditioned_on()`` before ``forward`` / ``get_log_probs`` / ``embed``.

    The wrapper runs the MPNN decoder in **conditional_minus_self** mode:
    each position's prediction is conditioned on all other positions'
    sequence identities (and the structure) but not its own identity.
    This gives a pseudo-likelihood at every position, compatible with
    masked-diffusion sampling.

    Available checkpoints (from Foundry registry):

    ============  =======  ====================================
    checkpoint    params   description
    ============  =======  ====================================
    proteinmpnn   1.7M     standard ProteinMPNN (default)
    solublempnn   1.7M     trained on soluble proteins only
    ============  =======  ====================================

    Tensor Index Legend::

        S: batch index
        P: position index
        T: token/vocab dimension (OUTPUT_DIM = 22)
        D: embedding dimension (EMB_DIM = 128)
        K: number of neighbors (up to 48)
        H: hidden dim (128, same as EMB_DIM)

    Example::

        from proteingen.data import load_pdb

        model = ProteinMPNN()
        structure = load_pdb("1YCR.pdb")
        with model.conditioned_on({"structure": structure, "design_chains": ["B"]}):
            log_probs = model.get_log_probs(seq_SP)
    """

    OUTPUT_DIM = 22  # 21 MPNN tokens (20 AAs + UNK) + 1 mask token

    def __init__(self, checkpoint: str = "proteinmpnn"):
        self._checkpoint = checkpoint

        # Resolve checkpoint path from Foundry registry, auto-download if missing
        checkpoint_info = REGISTERED_CHECKPOINTS[checkpoint]
        checkpoint_path = checkpoint_info.get_default_path()
        if not checkpoint_path.exists():
            from foundry_cli.download_checkpoints import install_model
            from foundry.inference_engines.checkpoint_registry import (
                get_default_checkpoint_dirs,
            )

            install_model(checkpoint, get_default_checkpoint_dirs()[0])
            checkpoint_path = checkpoint_info.get_default_path()

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Could not find Foundry checkpoint for '{checkpoint}' at {checkpoint_path}"
            )

        # Load model with legacy weights
        mpnn = _ProteinMPNN()
        load_legacy_weights(mpnn, str(checkpoint_path))
        mpnn.eval()

        self.EMB_DIM = mpnn.hidden_dim  # 128

        tokenizer = MPNNTokenizer(include_mask_token=True)
        logit_formatter = _BlockUNKLogitFormatter(unk_idx=tokenizer.unk_token_id)

        super().__init__(
            model=mpnn, tokenizer=tokenizer, logit_formatter=logit_formatter
        )

    def _save_args(self) -> dict:
        return {"checkpoint": self._checkpoint}

    # ── Conditioning ─────────────────────────────────────────────────────

    @staticmethod
    def _encode_structure(structure) -> _MPNNCondition:
        """Convert a PDBStructure to the internal conditioning dict.

        Uses Foundry's MPNN atom encoding with occupancy-based masking,
        matching the original ProteinMPNN featurization exactly.
        """
        import numpy as np
        from atomworks.ml.transforms.encoding import atom_array_to_encoding
        from mpnn.transforms.feature_aggregation.mpnn import MPNN_TOKEN_ENCODING

        encoded = atom_array_to_encoding(
            structure.atom_array,
            encoding=MPNN_TOKEN_ENCODING,
            default_coord=0.0,
            occupancy_threshold=0.5,
        )
        X = torch.from_numpy(encoded["xyz"].astype(np.float32))  # (L, 37, 3)
        X_m = torch.from_numpy(encoded["mask"].astype(np.bool_))  # (L, 37)
        L = X.shape[0]

        unique_chains = np.unique(structure.chain_ids)
        chain_to_int = {c: i for i, c in enumerate(unique_chains)}
        chain_labels = torch.tensor(
            [chain_to_int[c] for c in structure.chain_ids], dtype=torch.long
        )  # (L,)

        R_idx = torch.zeros(L, dtype=torch.long)
        for chain_int in range(len(unique_chains)):
            mask = chain_labels == chain_int
            R_idx[mask] = torch.arange(mask.sum())

        return {
            "X": X,
            "X_m": X_m,
            "R_idx": R_idx,
            "chain_labels": chain_labels,
            "residue_mask": torch.ones(L, dtype=torch.bool),
        }

    def preprocess_observations(self, observations: dict) -> dict[str, torch.Tensor]:
        """Encode structure via graph featurization + MPNN encoder.

        Args:
            observations: ``{"structure": PDBStructure}`` with optional
                ``"design_chains": ["B"]`` to restrict designable positions.

        Runs the expensive structure processing once. The resulting encoder
        node/edge features and graph topology are cached and reused for
        every subsequent forward pass.
        """
        from proteingen.data import PDBStructure

        structure = observations["structure"]
        assert isinstance(structure, PDBStructure)
        condition = self._encode_structure(structure)
        design_chains = observations.get("design_chains")
        if design_chains is not None:
            condition["residue_mask"] = torch.tensor(
                [c in design_chains for c in structure.chain_ids],
                dtype=torch.bool,
            )

        device = self.device
        L = condition["X"].shape[0]

        # Build input dict with batch dim for MPNN internals.
        # S=0 (ALA) is a safe placeholder: backbone atoms N/CA/C/O are at
        # positions 0-3 for all amino acids, and ProteinMPNN only uses
        # backbone geometry for graph featurization.
        input_features: dict = {
            "X": condition["X"].unsqueeze(0).to(device).float(),
            "X_m": condition["X_m"].unsqueeze(0).to(device).bool(),
            "S": torch.zeros(1, L, dtype=torch.long, device=device),
            "R_idx": condition["R_idx"].unsqueeze(0).to(device).long(),
            "chain_labels": condition["chain_labels"].unsqueeze(0).to(device).long(),
            "residue_mask": condition["residue_mask"].unsqueeze(0).to(device).bool(),
            "structure_noise": 0.0,
        }

        with torch.no_grad():
            graph_features = self.model.graph_featurization(input_features)
            encoder_features = self.model.encode(input_features, graph_features)

        return {
            "h_V": encoder_features["h_V"].squeeze(0).detach(),  # (L, H)
            "h_E": encoder_features["h_E"].squeeze(0).detach(),  # (L, K, H)
            "E_idx": graph_features["E_idx"].squeeze(0).detach(),  # (L, K)
            "residue_mask": input_features["residue_mask"].squeeze(0).detach(),  # (L,)
        }

    # collate_observations: default from ProbabilityModel works (tiles each
    # tensor to batch size via unsqueeze(0).expand).

    # ── Decoder ──────────────────────────────────────────────────────────

    def _run_decoder(
        self,
        h_S: torch.FloatTensor,
        h_V_enc: torch.FloatTensor,
        h_E_enc: torch.FloatTensor,
        E_idx: torch.LongTensor,
        residue_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """Run MPNN decoder with conditional_minus_self causality.

        Each position attends to all other positions' sequence embeddings
        and decoder node features, but not its own. This gives a
        pseudo-likelihood estimate at every position.

        Args:
            h_S: (B, L, H) sequence embeddings
            h_V_enc: (B, L, H) encoder node features
            h_E_enc: (B, L, K, H) encoder edge features
            E_idx: (B, L, K) neighbor indices
            residue_mask: (B, L) valid residue mask

        Returns:
            (B, L, H) decoder node features
        """
        B, L = residue_mask.shape
        device = h_S.device

        # Causal mask: conditional_minus_self — see all neighbors except self
        causal_all = 1.0 - torch.eye(L, device=device)
        causal_nn = torch.gather(
            causal_all.unsqueeze(0).expand(B, -1, -1), 2, E_idx
        ).unsqueeze(-1)  # (B, L, K, 1)
        causal_mask = causal_nn * residue_mask.view(B, L, 1, 1).float()
        anti_causal_mask = (1.0 - causal_nn) * residue_mask.view(B, L, 1, 1).float()

        # Encoder embeddings masked with anti-causal pattern
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_V_enc), h_E_enc, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)
        h_EXV_encoder_anti_causal = h_EXV_encoder * anti_causal_mask

        # Edge validity mask
        mask_E = gather_nodes(residue_mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_E = residue_mask.unsqueeze(-1) * mask_E

        # Sequence-edge concatenation
        h_ES = cat_neighbors_nodes(h_S, h_E_enc, E_idx)

        # Run decoder layers
        h_V_decoder = h_V_enc
        for layer in self.model.decoder_layers:
            h_ESV_decoder = cat_neighbors_nodes(h_V_decoder, h_ES, E_idx)
            h_ESV = causal_mask * h_ESV_decoder + h_EXV_encoder_anti_causal
            h_V_decoder = layer(h_V_decoder, h_ESV, mask_V=residue_mask, mask_E=mask_E)

        return h_V_decoder

    def _pad_logits(self, logits_21: torch.FloatTensor) -> torch.FloatTensor:
        """Pad 21-dim MPNN logits to 22-dim by appending -inf for mask token."""
        pad = torch.full(
            (*logits_21.shape[:-1], 1),
            float("-inf"),
            device=logits_21.device,
            dtype=logits_21.dtype,
        )
        return torch.cat([logits_21, pad], dim=-1)

    # ── GenerativeModelWithEmbedding interface ───────────────────────────

    def differentiable_embedding(
        self, ohe_seq_SPT: torch.FloatTensor
    ) -> torch.FloatTensor:
        """OHE (or soft distribution) → deep embeddings via MPNN decoder.

        Requires structure conditioning (``self.observations`` must be set).
        The encoder features are retrieved from the cache and tiled to match
        the batch size of the input.

        For mask-token positions, the OHE slice over MPNN's 21 tokens is all
        zeros, giving a zero sequence embedding — equivalent to MPNN's
        "unknown sequence" initialization during inference.
        """
        if self.observations is None:
            raise ValueError(
                "ProteinMPNN requires structure conditioning. "
                "Call set_condition_() or use conditioned_on() first."
            )

        B, L, T = ohe_seq_SPT.shape
        device = ohe_seq_SPT.device

        # Tile cached encoder features to batch size.
        # collate_observations expects a (B, ...) tensor to read batch size.
        dummy = torch.zeros(B, dtype=torch.long, device=device)
        obs = self.collate_observations(dummy, self.observations)
        h_V_enc = obs["h_V"].to(device)
        h_E_enc = obs["h_E"].to(device)
        E_idx = obs["E_idx"].to(device)
        residue_mask = obs["residue_mask"].to(device)

        # Soft sequence embedding — slice to MPNN's 21-token vocab.
        # Mask token (index 21) contributes zero embedding.
        h_S = ohe_seq_SPT[:, :, : self.model.vocab_size] @ self.model.W_s.weight

        return self._run_decoder(h_S, h_V_enc, h_E_enc, E_idx, residue_mask)

    def embedding_to_outputs(
        self, embedding_SPD: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Decoder node features → padded logits (21 AA tokens + mask col)."""
        logits_21 = self.model.W_out(embedding_SPD)
        return self._pad_logits(logits_21)

    def forward(self, seq_SP: torch.LongTensor, **kwargs) -> torch.FloatTensor:
        """Full forward: embed → decoder → logits.

        Uses the differentiable embedding path so that forward and embed
        produce consistent results.
        """
        embedding = self.embed(seq_SP)
        return self.embedding_to_outputs(embedding)

    def format_raw_to_logits(
        self, raw_output: torch.FloatTensor, seq_SP: torch.LongTensor, **kwargs
    ) -> torch.FloatTensor:
        return self.logit_formatter(raw_output.float(), seq_SP)
