"""Tests for protstar.models.esm.ESM3 (GenerativeModelWithEmbedding)."""

import pytest
import torch
from torch.nn import functional as F


@pytest.fixture(scope="module")
def model():
    from protstar.models.esm import ESM3

    return ESM3()


@pytest.fixture(scope="module")
def tokenizer(model):
    return model.tokenizer


@pytest.fixture(scope="module")
def sample_tokens(tokenizer):
    return tokenizer(["ACDE"], return_tensors="pt")["input_ids"]


class TestConstruction:
    def test_emb_dim(self, model):
        assert model.EMB_DIM == 1536

    def test_output_dim(self, model):
        assert model.OUTPUT_DIM == 64

    def test_logit_formatter_output_dim(self, model):
        assert model.logit_formatter.output_dim == 64

    def test_is_float32(self, model):
        assert next(model.parameters()).dtype == torch.float32


class TestEmbedPathMatchesForward:
    """Differentiable embed path should produce same logits as full ESM3 forward."""

    def test_logits_match(self, model, sample_tokens):
        with torch.no_grad():
            emb = model.embed(sample_tokens)
            logits_embed = model.embedding_to_outputs(emb)
            out_full = model.forward(sample_tokens)
            logits_full = out_full.sequence_logits.float()
        assert logits_embed.shape == logits_full.shape
        assert torch.allclose(logits_embed, logits_full, atol=1e-4)


class TestEmbed:
    def test_output_shape(self, model, sample_tokens):
        emb = model.embed(sample_tokens)
        B, L = sample_tokens.shape
        assert emb.shape == (B, L, model.EMB_DIM)

    def test_gradient_flows(self, model, sample_tokens):
        ohe = F.one_hot(sample_tokens, num_classes=model.tokenizer.vocab_size).float()
        ohe.requires_grad_(True)
        emb = model.differentiable_embedding(ohe)
        emb.sum().backward()
        assert ohe.grad is not None
        assert (ohe.grad != 0).any()


class TestGetLogProbs:
    def test_output_shape(self, model, sample_tokens):
        with torch.no_grad():
            lp = model.get_log_probs(sample_tokens)
        assert lp.shape == (sample_tokens.shape[0], sample_tokens.shape[1], model.OUTPUT_DIM)

    def test_normalized(self, model, sample_tokens):
        with torch.no_grad():
            lp = model.get_log_probs(sample_tokens)
        sums = lp.exp().sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_all_le_zero(self, model, sample_tokens):
        with torch.no_grad():
            lp = model.get_log_probs(sample_tokens)
        assert torch.all(lp <= 0.0)

    def test_no_nan(self, model, sample_tokens):
        with torch.no_grad():
            lp = model.get_log_probs(sample_tokens)
        assert not torch.any(torch.isnan(lp))


class TestBatching:
    def test_batch_vs_individual(self, model, tokenizer):
        seq1, seq2 = "ACDE", "FGHIK"
        tok1 = tokenizer([seq1], return_tensors="pt")["input_ids"]
        tok2 = tokenizer([seq2], return_tensors="pt")["input_ids"]

        with torch.no_grad():
            emb1 = model.embed(tok1)
            emb2 = model.embed(tok2)

        # Same-length batch should match individual
        if tok1.shape[1] == tok2.shape[1]:
            batch = torch.cat([tok1, tok2], dim=0)
            with torch.no_grad():
                emb_batch = model.embed(batch)
            assert torch.allclose(emb_batch[0], emb1[0], atol=1e-4)
            assert torch.allclose(emb_batch[1], emb2[0], atol=1e-4)


class TestStructureConditioning:
    """Structure conditioning via set_condition_ / conditioned_on."""

    @pytest.fixture
    def dummy_coords(self):
        """Backbone-only atom37 coords (L=4)."""
        L = 4
        coords = torch.randn(L, 37, 3)
        coords[:, 3, :] = float("nan")  # CB
        coords[:, 5:, :] = float("nan")  # sidechain
        return coords

    @pytest.fixture
    def masked_tokens(self, tokenizer):
        mask_id = tokenizer.vocab["<mask>"]
        return torch.tensor([[0, mask_id, mask_id, mask_id, mask_id, 2]])

    def test_preprocess_observations(self, model, dummy_coords):
        obs = model.preprocess_observations({"coords_RAX": dummy_coords})
        assert "structure_tokens" in obs
        assert "coordinates" in obs
        # L=4 + BOS + EOS = 6
        assert obs["structure_tokens"].shape == (6,)
        assert obs["coordinates"].shape == (6, 37, 3)

    def test_preprocess_accepts_numpy(self, model, dummy_coords):
        obs = model.preprocess_observations({"coords_RAX": dummy_coords.numpy()})
        assert obs["structure_tokens"].shape == (6,)

    def test_conditioning_changes_log_probs(self, model, dummy_coords, masked_tokens):
        with torch.no_grad():
            lp_uncond = model.get_log_probs(masked_tokens)
            with model.conditioned_on({"coords_RAX": dummy_coords}):
                lp_cond = model.get_log_probs(masked_tokens)
        # Compare only finite values (skip -inf at blocked output positions)
        delta = lp_cond - lp_uncond
        finite_mask = delta.isfinite()
        assert finite_mask.any()
        assert delta[finite_mask].abs().max() > 0.01

    def test_conditioning_changes_embeddings(self, model, dummy_coords, masked_tokens):
        with torch.no_grad():
            emb_uncond = model.embed(masked_tokens)
            with model.conditioned_on({"coords_RAX": dummy_coords}):
                emb_cond = model.embed(masked_tokens)
        assert (emb_cond - emb_uncond).abs().max() > 0.01

    def test_embed_matches_forward_when_conditioned(self, model, dummy_coords, masked_tokens):
        with model.conditioned_on({"coords_RAX": dummy_coords}):
            with torch.no_grad():
                emb = model.embed(masked_tokens)
                logits_embed = model.embedding_to_outputs(emb)
                obs = model.collate_observations(masked_tokens, model.observations)
                out_full = model.forward(masked_tokens, **obs)
                logits_full = out_full.sequence_logits.float()
        assert torch.allclose(logits_embed, logits_full, atol=1e-4)

    def test_conditioned_on_reverts(self, model, dummy_coords, masked_tokens):
        assert model.observations is None
        with model.conditioned_on({"coords_RAX": dummy_coords}):
            assert model.observations is not None
        assert model.observations is None

    def test_conditioned_log_probs_normalized(self, model, dummy_coords, masked_tokens):
        with model.conditioned_on({"coords_RAX": dummy_coords}):
            with torch.no_grad():
                lp = model.get_log_probs(masked_tokens)
        assert torch.allclose(lp.exp().sum(-1), torch.ones(1, 6), atol=1e-4)

    def test_conditioned_log_probs_no_nan(self, model, dummy_coords, masked_tokens):
        with model.conditioned_on({"coords_RAX": dummy_coords}):
            with torch.no_grad():
                lp = model.get_log_probs(masked_tokens)
        assert not torch.any(torch.isnan(lp))


class TestTemperature:
    def test_temperature_changes_output(self, model, tokenizer):
        """Use masked input so logit formatter allows multiple outputs per position."""
        mask_id = tokenizer.vocab["<mask>"]
        # CLS, mask, mask, mask, EOS
        tokens = torch.tensor([[0, mask_id, mask_id, mask_id, 2]], dtype=torch.long)
        with torch.no_grad():
            lp1 = model.get_log_probs(tokens)
            with model.with_temp(2.0):
                lp2 = model.get_log_probs(tokens)
        # Only check mask positions (1:-1) where multiple tokens are valid
        lp1_mask = lp1[:, 1:-1]
        lp2_mask = lp2[:, 1:-1]
        assert not torch.allclose(lp1_mask, lp2_mask)
        # Higher temp → flatter distribution → higher entropy
        entropy1 = -(lp1_mask.exp() * lp1_mask).nan_to_num().sum(-1).mean()
        entropy2 = -(lp2_mask.exp() * lp2_mask).nan_to_num().sum(-1).mean()
        assert entropy2 > entropy1
