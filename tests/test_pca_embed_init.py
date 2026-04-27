"""Tests for EmbeddingMLP.init_embed_from_pretrained_pca.

Tests the method that initializes an EmbeddingMLP's embedding layer from PCA
of a pretrained model's embeddings, with automatic cross-tokenizer mapping.
Uses synthetic embeddings for fast tests and ESMC for integration tests.
"""

import torch
import pytest

from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from proteingen.modeling import EmbeddingMLP, binary_logits


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"

# 20 standard amino acids, indexed 0–19; index 20 reserved for padding.
AA20_VOCAB: dict[str, int] = {aa: i for i, aa in enumerate(STANDARD_AAS)}
AA20_VOCAB_SIZE = len(AA20_VOCAB) + 1  # +1 for padding slot
AA20_PADDING_IDX = 20

SEQ_LEN = 10
MODEL_DIM = 32
N_LAYERS = 2
MLP_OUTPUT_DIM = 1


# ---------------------------------------------------------------------------
# Concrete subclass (EmbeddingMLP is ABC)
# ---------------------------------------------------------------------------


class ConcreteEmbeddingMLP(EmbeddingMLP):
    def format_raw_to_logits(self, raw_output, ohe_seq_SPT, **kwargs):
        return binary_logits(raw_output.reshape(-1))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tokenizer(vocab_size=AA20_VOCAB_SIZE, pad_token_id=AA20_PADDING_IDX):
    return SimpleNamespace(vocab_size=vocab_size, pad_token_id=pad_token_id)


def make_model(embed_dim: int = 5) -> ConcreteEmbeddingMLP:
    return ConcreteEmbeddingMLP(
        tokenizer=make_tokenizer(),
        sequence_length=SEQ_LEN,
        embed_dim=embed_dim,
        model_dim=MODEL_DIM,
        n_layers=N_LAYERS,
        output_dim=MLP_OUTPUT_DIM,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_source():
    """A small fake pretrained embedding: 30 tokens, 64-dim."""
    torch.manual_seed(42)
    vocab = {aa: i + 5 for i, aa in enumerate(STANDARD_AAS)}  # offset indices
    vocab.update({"<cls>": 0, "<pad>": 1, "<eos>": 2, "<unk>": 3, "<mask>": 4})
    embed = nn.Embedding(30, 64)
    return embed, vocab


@pytest.fixture(scope="module")
def esmc_source():
    """Load ESMC-300m and return (embedding_layer, esm_vocab)."""
    from esm.models.esmc import ESMC
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

    model = ESMC.from_pretrained("esmc_300m", device=torch.device("cpu"))
    tokenizer = EsmSequenceTokenizer()
    return model.embed, dict(tokenizer.vocab)


# ---------------------------------------------------------------------------
# Core behavior
# ---------------------------------------------------------------------------


class TestInitEmbedFromPretrainedPca:
    """Tests using the method on EmbeddingMLP with synthetic source embeddings."""

    def test_weights_are_set(self, synthetic_source):
        source, source_vocab = synthetic_source
        model = make_model(embed_dim=5)
        model.init_embed_from_pretrained_pca(source, source_vocab, AA20_VOCAB)
        for aa in STANDARD_AAS:
            assert model.embed.weight.data[AA20_VOCAB[aa]].abs().sum() > 0

    def test_padding_row_is_zero(self, synthetic_source):
        source, source_vocab = synthetic_source
        model = make_model(embed_dim=5)
        model.init_embed_from_pretrained_pca(source, source_vocab, AA20_VOCAB)
        assert (model.embed.weight.data[AA20_PADDING_IDX] == 0).all()

    def test_unmatched_rows_are_zero(self, synthetic_source):
        """Target tokens not in source vocab get zero embeddings."""
        source, source_vocab = synthetic_source
        # Drop "A" from source so target index 0 has no match
        partial_vocab = {k: v for k, v in source_vocab.items() if k != "A"}
        model = make_model(embed_dim=5)
        model.init_embed_from_pretrained_pca(source, partial_vocab, AA20_VOCAB)
        assert (model.embed.weight.data[AA20_VOCAB["A"]] == 0).all()
        assert model.embed.weight.data[AA20_VOCAB["C"]].abs().sum() > 0

    def test_components_are_orthogonal(self, synthetic_source):
        source, source_vocab = synthetic_source
        model = make_model(embed_dim=5)
        model.init_embed_from_pretrained_pca(source, source_vocab, AA20_VOCAB)
        populated = model.embed.weight.data[[AA20_VOCAB[aa] for aa in STANDARD_AAS]]
        centered = populated - populated.mean(dim=0)
        cov = centered.T @ centered / (centered.shape[0] - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        assert off_diag.abs().max() < 1e-5

    def test_variance_is_descending(self, synthetic_source):
        source, source_vocab = synthetic_source
        model = make_model(embed_dim=5)
        model.init_embed_from_pretrained_pca(source, source_vocab, AA20_VOCAB)
        populated = model.embed.weight.data[[AA20_VOCAB[aa] for aa in STANDARD_AAS]]
        centered = populated - populated.mean(dim=0)
        variances = (centered**2).sum(dim=0) / (centered.shape[0] - 1)
        for i in range(len(variances) - 1):
            assert variances[i] >= variances[i + 1] - 1e-6

    def test_token_mapping_is_correct(self, synthetic_source):
        """Swapping two tokens in source vocab should change the output."""
        source, source_vocab = synthetic_source
        model_a = make_model(embed_dim=5)
        model_a.init_embed_from_pretrained_pca(source, source_vocab, AA20_VOCAB)

        swapped = dict(source_vocab)
        swapped["A"], swapped["C"] = source_vocab["C"], source_vocab["A"]
        model_b = make_model(embed_dim=5)
        model_b.init_embed_from_pretrained_pca(source, swapped, AA20_VOCAB)

        assert not torch.allclose(
            model_a.embed.weight.data[AA20_VOCAB["A"]],
            model_b.embed.weight.data[AA20_VOCAB["A"]],
        )

    def test_forward_works(self, synthetic_source):
        source, source_vocab = synthetic_source
        model = make_model(embed_dim=5)
        model.init_embed_from_pretrained_pca(source, source_vocab, AA20_VOCAB)
        x = torch.randint(0, len(STANDARD_AAS), (4, SEQ_LEN))
        ohe = F.one_hot(x, num_classes=AA20_VOCAB_SIZE).float()
        y = model(ohe)
        assert y.shape == (4, MLP_OUTPUT_DIM)
        assert not torch.isnan(y).any()

    def test_still_learnable(self, synthetic_source):
        source, source_vocab = synthetic_source
        model = make_model(embed_dim=5)
        model.init_embed_from_pretrained_pca(source, source_vocab, AA20_VOCAB)
        assert model.embed.weight.requires_grad
        x = torch.randint(0, len(STANDARD_AAS), (4, SEQ_LEN))
        ohe = F.one_hot(x, num_classes=AA20_VOCAB_SIZE).float()
        loss = model(ohe).sum()
        loss.backward()
        assert model.embed.weight.grad is not None

    def test_embed_dim_equals_shared_tokens(self, synthetic_source):
        """embed_dim == n_shared_tokens should work (max rank)."""
        source, source_vocab = synthetic_source
        model = make_model(embed_dim=20)
        model.init_embed_from_pretrained_pca(source, source_vocab, AA20_VOCAB)
        assert model.embed.weight.shape == (AA20_VOCAB_SIZE, 20)

    def test_single_component(self, synthetic_source):
        source, source_vocab = synthetic_source
        model = make_model(embed_dim=1)
        model.init_embed_from_pretrained_pca(source, source_vocab, AA20_VOCAB)
        assert model.embed.weight.shape == (AA20_VOCAB_SIZE, 1)

    def test_matches_manual_pca(self, synthetic_source):
        """Verify against a manual PCA computation."""
        source, source_vocab = synthetic_source
        n_comp = 5

        # Manual PCA
        weights = source.weight.detach()
        shared = sorted(set(source_vocab.keys()) & set(AA20_VOCAB.keys()))
        pre_idx = [source_vocab[t] for t in shared]
        shared_emb = weights[pre_idx].float()
        mean = shared_emb.mean(dim=0)
        centered = shared_emb - mean
        _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
        projected = centered @ Vt[:n_comp].T

        model = make_model(embed_dim=n_comp)
        model.init_embed_from_pretrained_pca(source, source_vocab, AA20_VOCAB)

        for i, token in enumerate(shared):
            assert torch.allclose(
                model.embed.weight.data[AA20_VOCAB[token]], projected[i], atol=1e-5
            )


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestInitEmbedErrors:
    def test_no_shared_tokens(self):
        source = nn.Embedding(5, 16)
        source_vocab = {"<cls>": 0, "<pad>": 1}
        model = make_model(embed_dim=5)
        with pytest.raises(ValueError, match="No shared tokens"):
            model.init_embed_from_pretrained_pca(source, source_vocab, AA20_VOCAB)

    def test_embed_dim_exceeds_shared_tokens(self, synthetic_source):
        source, source_vocab = synthetic_source
        model = make_model(embed_dim=21)  # only 20 shared tokens
        with pytest.raises(ValueError, match="exceeds number of shared tokens"):
            model.init_embed_from_pretrained_pca(source, source_vocab, AA20_VOCAB)

    def test_embed_dim_exceeds_source_dim(self):
        source = nn.Embedding(25, 3)  # only 3-dim embeddings
        source_vocab = {aa: i for i, aa in enumerate(STANDARD_AAS)}
        model = make_model(embed_dim=5)
        with pytest.raises(ValueError, match="exceeds pretrained embedding dim"):
            model.init_embed_from_pretrained_pca(source, source_vocab, AA20_VOCAB)


# ---------------------------------------------------------------------------
# End-to-end with ESMC (requires model download; slow)
# ---------------------------------------------------------------------------


class TestInitEmbedESMC:
    """Integration tests using real ESMC embeddings."""

    def test_all_aas_populated(self, esmc_source):
        source, source_vocab = esmc_source
        model = make_model(embed_dim=20)
        model.init_embed_from_pretrained_pca(source, source_vocab, AA20_VOCAB)
        for aa in STANDARD_AAS:
            assert model.embed.weight.data[AA20_VOCAB[aa]].abs().sum() > 0

    def test_padding_row_is_zero(self, esmc_source):
        source, source_vocab = esmc_source
        model = make_model(embed_dim=20)
        model.init_embed_from_pretrained_pca(source, source_vocab, AA20_VOCAB)
        assert (model.embed.weight.data[AA20_PADDING_IDX] == 0).all()

    def test_variance_descending(self, esmc_source):
        source, source_vocab = esmc_source
        model = make_model(embed_dim=20)
        model.init_embed_from_pretrained_pca(source, source_vocab, AA20_VOCAB)
        populated = model.embed.weight.data[[AA20_VOCAB[aa] for aa in STANDARD_AAS]]
        centered = populated - populated.mean(dim=0)
        variances = (centered**2).sum(dim=0) / (centered.shape[0] - 1)
        for i in range(len(variances) - 1):
            assert variances[i] >= variances[i + 1] - 1e-6

    def test_components_orthogonal(self, esmc_source):
        source, source_vocab = esmc_source
        model = make_model(embed_dim=20)
        model.init_embed_from_pretrained_pca(source, source_vocab, AA20_VOCAB)
        populated = model.embed.weight.data[[AA20_VOCAB[aa] for aa in STANDARD_AAS]]
        centered = populated - populated.mean(dim=0)
        cov = centered.T @ centered / (centered.shape[0] - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        assert off_diag.abs().max() < 1e-4

    def test_token_mapping_matches_manual(self, esmc_source):
        source, source_vocab = esmc_source
        model = make_model(embed_dim=20)
        model.init_embed_from_pretrained_pca(source, source_vocab, AA20_VOCAB)

        # Manual PCA for Alanine
        weights = source.weight.detach()
        shared = sorted(set(source_vocab.keys()) & set(AA20_VOCAB.keys()))
        pre_idx = [source_vocab[t] for t in shared]
        shared_emb = weights[pre_idx].float()
        centered = shared_emb - shared_emb.mean(dim=0)
        _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
        ala_proj = centered[shared.index("A")] @ Vt[:20].T

        assert torch.allclose(
            model.embed.weight.data[AA20_VOCAB["A"]], ala_proj, atol=1e-4
        )

    def test_cumulative_variance_matches_svd(self, esmc_source):
        source, source_vocab = esmc_source
        model = make_model(embed_dim=20)
        model.init_embed_from_pretrained_pca(source, source_vocab, AA20_VOCAB)

        # Direct SVD for reference
        weights = source.weight.detach()
        shared = sorted(set(source_vocab.keys()) & set(AA20_VOCAB.keys()))
        pre_idx = [source_vocab[t] for t in shared]
        shared_emb = weights[pre_idx].float()
        centered = shared_emb - shared_emb.mean(dim=0)
        _, S, _ = torch.linalg.svd(centered, full_matrices=False)
        expected_var = (S[:20] ** 2 / (centered.shape[0] - 1)).sum()

        populated = model.embed.weight.data[[AA20_VOCAB[aa] for aa in STANDARD_AAS]]
        proj_centered = populated - populated.mean(dim=0)
        proj_var = (proj_centered**2).sum() / (proj_centered.shape[0] - 1)

        assert torch.allclose(proj_var, expected_var, rtol=1e-4)

    def test_forward_end_to_end(self, esmc_source):
        source, source_vocab = esmc_source
        model = make_model(embed_dim=20)
        model.init_embed_from_pretrained_pca(source, source_vocab, AA20_VOCAB)
        x = torch.randint(0, len(STANDARD_AAS), (8, SEQ_LEN))
        ohe = F.one_hot(x, num_classes=AA20_VOCAB_SIZE).float()
        y = model(ohe)
        assert y.shape == (8, MLP_OUTPUT_DIM)
        assert not torch.isnan(y).any()
