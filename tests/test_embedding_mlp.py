"""Tests for EmbeddingMLP predictive model architecture."""

import torch
import pytest
from torch import nn

from dfm.predictive_modeling import EmbeddingMLP


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB_SIZE = 21  # 20 AAs + UNK
SEQ_LEN = 10
EMBED_DIM = 8
MODEL_DIM = 32
N_LAYERS = 2
OUTPUT_DIM = 3
PADDING_IDX = 20


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model():
    return EmbeddingMLP(
        vocab_size=VOCAB_SIZE,
        sequence_length=SEQ_LEN,
        embed_dim=EMBED_DIM,
        model_dim=MODEL_DIM,
        n_layers=N_LAYERS,
        output_dim=OUTPUT_DIM,
        padding_idx=PADDING_IDX,
    )


@pytest.fixture
def model_with_dropout():
    return EmbeddingMLP(
        vocab_size=VOCAB_SIZE,
        sequence_length=SEQ_LEN,
        embed_dim=EMBED_DIM,
        model_dim=MODEL_DIM,
        n_layers=N_LAYERS,
        output_dim=OUTPUT_DIM,
        padding_idx=PADDING_IDX,
        dropout=0.5,
    )


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_is_nn_module(self, model):
        assert isinstance(model, nn.Module)

    def test_attributes_stored(self, model):
        assert model.vocab_size == VOCAB_SIZE
        assert model.sequence_length == SEQ_LEN
        assert model.embed_dim == EMBED_DIM
        assert model.model_dim == MODEL_DIM
        assert model.n_layers == N_LAYERS
        assert model.output_dim == OUTPUT_DIM
        assert model.padding_idx == PADDING_IDX
        assert model.dropout == 0.0

    def test_dropout_stored(self, model_with_dropout):
        assert model_with_dropout.dropout == 0.5

    def test_embedding_shape(self, model):
        assert model.embed.num_embeddings == VOCAB_SIZE
        assert model.embed.embedding_dim == EMBED_DIM

    def test_embedding_is_learned(self, model):
        """The embedding should be a regular learnable nn.Embedding (not frozen)."""
        assert model.embed.weight.requires_grad

    def test_padding_idx_set(self, model):
        assert model.embed.padding_idx == PADDING_IDX

    def test_padding_embedding_is_zero(self, model):
        assert (model.embed.weight[PADDING_IDX] == 0).all()

    def test_first_linear_input_dim(self, model):
        """First linear layer should accept flattened embeddings: seq_len * embed_dim."""
        first_linear = model.layers[0]
        assert isinstance(first_linear, nn.Linear)
        assert first_linear.in_features == SEQ_LEN * EMBED_DIM

    def test_last_linear_output_dim(self, model):
        """Last layer should output output_dim."""
        last_linear = model.layers[-1]
        assert isinstance(last_linear, nn.Linear)
        assert last_linear.out_features == OUTPUT_DIM

    def test_has_parameters(self, model):
        params = list(model.parameters())
        assert len(params) > 0

    def test_dropout_layers_present(self, model_with_dropout):
        dropout_layers = [m for m in model_with_dropout.layers if isinstance(m, nn.Dropout)]
        assert len(dropout_layers) > 0
        assert all(d.p == 0.5 for d in dropout_layers)

    def test_no_dropout_layers_when_zero(self, model):
        dropout_layers = [m for m in model.layers if isinstance(m, nn.Dropout)]
        assert len(dropout_layers) == 0


# ---------------------------------------------------------------------------
# Forward pass tests
# ---------------------------------------------------------------------------


class TestForward:
    def test_output_shape(self, model):
        x = torch.randint(0, VOCAB_SIZE - 1, (4, SEQ_LEN))
        y = model(x)
        assert y.shape == (4, OUTPUT_DIM)

    def test_single_sample(self, model):
        x = torch.randint(0, VOCAB_SIZE - 1, (1, SEQ_LEN))
        y = model(x)
        assert y.shape == (1, OUTPUT_DIM)

    def test_output_dtype_is_float(self, model):
        x = torch.randint(0, VOCAB_SIZE - 1, (2, SEQ_LEN))
        y = model(x)
        assert y.dtype == torch.float32

    def test_no_nans(self, model):
        x = torch.randint(0, VOCAB_SIZE - 1, (4, SEQ_LEN))
        y = model(x)
        assert not torch.isnan(y).any()

    def test_output_varies_with_input(self, model):
        """Different inputs should (almost certainly) produce different outputs."""
        model.eval()
        x1 = torch.zeros(1, SEQ_LEN, dtype=torch.long)
        x2 = torch.ones(1, SEQ_LEN, dtype=torch.long)
        y1 = model(x1)
        y2 = model(x2)
        assert not torch.allclose(y1, y2)

    def test_padding_token_input(self, model):
        """Should handle sequences containing padding tokens without error."""
        x = torch.full((2, SEQ_LEN), PADDING_IDX, dtype=torch.long)
        y = model(x)
        assert y.shape == (2, OUTPUT_DIM)
        assert not torch.isnan(y).any()

    def test_deterministic_in_eval(self, model):
        """Same input should produce same output in eval mode."""
        model.eval()
        x = torch.randint(0, VOCAB_SIZE - 1, (3, SEQ_LEN))
        y1 = model(x)
        y2 = model(x)
        assert torch.equal(y1, y2)

    def test_large_batch(self, model):
        x = torch.randint(0, VOCAB_SIZE - 1, (128, SEQ_LEN))
        y = model(x)
        assert y.shape == (128, OUTPUT_DIM)


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------


class TestGradients:
    def test_embedding_receives_gradients(self, model):
        """Embedding weights should receive gradients during backprop."""
        x = torch.randint(0, VOCAB_SIZE - 1, (4, SEQ_LEN))
        y = model(x)
        loss = y.sum()
        loss.backward()
        assert model.embed.weight.grad is not None
        # Non-padding tokens should have non-zero grad
        non_pad_grad = model.embed.weight.grad[:PADDING_IDX]
        assert non_pad_grad.abs().sum() > 0

    def test_padding_embedding_no_gradient(self, model):
        """Padding token embedding should not receive gradients."""
        x = torch.randint(0, VOCAB_SIZE - 1, (4, SEQ_LEN))
        y = model(x)
        loss = y.sum()
        loss.backward()
        pad_grad = model.embed.weight.grad[PADDING_IDX]
        assert (pad_grad == 0).all()

    def test_mlp_layers_receive_gradients(self, model):
        x = torch.randint(0, VOCAB_SIZE - 1, (4, SEQ_LEN))
        y = model(x)
        loss = y.sum()
        loss.backward()
        for name, param in model.layers.named_parameters():
            assert param.grad is not None, f"Layer {name} has no gradient"
            assert param.grad.abs().sum() > 0, f"Layer {name} has zero gradient"


# ---------------------------------------------------------------------------
# Architecture variation tests
# ---------------------------------------------------------------------------


class TestArchitectureVariations:
    def test_single_layer(self):
        model = EmbeddingMLP(
            vocab_size=VOCAB_SIZE,
            sequence_length=SEQ_LEN,
            embed_dim=4,
            model_dim=16,
            n_layers=1,
            output_dim=2,
            padding_idx=PADDING_IDX,
        )
        x = torch.randint(0, VOCAB_SIZE - 1, (2, SEQ_LEN))
        y = model(x)
        assert y.shape == (2, 2)

    def test_many_layers(self):
        model = EmbeddingMLP(
            vocab_size=VOCAB_SIZE,
            sequence_length=SEQ_LEN,
            embed_dim=4,
            model_dim=16,
            n_layers=5,
            output_dim=2,
            padding_idx=PADDING_IDX,
        )
        x = torch.randint(0, VOCAB_SIZE - 1, (2, SEQ_LEN))
        y = model(x)
        assert y.shape == (2, 2)

    def test_large_embed_dim(self):
        model = EmbeddingMLP(
            vocab_size=VOCAB_SIZE,
            sequence_length=SEQ_LEN,
            embed_dim=64,
            model_dim=32,
            n_layers=2,
            output_dim=5,
            padding_idx=PADDING_IDX,
        )
        x = torch.randint(0, VOCAB_SIZE - 1, (2, SEQ_LEN))
        y = model(x)
        assert y.shape == (2, 5)

    def test_binary_output(self):
        """Single output dimension for binary classification."""
        model = EmbeddingMLP(
            vocab_size=VOCAB_SIZE,
            sequence_length=SEQ_LEN,
            embed_dim=8,
            model_dim=16,
            n_layers=2,
            output_dim=1,
            padding_idx=PADDING_IDX,
        )
        x = torch.randint(0, VOCAB_SIZE - 1, (4, SEQ_LEN))
        y = model(x)
        assert y.shape == (4, 1)


# ---------------------------------------------------------------------------
# nn.Module integration tests
# ---------------------------------------------------------------------------


class TestModuleIntegration:
    def test_state_dict_saveable(self, model, tmp_path):
        path = tmp_path / "model.pt"
        torch.save(model.state_dict(), path)
        loaded = torch.load(path, weights_only=True)
        assert set(loaded.keys()) == set(model.state_dict().keys())

    def test_state_dict_loadable(self, tmp_path):
        """Save and reload should produce identical outputs."""
        model1 = EmbeddingMLP(
            vocab_size=VOCAB_SIZE,
            sequence_length=SEQ_LEN,
            embed_dim=EMBED_DIM,
            model_dim=MODEL_DIM,
            n_layers=N_LAYERS,
            output_dim=OUTPUT_DIM,
            padding_idx=PADDING_IDX,
        )
        path = tmp_path / "model.pt"
        torch.save(model1.state_dict(), path)

        model2 = EmbeddingMLP(
            vocab_size=VOCAB_SIZE,
            sequence_length=SEQ_LEN,
            embed_dim=EMBED_DIM,
            model_dim=MODEL_DIM,
            n_layers=N_LAYERS,
            output_dim=OUTPUT_DIM,
            padding_idx=PADDING_IDX,
        )
        model2.load_state_dict(torch.load(path, weights_only=True))

        model1.eval()
        model2.eval()
        x = torch.randint(0, VOCAB_SIZE - 1, (3, SEQ_LEN))
        assert torch.equal(model1(x), model2(x))

    def test_eval_and_train_modes(self, model):
        model.eval()
        assert not model.training
        model.train()
        assert model.training

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_propagation(self):
        model = EmbeddingMLP(
            vocab_size=VOCAB_SIZE,
            sequence_length=SEQ_LEN,
            embed_dim=EMBED_DIM,
            model_dim=MODEL_DIM,
            n_layers=N_LAYERS,
            output_dim=OUTPUT_DIM,
            padding_idx=PADDING_IDX,
        )
        model = model.cuda()
        assert model.embed.weight.device.type == "cuda"
        x = torch.randint(0, VOCAB_SIZE - 1, (2, SEQ_LEN), device="cuda")
        y = model(x)
        assert y.device.type == "cuda"


# ---------------------------------------------------------------------------
# Comparison with OneHotMLP
# ---------------------------------------------------------------------------


class TestComparisonWithOneHotMLP:
    def test_fewer_params_with_small_embed_dim(self):
        """EmbeddingMLP with small embed_dim should have fewer params than OneHotMLP."""
        from dfm.predictive_modeling import OneHotMLP

        ohe = OneHotMLP(
            vocab_size=VOCAB_SIZE,
            sequence_length=SEQ_LEN,
            model_dim=MODEL_DIM,
            n_layers=N_LAYERS,
            output_dim=OUTPUT_DIM,
            padding_idx=PADDING_IDX,
        )
        emb = EmbeddingMLP(
            vocab_size=VOCAB_SIZE,
            sequence_length=SEQ_LEN,
            embed_dim=4,  # much smaller than vocab_size=21
            model_dim=MODEL_DIM,
            n_layers=N_LAYERS,
            output_dim=OUTPUT_DIM,
            padding_idx=PADDING_IDX,
        )
        ohe_params = sum(p.numel() for p in ohe.parameters())
        emb_params = sum(p.numel() for p in emb.parameters())
        # OHE first layer: seq_len*vocab_size * model_dim = 10*21*32 = 6720
        # Emb first layer: seq_len*embed_dim * model_dim = 10*4*32 = 1280
        # Plus embedding table: vocab_size*embed_dim = 21*4 = 84
        # So EmbeddingMLP should have fewer total params
        assert emb_params < ohe_params

    def test_same_output_interface(self):
        """Both models should accept LongTensor and return same-shaped output."""
        from dfm.predictive_modeling import OneHotMLP

        ohe = OneHotMLP(
            vocab_size=VOCAB_SIZE,
            sequence_length=SEQ_LEN,
            model_dim=MODEL_DIM,
            n_layers=N_LAYERS,
            output_dim=OUTPUT_DIM,
            padding_idx=PADDING_IDX,
        )
        emb = EmbeddingMLP(
            vocab_size=VOCAB_SIZE,
            sequence_length=SEQ_LEN,
            embed_dim=EMBED_DIM,
            model_dim=MODEL_DIM,
            n_layers=N_LAYERS,
            output_dim=OUTPUT_DIM,
            padding_idx=PADDING_IDX,
        )
        x = torch.randint(0, VOCAB_SIZE - 1, (4, SEQ_LEN))
        y_ohe = ohe(x)
        y_emb = emb(x)
        assert y_ohe.shape == y_emb.shape == (4, OUTPUT_DIM)
