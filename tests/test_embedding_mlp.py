"""Tests for EmbeddingMLP and OneHotMLP predictive model architectures."""

import torch
import pytest
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F

from protstar.modeling import EmbeddingMLP, OneHotMLP, binary_logits


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB_SIZE = 21  # 20 AAs + padding
SEQ_LEN = 10
EMBED_DIM = 8
MODEL_DIM = 32
N_LAYERS = 2
OUTPUT_DIM = 1  # binary output for format_raw_to_logits
PADDING_IDX = 20


# ---------------------------------------------------------------------------
# Concrete subclasses (these are ABCs — need format_raw_to_logits)
# ---------------------------------------------------------------------------


class ConcreteEmbeddingMLP(EmbeddingMLP):
    def format_raw_to_logits(self, raw_output, ohe_seq_SPT, **kwargs):
        return binary_logits(raw_output.reshape(-1))


class ConcreteOneHotMLP(OneHotMLP):
    def format_raw_to_logits(self, raw_output, ohe_seq_SPT, **kwargs):
        return binary_logits(raw_output.reshape(-1))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tokenizer(vocab_size=VOCAB_SIZE, pad_token_id=PADDING_IDX):
    return SimpleNamespace(vocab_size=vocab_size, pad_token_id=pad_token_id)


def make_ohe(x_SP, vocab_size=VOCAB_SIZE):
    return F.one_hot(x_SP, num_classes=vocab_size).float()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tokenizer():
    return make_tokenizer()


@pytest.fixture
def model(tokenizer):
    return ConcreteEmbeddingMLP(
        tokenizer=tokenizer,
        sequence_length=SEQ_LEN,
        embed_dim=EMBED_DIM,
        model_dim=MODEL_DIM,
        n_layers=N_LAYERS,
        output_dim=OUTPUT_DIM,
    )


@pytest.fixture
def model_with_dropout(tokenizer):
    return ConcreteEmbeddingMLP(
        tokenizer=tokenizer,
        sequence_length=SEQ_LEN,
        embed_dim=EMBED_DIM,
        model_dim=MODEL_DIM,
        n_layers=N_LAYERS,
        output_dim=OUTPUT_DIM,
        dropout=0.5,
    )


@pytest.fixture
def ohe_model(tokenizer):
    return ConcreteOneHotMLP(
        tokenizer=tokenizer,
        sequence_length=SEQ_LEN,
        model_dim=MODEL_DIM,
        n_layers=N_LAYERS,
        output_dim=OUTPUT_DIM,
    )


# ---------------------------------------------------------------------------
# EmbeddingMLP construction tests
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
        assert model.embed.weight.requires_grad

    def test_padding_idx_set(self, model):
        assert model.embed.padding_idx == PADDING_IDX

    def test_padding_embedding_is_zero(self, model):
        assert (model.embed.weight[PADDING_IDX] == 0).all()

    def test_first_linear_input_dim(self, model):
        first_linear = model.layers[0]
        assert isinstance(first_linear, nn.Linear)
        assert first_linear.in_features == SEQ_LEN * EMBED_DIM

    def test_last_linear_output_dim(self, model):
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

    def test_padding_idx_defaults_from_tokenizer(self, tokenizer):
        m = ConcreteEmbeddingMLP(
            tokenizer=tokenizer,
            sequence_length=SEQ_LEN,
            embed_dim=EMBED_DIM,
            model_dim=MODEL_DIM,
            n_layers=N_LAYERS,
            output_dim=OUTPUT_DIM,
        )
        assert m.padding_idx == tokenizer.pad_token_id

    def test_padding_idx_override(self, tokenizer):
        m = ConcreteEmbeddingMLP(
            tokenizer=tokenizer,
            sequence_length=SEQ_LEN,
            embed_dim=EMBED_DIM,
            model_dim=MODEL_DIM,
            n_layers=N_LAYERS,
            output_dim=OUTPUT_DIM,
            padding_idx=5,
        )
        assert m.padding_idx == 5
        assert m.embed.padding_idx == 5

    def test_is_abstract(self):
        """Can't instantiate EmbeddingMLP directly — format_raw_to_logits is abstract."""
        tokenizer = make_tokenizer()
        with pytest.raises(TypeError):
            EmbeddingMLP(
                tokenizer=tokenizer,
                sequence_length=SEQ_LEN,
                embed_dim=EMBED_DIM,
                model_dim=MODEL_DIM,
                n_layers=N_LAYERS,
                output_dim=OUTPUT_DIM,
            )


# ---------------------------------------------------------------------------
# EmbeddingMLP forward pass tests
# ---------------------------------------------------------------------------


class TestForward:
    def test_output_shape(self, model):
        ohe = make_ohe(torch.randint(0, VOCAB_SIZE - 1, (4, SEQ_LEN)))
        y = model(ohe)
        assert y.shape == (4, OUTPUT_DIM)

    def test_single_sample(self, model):
        ohe = make_ohe(torch.randint(0, VOCAB_SIZE - 1, (1, SEQ_LEN)))
        y = model(ohe)
        assert y.shape == (1, OUTPUT_DIM)

    def test_output_dtype_is_float(self, model):
        ohe = make_ohe(torch.randint(0, VOCAB_SIZE - 1, (2, SEQ_LEN)))
        y = model(ohe)
        assert y.dtype == torch.float32

    def test_no_nans(self, model):
        ohe = make_ohe(torch.randint(0, VOCAB_SIZE - 1, (4, SEQ_LEN)))
        y = model(ohe)
        assert not torch.isnan(y).any()

    def test_output_varies_with_input(self, model):
        model.eval()
        ohe1 = make_ohe(torch.zeros(1, SEQ_LEN, dtype=torch.long))
        ohe2 = make_ohe(torch.ones(1, SEQ_LEN, dtype=torch.long))
        y1 = model(ohe1)
        y2 = model(ohe2)
        assert not torch.allclose(y1, y2)

    def test_padding_token_input(self, model):
        ohe = make_ohe(torch.full((2, SEQ_LEN), PADDING_IDX, dtype=torch.long))
        y = model(ohe)
        assert y.shape == (2, OUTPUT_DIM)
        assert not torch.isnan(y).any()

    def test_deterministic_in_eval(self, model):
        model.eval()
        ohe = make_ohe(torch.randint(0, VOCAB_SIZE - 1, (3, SEQ_LEN)))
        y1 = model(ohe)
        y2 = model(ohe)
        assert torch.equal(y1, y2)

    def test_large_batch(self, model):
        ohe = make_ohe(torch.randint(0, VOCAB_SIZE - 1, (128, SEQ_LEN)))
        y = model(ohe)
        assert y.shape == (128, OUTPUT_DIM)


# ---------------------------------------------------------------------------
# EmbeddingMLP gradient tests
# ---------------------------------------------------------------------------


class TestGradients:
    def test_embedding_receives_gradients(self, model):
        ohe = make_ohe(torch.randint(0, VOCAB_SIZE - 1, (4, SEQ_LEN)))
        y = model(ohe)
        loss = y.sum()
        loss.backward()
        assert model.embed.weight.grad is not None
        non_pad_grad = model.embed.weight.grad[:PADDING_IDX]
        assert non_pad_grad.abs().sum() > 0

    def test_mlp_layers_receive_gradients(self, model):
        ohe = make_ohe(torch.randint(0, VOCAB_SIZE - 1, (4, SEQ_LEN)))
        y = model(ohe)
        loss = y.sum()
        loss.backward()
        for name, param in model.layers.named_parameters():
            assert param.grad is not None, f"Layer {name} has no gradient"
            assert param.grad.abs().sum() > 0, f"Layer {name} has zero gradient"

    def test_ohe_receives_gradients(self, model):
        """OHE input should receive gradients (needed for TAG)."""
        ohe = make_ohe(torch.randint(0, VOCAB_SIZE - 1, (4, SEQ_LEN)))
        ohe.requires_grad_(True)
        y = model(ohe)
        y.sum().backward()
        assert ohe.grad is not None
        assert ohe.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# EmbeddingMLP architecture variation tests
# ---------------------------------------------------------------------------


class TestArchitectureVariations:
    def test_single_layer(self):
        tok = make_tokenizer()
        model = ConcreteEmbeddingMLP(
            tokenizer=tok, sequence_length=SEQ_LEN, embed_dim=4,
            model_dim=16, n_layers=1, output_dim=2,
        )
        ohe = make_ohe(torch.randint(0, VOCAB_SIZE - 1, (2, SEQ_LEN)))
        assert model(ohe).shape == (2, 2)

    def test_many_layers(self):
        tok = make_tokenizer()
        model = ConcreteEmbeddingMLP(
            tokenizer=tok, sequence_length=SEQ_LEN, embed_dim=4,
            model_dim=16, n_layers=5, output_dim=2,
        )
        ohe = make_ohe(torch.randint(0, VOCAB_SIZE - 1, (2, SEQ_LEN)))
        assert model(ohe).shape == (2, 2)

    def test_large_embed_dim(self):
        tok = make_tokenizer()
        model = ConcreteEmbeddingMLP(
            tokenizer=tok, sequence_length=SEQ_LEN, embed_dim=64,
            model_dim=32, n_layers=2, output_dim=5,
        )
        ohe = make_ohe(torch.randint(0, VOCAB_SIZE - 1, (2, SEQ_LEN)))
        assert model(ohe).shape == (2, 5)


# ---------------------------------------------------------------------------
# EmbeddingMLP nn.Module integration tests
# ---------------------------------------------------------------------------


class TestModuleIntegration:
    def test_state_dict_saveable(self, model, tmp_path):
        path = tmp_path / "model.pt"
        torch.save(model.state_dict(), path)
        loaded = torch.load(path, weights_only=True)
        assert set(loaded.keys()) == set(model.state_dict().keys())

    def test_state_dict_loadable(self, tmp_path, tokenizer):
        model1 = ConcreteEmbeddingMLP(
            tokenizer=tokenizer, sequence_length=SEQ_LEN, embed_dim=EMBED_DIM,
            model_dim=MODEL_DIM, n_layers=N_LAYERS, output_dim=OUTPUT_DIM,
        )
        path = tmp_path / "model.pt"
        torch.save(model1.state_dict(), path)

        model2 = ConcreteEmbeddingMLP(
            tokenizer=tokenizer, sequence_length=SEQ_LEN, embed_dim=EMBED_DIM,
            model_dim=MODEL_DIM, n_layers=N_LAYERS, output_dim=OUTPUT_DIM,
        )
        model2.load_state_dict(torch.load(path, weights_only=True))

        model1.eval()
        model2.eval()
        ohe = make_ohe(torch.randint(0, VOCAB_SIZE - 1, (3, SEQ_LEN)))
        assert torch.equal(model1(ohe), model2(ohe))

    def test_eval_and_train_modes(self, model):
        model.eval()
        assert not model.training
        model.train()
        assert model.training

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_propagation(self):
        tok = make_tokenizer()
        model = ConcreteEmbeddingMLP(
            tokenizer=tok, sequence_length=SEQ_LEN, embed_dim=EMBED_DIM,
            model_dim=MODEL_DIM, n_layers=N_LAYERS, output_dim=OUTPUT_DIM,
        )
        model = model.cuda()
        assert model.embed.weight.device.type == "cuda"
        ohe = make_ohe(torch.randint(0, VOCAB_SIZE - 1, (2, SEQ_LEN))).cuda()
        y = model(ohe)
        assert y.device.type == "cuda"


# ---------------------------------------------------------------------------
# PredictiveModel pipeline tests (get_log_probs, grad_log_prob)
# ---------------------------------------------------------------------------


class TestPredictiveModelPipeline:
    def test_get_log_probs_shape(self, model):
        model.set_target_(True)
        seq = torch.randint(0, VOCAB_SIZE - 1, (4, SEQ_LEN))
        log_probs = model.get_log_probs(seq)
        assert log_probs.shape == (4,)

    def test_get_log_probs_are_negative(self, model):
        model.set_target_(True)
        seq = torch.randint(0, VOCAB_SIZE - 1, (4, SEQ_LEN))
        log_probs = model.get_log_probs(seq)
        assert (log_probs <= 0).all()

    def test_get_log_probs_requires_target(self, model):
        seq = torch.randint(0, VOCAB_SIZE - 1, (2, SEQ_LEN))
        with pytest.raises(AssertionError, match="Target not set"):
            model.get_log_probs(seq)

    def test_grad_log_prob_shape(self, model):
        model.set_target_(True)
        seq = torch.randint(0, VOCAB_SIZE - 1, (2, SEQ_LEN))
        grad = model.grad_log_prob(seq)
        assert grad.shape == (2, SEQ_LEN, VOCAB_SIZE)

    def test_grad_log_prob_nonzero(self, model):
        model.set_target_(True)
        seq = torch.randint(0, VOCAB_SIZE - 1, (2, SEQ_LEN))
        grad = model.grad_log_prob(seq)
        assert grad.abs().sum() > 0

    def test_with_target_context_manager(self, model):
        assert model.target is None
        with model.with_target(True):
            assert model.target is True
            seq = torch.randint(0, VOCAB_SIZE - 1, (2, SEQ_LEN))
            log_probs = model.get_log_probs(seq)
            assert log_probs.shape == (2,)
        assert model.target is None

    def test_temperature_affects_log_probs(self, model):
        model.set_target_(True)
        seq = torch.randint(0, VOCAB_SIZE - 1, (4, SEQ_LEN))
        lp1 = model.get_log_probs(seq).detach().clone()
        model.set_temp_(2.0)
        lp2 = model.get_log_probs(seq).detach().clone()
        model.set_temp_(1.0)
        assert not torch.allclose(lp1, lp2)


# ---------------------------------------------------------------------------
# OneHotMLP tests
# ---------------------------------------------------------------------------


class TestOneHotMLP:
    def test_is_abstract(self):
        tok = make_tokenizer()
        with pytest.raises(TypeError):
            OneHotMLP(
                tokenizer=tok, sequence_length=SEQ_LEN,
                model_dim=MODEL_DIM, n_layers=N_LAYERS, output_dim=OUTPUT_DIM,
            )

    def test_forward_shape(self, ohe_model):
        ohe = make_ohe(torch.randint(0, VOCAB_SIZE - 1, (4, SEQ_LEN)))
        y = ohe_model(ohe)
        assert y.shape == (4, OUTPUT_DIM)

    def test_first_linear_input_dim(self, ohe_model):
        first_linear = ohe_model.layers[0]
        assert first_linear.in_features == SEQ_LEN * VOCAB_SIZE

    def test_no_embedding_layer(self, ohe_model):
        """OneHotMLP should not have an embed attribute — OHE comes from get_log_probs."""
        assert not hasattr(ohe_model, "embed")

    def test_get_log_probs(self, ohe_model):
        ohe_model.set_target_(True)
        seq = torch.randint(0, VOCAB_SIZE - 1, (4, SEQ_LEN))
        log_probs = ohe_model.get_log_probs(seq)
        assert log_probs.shape == (4,)
        assert (log_probs <= 0).all()

    def test_grad_log_prob(self, ohe_model):
        ohe_model.set_target_(True)
        seq = torch.randint(0, VOCAB_SIZE - 1, (2, SEQ_LEN))
        grad = ohe_model.grad_log_prob(seq)
        assert grad.shape == (2, SEQ_LEN, VOCAB_SIZE)
        assert grad.abs().sum() > 0

    def test_ohe_receives_gradients(self, ohe_model):
        ohe = make_ohe(torch.randint(0, VOCAB_SIZE - 1, (4, SEQ_LEN)))
        ohe.requires_grad_(True)
        y = ohe_model(ohe)
        y.sum().backward()
        assert ohe.grad is not None
        assert ohe.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Comparison: EmbeddingMLP vs OneHotMLP
# ---------------------------------------------------------------------------


class TestComparisonWithOneHotMLP:
    def test_fewer_params_with_small_embed_dim(self):
        tok = make_tokenizer()
        ohe = ConcreteOneHotMLP(
            tokenizer=tok, sequence_length=SEQ_LEN,
            model_dim=MODEL_DIM, n_layers=N_LAYERS, output_dim=OUTPUT_DIM,
        )
        emb = ConcreteEmbeddingMLP(
            tokenizer=tok, sequence_length=SEQ_LEN, embed_dim=4,
            model_dim=MODEL_DIM, n_layers=N_LAYERS, output_dim=OUTPUT_DIM,
        )
        ohe_params = sum(p.numel() for p in ohe.parameters())
        emb_params = sum(p.numel() for p in emb.parameters())
        assert emb_params < ohe_params

    def test_same_output_interface(self):
        tok = make_tokenizer()
        ohe = ConcreteOneHotMLP(
            tokenizer=tok, sequence_length=SEQ_LEN,
            model_dim=MODEL_DIM, n_layers=N_LAYERS, output_dim=OUTPUT_DIM,
        )
        emb = ConcreteEmbeddingMLP(
            tokenizer=tok, sequence_length=SEQ_LEN, embed_dim=EMBED_DIM,
            model_dim=MODEL_DIM, n_layers=N_LAYERS, output_dim=OUTPUT_DIM,
        )
        ohe_input = make_ohe(torch.randint(0, VOCAB_SIZE - 1, (4, SEQ_LEN)))
        y_ohe = ohe(ohe_input)
        y_emb = emb(ohe_input)
        assert y_ohe.shape == y_emb.shape == (4, OUTPUT_DIM)
