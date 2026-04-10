"""Tests for GenerativeModel.

GenerativeModel is a concrete class that wraps an nn.Module backbone,
a tokenizer, and a LogitFormatter into a ready-to-use ProbabilityModel.
"""

import torch
import pytest
from torch import nn
from torch.nn import functional as F

from protstar.generative_modeling import (
    GenerativeModel,
    LogitFormatter,
    PassThroughLogitFormatter,
    MaskedModelLogitFormatter,
)


from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

FakeTokenizer = EsmSequenceTokenizer

# ---------------------------------------------------------------------------
# Mock backbones
# ---------------------------------------------------------------------------

VOCAB_SIZE = 33  # ESM vocab
OUTPUT_DIM = 64  # like ESM's 64-dim padded output


class RandomBackbone(nn.Module):
    """Returns random logits of shape (S, P, output_dim)."""

    def __init__(self, output_dim: int = OUTPUT_DIM):
        super().__init__()
        self._output_dim = output_dim
        self._dummy = nn.Linear(1, 1)  # gives us a parameter

    def forward(self, seq_SP: torch.LongTensor, **kwargs) -> torch.FloatTensor:
        S, P = seq_SP.shape
        return torch.randn(S, P, self._output_dim, device=seq_SP.device)


class RecordingBackbone(nn.Module):
    """Records every forward call's input tensor for assertion."""

    def __init__(self, output_dim: int = VOCAB_SIZE):
        super().__init__()
        self._output_dim = output_dim
        self._dummy = nn.Linear(1, 1)
        self.forward_calls: list[torch.Tensor] = []

    def forward(self, seq_SP: torch.LongTensor, **kwargs) -> torch.FloatTensor:
        self.forward_calls.append(seq_SP.clone())
        S, P = seq_SP.shape
        return torch.randn(S, P, self._output_dim)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def model(tokenizer):
    backbone = RandomBackbone(output_dim=OUTPUT_DIM)
    formatter = PassThroughLogitFormatter()
    return GenerativeModel(backbone, tokenizer, formatter)


@pytest.fixture
def recording_backbone():
    return RecordingBackbone()


@pytest.fixture
def recording_model(tokenizer, recording_backbone):
    formatter = PassThroughLogitFormatter()
    return GenerativeModel(recording_backbone, tokenizer, formatter)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_instantiates(self, model):
        assert isinstance(model, GenerativeModel)
        assert isinstance(model, nn.Module)

    def test_tokenizer_set_by_init(self, model, tokenizer):
        assert model.tokenizer is tokenizer

    def test_logit_formatter_set_by_init(self):
        fmt = PassThroughLogitFormatter()
        tok = FakeTokenizer()
        backbone = RandomBackbone()
        m = GenerativeModel(backbone, tok, fmt)
        assert m.logit_formatter is fmt

    def test_model_set_by_init(self):
        backbone = RandomBackbone()
        tok = FakeTokenizer()
        fmt = PassThroughLogitFormatter()
        m = GenerativeModel(backbone, tok, fmt)
        assert m.model is backbone


# ---------------------------------------------------------------------------
# device property tests
# ---------------------------------------------------------------------------


class TestDevice:
    def test_device_returns_cpu_by_default(self, model):
        assert model.device == torch.device("cpu")

    def test_device_matches_parameter_device(self, model):
        assert model.device == next(model.parameters()).device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_follows_to_cuda(self, model):
        model = model.cuda()
        assert model.device.type == "cuda"
        model = model.cpu()
        assert model.device.type == "cpu"


# ---------------------------------------------------------------------------
# forward tests
# ---------------------------------------------------------------------------


class TestForward:
    def test_output_shape(self, model):
        seq = torch.tensor([[21, 0, 4, 7, 22]])  # CLS A E H EOS
        out = model(seq)
        assert out.shape == (1, 5, OUTPUT_DIM)

    def test_batched_forward(self, model):
        seq = torch.tensor(
            [
                [21, 0, 4, 7, 22, 20],  # CLS A E H EOS PAD
                [21, 1, 2, 3, 8, 22],  # CLS C D F I EOS
            ]
        )
        out = model(seq)
        assert out.shape == (2, 6, OUTPUT_DIM)

    def test_forward_called_with_correct_tensor(
        self, recording_model, recording_backbone
    ):
        seq = torch.tensor([[21, 0, 4, 22]])
        recording_model(seq)
        assert len(recording_backbone.forward_calls) == 1
        assert torch.equal(recording_backbone.forward_calls[0], seq)


# ---------------------------------------------------------------------------
# get_log_probs tests
# ---------------------------------------------------------------------------


class TestGetLogProbs:
    def test_output_is_log_probs(self, model):
        """get_log_probs output should exponentiate to probabilities summing to 1."""
        seq = torch.tensor([[21, 0, 4, 7, 22]])
        log_probs = model.get_log_probs(seq)
        probs = log_probs.exp()
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_temperature_scaling(self, model):
        """Higher temperature should produce more uniform distributions."""
        seq = torch.tensor([[21, 0, 4, 7, 22]])

        with model.with_temp(0.1):
            log_probs_cold = model.get_log_probs(seq)
        with model.with_temp(10.0):
            log_probs_hot = model.get_log_probs(seq)

        # Hot distribution should have higher entropy (more uniform)
        entropy_cold = -(log_probs_cold.exp() * log_probs_cold).sum(dim=-1).mean()
        entropy_hot = -(log_probs_hot.exp() * log_probs_hot).sum(dim=-1).mean()
        assert entropy_hot > entropy_cold


# ---------------------------------------------------------------------------
# get_log_probs_from_string tests
# ---------------------------------------------------------------------------


class TestGetLogProbsFromString:
    def test_single_sequence(self, recording_model, recording_backbone, tokenizer):
        recording_model.get_log_probs_from_string(["ACE"])
        assert len(recording_backbone.forward_calls) == 1
        tensor = recording_backbone.forward_calls[0]
        expected = torch.tensor([tokenizer.encode("ACE")], dtype=torch.long)
        assert torch.equal(tensor, expected)

    def test_output_shape(self, model):
        out = model.get_log_probs_from_string(["ACE"])
        # encode adds CLS + EOS: "ACE" → [CLS, A, C, E, EOS] = 5 tokens
        assert out.shape == (1, 5, OUTPUT_DIM)

    def test_output_is_log_probs(self, model):
        log_probs = model.get_log_probs_from_string(["ACE"])
        probs = log_probs.exp()
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_batch_of_sequences(self, recording_model, recording_backbone):
        recording_model.get_log_probs_from_string(["AC", "ACDE"])
        tensor = recording_backbone.forward_calls[0]
        assert tensor.shape[0] == 2  # batch size

    def test_padding_to_max_length(
        self, recording_model, recording_backbone, tokenizer
    ):
        """Shorter sequences should be right-padded to the longest."""
        recording_model.get_log_probs_from_string(["AC", "ACDE"])
        tensor = recording_backbone.forward_calls[0]

        # "AC"   → [CLS, A, C, EOS]       = 4 tokens → padded to 6
        # "ACDE" → [CLS, A, C, D, E, EOS] = 6 tokens
        assert tensor.shape == (2, 6)
        # Last 2 positions of first sequence should be pad tokens
        assert tensor[0, -1].item() == tokenizer.pad_token_id
        assert tensor[0, -2].item() == tokenizer.pad_token_id
        # Second sequence should have no padding
        assert tensor[1, -1].item() == tokenizer.eos_token_id

    def test_equal_length_no_padding(
        self, recording_model, recording_backbone, tokenizer
    ):
        recording_model.get_log_probs_from_string(["ACE", "FGH"])
        tensor = recording_backbone.forward_calls[0]
        # Both encode to length 5 (CLS + 3 AA + EOS), no padding needed
        assert tensor.shape == (2, 5)
        assert (tensor != tokenizer.pad_token_id).all()

    def test_device_propagation(self, recording_model, recording_backbone):
        """Tensor passed to forward should be on the model's device."""
        recording_model.get_log_probs_from_string(["ACE"])
        tensor = recording_backbone.forward_calls[0]
        assert tensor.device == recording_model.device

    def test_dtype_is_long(self, recording_model, recording_backbone):
        recording_model.get_log_probs_from_string(["ACE"])
        tensor = recording_backbone.forward_calls[0]
        assert tensor.dtype == torch.long

    def test_empty_sequence(self, recording_model, recording_backbone, tokenizer):
        """Empty string should still produce CLS + EOS."""
        recording_model.get_log_probs_from_string([""])
        tensor = recording_backbone.forward_calls[0]
        assert tensor.shape == (1, 2)
        assert tensor[0, 0].item() == tokenizer.cls_token_id
        assert tensor[0, 1].item() == tokenizer.eos_token_id


# ---------------------------------------------------------------------------
# logit_formatter interaction tests
# ---------------------------------------------------------------------------


class TestFormatLogitsIntegration:
    """Verify that GenerativeModel + MaskedModelLogitFormatter work together."""

    @pytest.fixture
    def esm_like_model(self, tokenizer):
        """Model using MaskedModelLogitFormatter, like the real ESM class."""
        backbone = RandomBackbone(output_dim=OUTPUT_DIM)
        formatter = MaskedModelLogitFormatter(tokenizer, output_dim=OUTPUT_DIM)
        return GenerativeModel(backbone, tokenizer, formatter)

    def test_mask_positions_block_special_tokens(self, esm_like_model, tokenizer):
        mask_id = tokenizer.vocab["<mask>"]
        seq = torch.tensor([[tokenizer.cls_token_id, mask_id, tokenizer.eos_token_id]])
        log_probs = esm_like_model.get_log_probs(seq)

        # At the mask position, special tokens should have prob ~0
        mask_pos_probs = log_probs[0, 1].exp()
        for special_id in tokenizer.added_tokens_decoder:
            assert mask_pos_probs[special_id] < 1e-6, (
                f"Special token {special_id} should be blocked at mask position"
            )

    def test_special_positions_predict_themselves(self, esm_like_model, tokenizer):
        mask_id = tokenizer.vocab["<mask>"]
        seq = torch.tensor([[tokenizer.cls_token_id, mask_id, tokenizer.eos_token_id]])
        log_probs = esm_like_model.get_log_probs(seq)

        # CLS position: all probability on CLS
        cls_probs = log_probs[0, 0].exp()
        assert torch.isclose(
            cls_probs[tokenizer.cls_token_id], torch.tensor(1.0), atol=1e-5
        )

        # EOS position: all probability on EOS
        eos_probs = log_probs[0, 2].exp()
        assert torch.isclose(
            eos_probs[tokenizer.eos_token_id], torch.tensor(1.0), atol=1e-5
        )

    def test_regular_token_predicts_itself(self, esm_like_model, tokenizer):
        """A non-mask, non-special token should have all mass on itself."""
        aa_id = tokenizer.vocab["A"]
        seq = torch.tensor([[tokenizer.cls_token_id, aa_id, tokenizer.eos_token_id]])
        log_probs = esm_like_model.get_log_probs(seq)
        aa_probs = log_probs[0, 1].exp()
        assert torch.isclose(aa_probs[aa_id], torch.tensor(1.0), atol=1e-5)

    def test_mask_position_allows_canonical_aas(self, esm_like_model, tokenizer):
        """Mask position should have nonzero probability for canonical AAs only."""
        mask_id = tokenizer.vocab["<mask>"]
        seq = torch.tensor([[tokenizer.cls_token_id, mask_id, tokenizer.eos_token_id]])
        log_probs = esm_like_model.get_log_probs(seq)
        mask_probs = log_probs[0, 1].exp()

        canonical_aas = set("ACDEFGHIKLMNPQRSTVWY")
        idx_to_tok = {idx: tok for tok, idx in tokenizer.vocab.items()}
        special = set(tokenizer.added_tokens_decoder)
        for idx in range(tokenizer.vocab_size):
            if idx in special:
                continue
            tok = idx_to_tok.get(idx, "")
            if tok in canonical_aas:
                assert mask_probs[idx] > 0, (
                    f"Canonical AA {tok} ({idx}) should have nonzero prob"
                )
            else:
                assert mask_probs[idx] == 0, (
                    f"Non-canonical token {tok} ({idx}) should have zero prob"
                )

    def test_logit_formatter_is_accessible(self, esm_like_model):
        assert isinstance(esm_like_model.logit_formatter, LogitFormatter)
        assert isinstance(esm_like_model.logit_formatter, MaskedModelLogitFormatter)


# ---------------------------------------------------------------------------
# conditioning tests
# ---------------------------------------------------------------------------


class TestConditioning:
    def test_set_condition_and_clear(self, model):
        model.set_condition_({"key": torch.tensor([1.0])})
        assert model.observations is not None
        model.observations = None
        assert model.observations is None

    def test_conditioned_on_context_manager(self, model):
        assert model.observations is None
        with model.conditioned_on({"key": torch.tensor([1.0])}):
            assert model.observations is not None
        assert model.observations is None


# ---------------------------------------------------------------------------
# nn.Module integration tests
# ---------------------------------------------------------------------------


class TestModuleIntegration:
    def test_is_nn_module(self, model):
        assert isinstance(model, nn.Module)

    def test_parameters_iterable(self, model):
        params = list(model.parameters())
        assert len(params) > 0

    def test_state_dict_saveable(self, model, tmp_path):
        path = tmp_path / "model.pt"
        torch.save(model.state_dict(), path)
        loaded = torch.load(path, weights_only=True)
        assert set(loaded.keys()) == set(model.state_dict().keys())

    def test_eval_and_train_modes(self, model):
        model.eval()
        assert not model.training
        model.train()
        assert model.training

    def test_formatter_submodule_device_propagation(self, tokenizer):
        """When logit_formatter is an nn.Module (like MaskedModelLogitFormatter),
        registering it as a submodule lets .to(device) propagate buffers."""
        formatter = MaskedModelLogitFormatter(tokenizer, output_dim=OUTPUT_DIM)
        backbone = RandomBackbone()

        class SubmoduleModel(GenerativeModel):
            def __init__(self):
                super().__init__(backbone, tokenizer, formatter)
                # Register formatter as a named submodule so .to() propagates
                self.add_module("_formatter_module", formatter)

        m = SubmoduleModel()
        child_types = [type(c) for c in m.children()]
        assert MaskedModelLogitFormatter in child_types
