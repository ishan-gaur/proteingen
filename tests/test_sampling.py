"""Tests for any-order ancestral sampling with n_parallel > 1.

Uses a deterministic mock transition model so we can verify exact behavior
of position selection, sampling, and convergence.
"""

import torch
import pytest
from torch import nn
from torch.nn import functional as F

from proteingen.generative_modeling import (
    TransitionModel,
    PassThroughLogitFormatter,
)
from proteingen.sampling import any_order_ancestral_step, sample_any_order_ancestral


# ---------------------------------------------------------------------------
# Mock tokenizer — minimal HF-like interface needed by sampling code
# ---------------------------------------------------------------------------


class MockTokenizer:
    """Tiny tokenizer: tokens 0-4 are 'vocab', 5=pad, 6=cls, 7=eos, 8=mask."""

    VOCAB_SIZE = 9

    def __init__(self):
        self.pad_token_id = 5
        self.cls_token_id = 6
        self.eos_token_id = 7
        self.mask_token_id = 8
        self.vocab = {str(i): i for i in range(5)}
        self.vocab.update(
            {"<pad>": 5, "<cls>": 6, "<eos>": 7, "<mask>": 8}
        )
        self.added_tokens_decoder = {5: "<pad>", 6: "<cls>", 7: "<eos>", 8: "<mask>"}

    @property
    def vocab_size(self):
        return self.VOCAB_SIZE

    def __call__(self, sequences, padding=True, return_tensors="pt"):
        encoded = []
        for seq in sequences:
            ids = [self.cls_token_id]
            ids.extend(int(c) for c in seq)
            ids.append(self.eos_token_id)
            encoded.append(ids)
        max_len = max(len(e) for e in encoded)
        for e in encoded:
            while len(e) < max_len:
                e.append(self.pad_token_id)
        return {"input_ids": torch.tensor(encoded, dtype=torch.long)}

    def batch_decode(self, tensor):
        results = []
        for row in tensor:
            results.append(" ".join(str(t.item()) for t in row))
        return results


# ---------------------------------------------------------------------------
# Mock model — returns uniform or deterministic log-probs
# ---------------------------------------------------------------------------


class DeterministicModel(TransitionModel):
    """Always predicts token 0 with probability 1 at every position.

    This makes sampling deterministic: every masked position will be
    filled with token 0.
    """

    def __init__(self):
        tok = MockTokenizer()
        super().__init__(
            model=nn.Linear(1, 1),  # dummy, gives us a parameter
            tokenizer=tok,
            logit_formatter=PassThroughLogitFormatter(),
        )
        self._vocab_size = tok.VOCAB_SIZE

    def forward(self, seq_SP, **kwargs):
        S, P = seq_SP.shape
        logits = torch.full((S, P, self._vocab_size), -1e9)
        logits[:, :, 0] = 0.0  # token 0 gets all the mass
        return logits

    def format_raw_to_logits(self, raw, seq_SP, **kwargs):
        return raw.float()


class UniformModel(TransitionModel):
    """Returns uniform probabilities over tokens 0-4 (the 5 'real' tokens)."""

    def __init__(self):
        tok = MockTokenizer()
        super().__init__(
            model=nn.Linear(1, 1),
            tokenizer=tok,
            logit_formatter=PassThroughLogitFormatter(),
        )
        self._vocab_size = tok.VOCAB_SIZE

    def forward(self, seq_SP, **kwargs):
        S, P = seq_SP.shape
        logits = torch.full((S, P, self._vocab_size), -1e9)
        logits[:, :, :5] = 0.0  # uniform over real tokens
        return logits

    def format_raw_to_logits(self, raw, seq_SP, **kwargs):
        return raw.float()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def det_model():
    return DeterministicModel()


@pytest.fixture
def uniform_model():
    return UniformModel()


@pytest.fixture
def mask_id():
    return MockTokenizer().mask_token_id


# ---------------------------------------------------------------------------
# any_order_ancestral_step — basic behavior
# ---------------------------------------------------------------------------


class TestAnyOrderAncestralStepBasic:
    """Tests for single-step behavior of any_order_ancestral_step."""

    def test_single_position_unmasked(self, det_model, mask_id):
        """n_parallel=1 should unmask exactly one position per sequence."""
        # sequence: [cls, mask, mask, mask, eos]
        x = torch.tensor([[6, 8, 8, 8, 7]])
        result = any_order_ancestral_step(
            det_model, x, n_parallel=1, mask_token_id=mask_id
        )
        n_masks_remaining = (result == mask_id).sum().item()
        assert n_masks_remaining == 2  # started with 3, unmasked 1

    def test_no_masks_is_noop(self, det_model, mask_id):
        """If no positions are masked, step should return input unchanged."""
        x = torch.tensor([[6, 0, 1, 2, 7]])
        result = any_order_ancestral_step(
            det_model, x, n_parallel=3, mask_token_id=mask_id
        )
        assert torch.equal(result, x)

    def test_deterministic_fills_with_expected_token(self, det_model, mask_id):
        """DeterministicModel always predicts token 0."""
        x = torch.tensor([[6, 8, 8, 7]])
        result = any_order_ancestral_step(
            det_model, x, n_parallel=2, mask_token_id=mask_id
        )
        # Both mask positions should be filled with 0
        assert (result == mask_id).sum().item() == 0
        assert result[0, 1].item() == 0
        assert result[0, 2].item() == 0

    def test_non_mask_positions_untouched(self, det_model, mask_id):
        """CLS, EOS, and already-decoded tokens must not change."""
        x = torch.tensor([[6, 3, 8, 4, 7]])  # only position 2 is masked
        original = x.clone()
        result = any_order_ancestral_step(
            det_model, x, n_parallel=1, mask_token_id=mask_id
        )
        # positions 0,1,3,4 should be unchanged
        assert result[0, 0].item() == original[0, 0].item()
        assert result[0, 1].item() == original[0, 1].item()
        assert result[0, 3].item() == original[0, 3].item()
        assert result[0, 4].item() == original[0, 4].item()
        # position 2 should now be unmasked
        assert result[0, 2].item() != mask_id


# ---------------------------------------------------------------------------
# any_order_ancestral_step — n_parallel behavior
# ---------------------------------------------------------------------------


class TestNParallel:
    """Tests for multi-position parallel decoding."""

    def test_n_parallel_unmasks_multiple(self, det_model, mask_id):
        """n_parallel=3 on a sequence with 5 masks should unmask 3."""
        x = torch.tensor([[6, 8, 8, 8, 8, 8, 7]])
        result = any_order_ancestral_step(
            det_model, x, n_parallel=3, mask_token_id=mask_id
        )
        n_masks_remaining = (result == mask_id).sum().item()
        assert n_masks_remaining == 2  # 5 - 3 = 2

    def test_n_parallel_capped_at_available_masks(self, det_model, mask_id):
        """If n_parallel > number of masks, unmask all of them (don't crash)."""
        x = torch.tensor([[6, 8, 8, 7]])  # 2 mask positions
        result = any_order_ancestral_step(
            det_model, x, n_parallel=10, mask_token_id=mask_id
        )
        n_masks_remaining = (result == mask_id).sum().item()
        assert n_masks_remaining == 0  # all 2 unmasked

    def test_n_parallel_equals_1_single_unmask(self, det_model, mask_id):
        """n_parallel=1 should be equivalent to the original behavior."""
        x = torch.tensor([[6, 8, 8, 8, 7]])
        result = any_order_ancestral_step(
            det_model, x, n_parallel=1, mask_token_id=mask_id
        )
        n_masks_remaining = (result == mask_id).sum().item()
        assert n_masks_remaining == 2  # 3 - 1 = 2

    def test_n_parallel_exact_mask_count(self, det_model, mask_id):
        """n_parallel == number of masks should unmask all in one step."""
        x = torch.tensor([[6, 8, 8, 8, 7]])  # 3 masks
        result = any_order_ancestral_step(
            det_model, x, n_parallel=3, mask_token_id=mask_id
        )
        n_masks_remaining = (result == mask_id).sum().item()
        assert n_masks_remaining == 0


# ---------------------------------------------------------------------------
# any_order_ancestral_step — batched behavior with n_parallel
# ---------------------------------------------------------------------------


class TestBatchedNParallel:
    """Batched sequences with different mask counts."""

    def test_batch_independent_parallel(self, det_model, mask_id):
        """Each sequence in the batch should independently unmask n_parallel positions."""
        x = torch.tensor(
            [
                [6, 8, 8, 8, 8, 7],  # 4 masks
                [6, 0, 8, 8, 1, 7],  # 2 masks
            ]
        )
        result = any_order_ancestral_step(
            det_model, x, n_parallel=2, mask_token_id=mask_id
        )
        # seq 0: 4 masks - 2 = 2 remaining
        assert (result[0] == mask_id).sum().item() == 2
        # seq 1: 2 masks - 2 = 0 remaining
        assert (result[1] == mask_id).sum().item() == 0

    def test_batch_one_fully_decoded(self, det_model, mask_id):
        """One sequence already done, other still has masks."""
        x = torch.tensor(
            [
                [6, 0, 1, 2, 7],  # no masks
                [6, 8, 8, 8, 7],  # 3 masks
            ]
        )
        result = any_order_ancestral_step(
            det_model, x, n_parallel=2, mask_token_id=mask_id
        )
        # seq 0 unchanged
        assert torch.equal(result[0], x[0])
        # seq 1: 3 - 2 = 1 remaining
        assert (result[1] == mask_id).sum().item() == 1

    def test_batch_n_parallel_exceeds_one_sequence(self, det_model, mask_id):
        """n_parallel > masks in seq 1, but < masks in seq 0."""
        x = torch.tensor(
            [
                [6, 8, 8, 8, 8, 7],  # 4 masks
                [6, 0, 8, 1, 2, 7],  # 1 mask
            ]
        )
        result = any_order_ancestral_step(
            det_model, x, n_parallel=3, mask_token_id=mask_id
        )
        # seq 0: 4 - 3 = 1 remaining
        assert (result[0] == mask_id).sum().item() == 1
        # seq 1: 1 - 1 = 0 (capped)
        assert (result[1] == mask_id).sum().item() == 0


# ---------------------------------------------------------------------------
# any_order_ancestral_step — explicit next_pos_idx_SP
# ---------------------------------------------------------------------------


class TestExplicitPositionSelection:
    """Test caller-specified position indices."""

    def test_explicit_single_position(self, det_model, mask_id):
        x = torch.tensor([[6, 8, 8, 8, 7]])
        pos_idx = torch.LongTensor([[0, 2]])  # unmask position 2 only
        result = any_order_ancestral_step(
            det_model,
            x,
            n_parallel=1,
            mask_token_id=mask_id,
            next_pos_idx_SP=pos_idx,
        )
        assert result[0, 2].item() == 0  # deterministic → token 0
        assert result[0, 1].item() == mask_id  # untouched
        assert result[0, 3].item() == mask_id  # untouched

    def test_explicit_multiple_positions(self, det_model, mask_id):
        x = torch.tensor([[6, 8, 8, 8, 7]])
        pos_idx = torch.LongTensor([[0, 1], [0, 3]])  # unmask positions 1 and 3
        result = any_order_ancestral_step(
            det_model,
            x,
            n_parallel=2,
            mask_token_id=mask_id,
            next_pos_idx_SP=pos_idx,
        )
        assert result[0, 1].item() == 0
        assert result[0, 3].item() == 0
        assert result[0, 2].item() == mask_id  # not selected

    def test_explicit_empty_positions_is_noop(self, det_model, mask_id):
        x = torch.tensor([[6, 8, 8, 7]])
        pos_idx = torch.LongTensor([]).reshape(0, 2)
        original = x.clone()
        result = any_order_ancestral_step(
            det_model,
            x,
            n_parallel=1,
            mask_token_id=mask_id,
            next_pos_idx_SP=pos_idx,
        )
        assert torch.equal(result, original)


# ---------------------------------------------------------------------------
# sample_any_order_ancestral — full integration with n_parallel
# ---------------------------------------------------------------------------


class TestSampleAnyOrderAncestral:
    """End-to-end tests for the full sampling loop."""

    def test_fully_masked_completes(self, det_model):
        """A fully masked sequence should eventually have zero masks."""
        # [cls, mask, mask, mask, eos]
        mask_id = det_model.tokenizer.mask_token_id
        x = torch.tensor([[6, 8, 8, 8, 7]])
        result = sample_any_order_ancestral(
            det_model, x, n_parallel=1, return_string=False
        )
        assert (result == mask_id).sum().item() == 0

    def test_n_parallel_completes(self, det_model):
        """n_parallel > 1 should also fully decode."""
        mask_id = det_model.tokenizer.mask_token_id
        x = torch.tensor([[6, 8, 8, 8, 8, 8, 7]])
        result = sample_any_order_ancestral(
            det_model, x, n_parallel=3, return_string=False
        )
        assert (result == mask_id).sum().item() == 0

    def test_n_parallel_one_step_for_all(self, det_model):
        """If n_parallel >= n_masks, should complete in one step."""
        mask_id = det_model.tokenizer.mask_token_id
        x = torch.tensor([[6, 8, 8, 7]])  # 2 masks
        result = sample_any_order_ancestral(
            det_model, x, n_parallel=5, return_string=False
        )
        assert (result == mask_id).sum().item() == 0

    def test_deterministic_output_values(self, det_model):
        """DeterministicModel should fill all masks with token 0."""
        x = torch.tensor([[6, 8, 8, 8, 7]])
        result = sample_any_order_ancestral(
            det_model, x, n_parallel=2, return_string=False
        )
        assert result[0, 1].item() == 0
        assert result[0, 2].item() == 0
        assert result[0, 3].item() == 0

    def test_cls_eos_preserved(self, det_model):
        """Special tokens (CLS/EOS) should not be modified."""
        tok = det_model.tokenizer
        x = torch.tensor([[tok.cls_token_id, 8, 8, tok.eos_token_id]])
        result = sample_any_order_ancestral(
            det_model, x, n_parallel=2, return_string=False
        )
        assert result[0, 0].item() == tok.cls_token_id
        assert result[0, -1].item() == tok.eos_token_id

    def test_batched_different_mask_counts(self, det_model):
        """Batch with different number of masks per sequence."""
        mask_id = det_model.tokenizer.mask_token_id
        x = torch.tensor(
            [
                [6, 8, 8, 8, 8, 7],  # 4 masks
                [6, 0, 8, 1, 2, 7],  # 1 mask
            ]
        )
        result = sample_any_order_ancestral(
            det_model, x, n_parallel=2, return_string=False
        )
        assert (result == mask_id).sum().item() == 0

    def test_return_string(self, det_model):
        """return_string=True should return a list of strings."""
        x = torch.tensor([[6, 8, 8, 7]])
        result = sample_any_order_ancestral(
            det_model, x, n_parallel=1, return_string=True
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], str)

    def test_returns_to_original_device(self, det_model):
        """Output tensor should be on the same device as input."""
        x = torch.tensor([[6, 8, 8, 7]])  # CPU
        result = sample_any_order_ancestral(
            det_model, x, n_parallel=1, return_string=False
        )
        assert result.device == x.device

    def test_already_decoded_is_noop(self, det_model):
        """If input has no masks, should return immediately."""
        x = torch.tensor([[6, 0, 1, 2, 7]])
        original = x.clone()
        result = sample_any_order_ancestral(
            det_model, x, n_parallel=2, return_string=False
        )
        assert torch.equal(result, original)


# ---------------------------------------------------------------------------
# Stochastic property tests (with UniformModel)
# ---------------------------------------------------------------------------


class TestStochasticSampling:
    """Statistical tests with UniformModel to verify randomness works."""

    def test_samples_are_in_valid_range(self, uniform_model):
        """All decoded tokens should be valid vocab tokens (0-4)."""
        mask_id = uniform_model.tokenizer.mask_token_id
        x = torch.tensor([[6, 8, 8, 8, 8, 7]])
        result = sample_any_order_ancestral(
            uniform_model, x, n_parallel=2, return_string=False
        )
        inner_tokens = result[0, 1:-1]  # skip CLS and EOS
        assert (inner_tokens >= 0).all()
        assert (inner_tokens < 5).all()

    def test_n_parallel_produces_varied_samples(self, uniform_model):
        """Multiple runs should not always produce the same output."""
        mask_id = uniform_model.tokenizer.mask_token_id
        results = set()
        for _ in range(20):
            x = torch.tensor([[6, 8, 8, 8, 8, 7]])
            result = sample_any_order_ancestral(
                uniform_model, x, n_parallel=2, return_string=False
            )
            results.add(tuple(result[0, 1:-1].tolist()))
        # With 5 tokens and 4 positions, should get variety in 20 tries
        assert len(results) > 1

    def test_different_n_parallel_all_complete(self, uniform_model):
        """Various n_parallel values should all fully decode."""
        mask_id = uniform_model.tokenizer.mask_token_id
        for n_par in [1, 2, 3, 4, 5, 10]:
            x = torch.tensor([[6, 8, 8, 8, 8, 7]])
            result = sample_any_order_ancestral(
                uniform_model, x, n_parallel=n_par, return_string=False
            )
            assert (result == mask_id).sum().item() == 0, (
                f"Failed to fully decode with n_parallel={n_par}"
            )
