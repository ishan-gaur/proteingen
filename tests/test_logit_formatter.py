"""Tests for MaskedModelLogitFomatter with real HuggingFace tokenizers."""

import torch
import pytest
from torch.nn import functional as F

from proteingen.modeling import MaskedModelLogitFormatter, LogitFormatter

# ---------------------------------------------------------------------------
# Constants — mask token strings per tokenizer
# ---------------------------------------------------------------------------

ESM_MASK_TOKEN = "<mask>"
BERT_MASK_TOKEN = "[MASK]"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def esm_tokenizer():
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

    return EsmSequenceTokenizer()


@pytest.fixture
def bert_tokenizer():
    from transformers import BertTokenizer

    return BertTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture
def esm_formatter(esm_tokenizer):
    return MaskedModelLogitFormatter(esm_tokenizer)


@pytest.fixture
def esm_formatter_64(esm_tokenizer):
    return MaskedModelLogitFormatter(esm_tokenizer, output_dim=64)


@pytest.fixture
def bert_formatter(bert_tokenizer):
    return MaskedModelLogitFormatter(bert_tokenizer)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mask_token_id(tokenizer, mask_token: str) -> int:
    return tokenizer.vocab[mask_token]


STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def _non_special_ids(tokenizer) -> set[int]:
    """Return the set of token ids that are NOT special tokens."""
    all_ids = set(tokenizer.vocab.values())
    special_ids = set(tokenizer.added_tokens_decoder.keys())
    return all_ids - special_ids


def _canonical_aa_ids(tokenizer) -> set[int]:
    """Return token ids corresponding to canonical amino acids."""
    special_ids = set(tokenizer.added_tokens_decoder.keys())
    idx_to_tok = {idx: tok for tok, idx in tokenizer.vocab.items()}
    return {
        idx
        for idx in range(tokenizer.vocab_size)
        if idx not in special_ids and idx_to_tok.get(idx, "") in STANDARD_AAS
    }


def _special_ids(tokenizer) -> set[int]:
    return set(tokenizer.added_tokens_decoder.keys())


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_esm_basic(self, esm_formatter, esm_tokenizer):
        assert esm_formatter.valid_output_mask_TiTo.shape == (
            esm_tokenizer.vocab_size,
            esm_tokenizer.vocab_size,
        )

    def test_esm_output_dim_64(self, esm_formatter_64, esm_tokenizer):
        assert esm_formatter_64.valid_output_mask_TiTo.shape == (
            esm_tokenizer.vocab_size,
            64,
        )

    def test_bert_basic(self, bert_formatter, bert_tokenizer):
        assert bert_formatter.valid_output_mask_TiTo.shape == (
            bert_tokenizer.vocab_size,
            bert_tokenizer.vocab_size,
        )

    def test_output_dim_too_small_raises(self, esm_tokenizer):
        with pytest.raises(AssertionError):
            MaskedModelLogitFormatter(esm_tokenizer, output_dim=10)

    def test_satisfies_protocol(self, esm_formatter):
        assert isinstance(esm_formatter, LogitFormatter)

    def test_buffer_dtype_is_float32(self, esm_formatter):
        assert esm_formatter.valid_output_mask_TiTo.dtype == torch.float32


# ---------------------------------------------------------------------------
# Mask matrix tests — verify the (Ti, To) constraint matrix directly
# ---------------------------------------------------------------------------


class TestMaskMatrix:
    def test_non_mask_special_tokens_map_to_themselves(
        self, esm_formatter, esm_tokenizer
    ):
        """Each special token (except mask) should have 0.0 only at its own column."""
        mask = esm_formatter.valid_output_mask_TiTo
        mask_id = _mask_token_id(esm_tokenizer, ESM_MASK_TOKEN)
        for idx in _special_ids(esm_tokenizer):
            if idx == mask_id:
                continue  # mask token is special but has different rules
            row = mask[idx]
            assert row[idx] == 0.0
            finite_positions = torch.isfinite(row).nonzero(as_tuple=True)[0].tolist()
            assert finite_positions == [idx]

    def test_mask_token_maps_to_canonical_aas(self, esm_formatter, esm_tokenizer):
        """Mask row should allow only canonical AAs (default canonical_only=True)."""
        matrix = esm_formatter.valid_output_mask_TiTo
        mask_id = _mask_token_id(esm_tokenizer, ESM_MASK_TOKEN)
        row = matrix[mask_id]
        canonical = _canonical_aa_ids(esm_tokenizer)

        for idx in _special_ids(esm_tokenizer):
            assert row[idx] == float("-inf"), f"special token {idx} should be blocked"

        for idx in canonical:
            assert row[idx] == 0.0, f"canonical AA {idx} should be allowed"

        non_canonical_non_special = _non_special_ids(esm_tokenizer) - canonical
        for idx in non_canonical_non_special:
            assert row[idx] == float("-inf"), f"non-canonical token {idx} should be blocked"

    def test_mask_token_maps_to_all_non_special_when_canonical_off(
        self, esm_tokenizer
    ):
        """With canonical_only=False, mask allows all non-special tokens."""
        formatter = MaskedModelLogitFormatter(esm_tokenizer, canonical_only=False)
        matrix = formatter.valid_output_mask_TiTo
        mask_id = _mask_token_id(esm_tokenizer, ESM_MASK_TOKEN)
        row = matrix[mask_id]

        for idx in _non_special_ids(esm_tokenizer):
            assert row[idx] == 0.0, f"non-special token {idx} should be allowed"

    def test_regular_tokens_map_to_themselves(self, esm_formatter, esm_tokenizer):
        """Non-special, non-mask tokens should have 0.0 only at their own column."""
        matrix = esm_formatter.valid_output_mask_TiTo
        mask_id = _mask_token_id(esm_tokenizer, ESM_MASK_TOKEN)

        for idx in _non_special_ids(esm_tokenizer):
            if idx == mask_id:
                continue
            row = matrix[idx]
            assert row[idx] == 0.0
            finite_positions = torch.isfinite(row).nonzero(as_tuple=True)[0].tolist()
            assert finite_positions == [idx]

    def test_bert_mask_token_maps_to_non_special(self, bert_formatter, bert_tokenizer):
        """BERT has no standard AAs, so canonical_only falls back to all non-special."""
        matrix = bert_formatter.valid_output_mask_TiTo
        mask_id = _mask_token_id(bert_tokenizer, BERT_MASK_TOKEN)
        row = matrix[mask_id]

        for idx in _special_ids(bert_tokenizer):
            assert row[idx] == float("-inf")

        for idx in _non_special_ids(bert_tokenizer):
            assert row[idx] == 0.0

    def test_output_dim_64_extra_columns_blocked_for_all(
        self, esm_formatter_64, esm_tokenizer
    ):
        """Columns beyond vocab_size don't correspond to real tokens and should
        be blocked (-inf) for all input tokens, including mask."""
        matrix = esm_formatter_64.valid_output_mask_TiTo
        mask_id = _mask_token_id(esm_tokenizer, ESM_MASK_TOKEN)

        for col in range(esm_tokenizer.vocab_size, 64):
            assert matrix[mask_id, col] == float("-inf"), (
                f"mask should block extra column {col}"
            )

        # Regular AA token should be blocked at extra columns
        aa_idx = esm_tokenizer.vocab["A"]
        for col in range(esm_tokenizer.vocab_size, 64):
            assert matrix[aa_idx, col] == float("-inf")

        # Special tokens should be blocked at extra columns
        for col in range(esm_tokenizer.vocab_size, 64):
            assert matrix[esm_tokenizer.cls_token_id, col] == float("-inf")


# ---------------------------------------------------------------------------
# Forward pass tests
# ---------------------------------------------------------------------------


class TestForward:
    def test_output_shape_matches_input(self, esm_formatter, esm_tokenizer):
        seq = torch.tensor([[0, 5, 23, 32, 2]])  # CLS, A, C, <mask>, EOS
        logits = torch.randn(1, 5, esm_tokenizer.vocab_size)
        out = esm_formatter(logits, seq)
        assert out.shape == logits.shape

    def test_output_dtype_is_float32(self, esm_formatter, esm_tokenizer):
        seq = torch.tensor([[0, 5, 23, 32, 2]])
        # Feed float16 logits — output should be cast to float32
        logits = torch.randn(1, 5, esm_tokenizer.vocab_size, dtype=torch.float16)
        out = esm_formatter(logits, seq)
        assert out.dtype == torch.float32

    def test_no_nans(self, esm_formatter, esm_tokenizer):
        seq = torch.tensor([[0, 5, 23, 32, 2]])
        logits = torch.randn(1, 5, esm_tokenizer.vocab_size)
        out = esm_formatter(logits, seq)
        assert not torch.isnan(out).any()

    def test_special_token_position_only_self(self, esm_formatter, esm_tokenizer):
        """At a CLS position, only the CLS logit should be finite."""
        cls_id = esm_tokenizer.cls_token_id
        seq = torch.tensor([[cls_id, 5, 2]])  # CLS, A, EOS
        logits = torch.randn(1, 3, esm_tokenizer.vocab_size)
        out = esm_formatter(logits, seq)

        finite_mask = torch.isfinite(out[0, 0])
        assert finite_mask[cls_id]
        assert finite_mask.sum() == 1

    def test_mask_position_blocks_special(self, esm_formatter, esm_tokenizer):
        """At a <mask> position, all special token outputs should be -inf."""
        mask_id = _mask_token_id(esm_tokenizer, ESM_MASK_TOKEN)
        seq = torch.tensor([[0, mask_id, 2]])
        logits = torch.randn(1, 3, esm_tokenizer.vocab_size)
        out = esm_formatter(logits, seq)

        for s_id in _special_ids(esm_tokenizer):
            assert out[0, 1, s_id] == float("-inf")

    def test_mask_position_allows_canonical_aas(self, esm_formatter, esm_tokenizer):
        """At a <mask> position, canonical AA outputs should be finite."""
        mask_id = _mask_token_id(esm_tokenizer, ESM_MASK_TOKEN)
        seq = torch.tensor([[0, mask_id, 2]])
        logits = torch.randn(1, 3, esm_tokenizer.vocab_size)
        out = esm_formatter(logits, seq)

        for aa_id in _canonical_aa_ids(esm_tokenizer):
            assert torch.isfinite(out[0, 1, aa_id])

        non_canonical = _non_special_ids(esm_tokenizer) - _canonical_aa_ids(esm_tokenizer)
        for nc_id in non_canonical:
            assert out[0, 1, nc_id] == float("-inf")

    def test_regular_token_preserves_own_logit(self, esm_formatter, esm_tokenizer):
        """A regular token position should pass through its own logit unchanged."""
        aa_idx = esm_tokenizer.vocab["A"]  # 5
        seq = torch.tensor([[0, aa_idx, 2]])
        logits = torch.randn(1, 3, esm_tokenizer.vocab_size)
        out = esm_formatter(logits, seq)

        # The logit at the A position for token A should be unchanged (mask adds 0.0)
        assert out[0, 1, aa_idx] == logits[0, 1, aa_idx]

    def test_softmax_after_formatting_sums_to_one(self, esm_formatter, esm_tokenizer):
        """log_softmax after formatting should produce valid distributions."""
        mask_id = _mask_token_id(esm_tokenizer, ESM_MASK_TOKEN)
        seq = torch.tensor([[0, mask_id, 5, 2]])
        logits = torch.randn(1, 4, esm_tokenizer.vocab_size)
        out = esm_formatter(logits, seq)
        probs = F.softmax(out, dim=-1)

        for pos in range(4):
            assert torch.isclose(probs[0, pos].sum(), torch.tensor(1.0), atol=1e-5)

    def test_batch_dimension(self, esm_formatter, esm_tokenizer):
        """Should work with batched inputs."""
        mask_id = _mask_token_id(esm_tokenizer, ESM_MASK_TOKEN)
        seq = torch.tensor(
            [
                [0, 5, 23, mask_id, 2],
                [0, mask_id, mask_id, 10, 2],
            ]
        )
        logits = torch.randn(2, 5, esm_tokenizer.vocab_size)
        out = esm_formatter(logits, seq)
        assert out.shape == (2, 5, esm_tokenizer.vocab_size)
        assert not torch.isnan(out).any()

    def test_output_dim_64_forward(self, esm_formatter_64, esm_tokenizer):
        mask_id = _mask_token_id(esm_tokenizer, ESM_MASK_TOKEN)
        seq = torch.tensor([[0, 5, mask_id, 2]])
        logits = torch.randn(1, 4, 64)
        out = esm_formatter_64(logits, seq)
        assert out.shape == (1, 4, 64)
        assert not torch.isnan(out).any()

    def test_bert_forward(self, bert_formatter, bert_tokenizer):
        mask_id = _mask_token_id(bert_tokenizer, BERT_MASK_TOKEN)
        # [CLS]=101, hello=7592, [MASK]=103, [SEP]=102
        seq = torch.tensor([[101, 7592, mask_id, 102]])
        logits = torch.randn(1, 4, bert_tokenizer.vocab_size)
        out = bert_formatter(logits, seq)
        assert out.shape == logits.shape
        assert not torch.isnan(out).any()

        # CLS position: only idx 101 finite
        finite_cls = torch.isfinite(out[0, 0])
        assert finite_cls[101] and finite_cls.sum() == 1

        # MASK position: special blocked, non-special open
        for s_id in _special_ids(bert_tokenizer):
            assert out[0, 2, s_id] == float("-inf")


# ---------------------------------------------------------------------------
# Device tracking tests
# ---------------------------------------------------------------------------


class TestDeviceTracking:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_buffer_moves_with_module(self, esm_formatter):
        esm_formatter.cuda()
        assert esm_formatter.valid_output_mask_TiTo.device.type == "cuda"
        esm_formatter.cpu()
        assert esm_formatter.valid_output_mask_TiTo.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_buffer_moves_as_submodule(self, esm_tokenizer):
        """When embedded in a parent model, .to(device) should propagate."""

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.formatter = MaskedModelLogitFormatter(esm_tokenizer)

        model = DummyModel().cuda()
        assert model.formatter.valid_output_mask_TiTo.device.type == "cuda"
