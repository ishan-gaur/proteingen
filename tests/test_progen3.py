"""Tests for ProGen3 autoregressive protein model wrapper.

These tests require the progen3 package and a GPU with Flash Attention support.
Skip with: pytest -k "not progen3"
"""

import pytest
import torch

progen3_available = False
try:
    from proteingen.models.progen3 import ProGen3

    progen3_available = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not progen3_available, reason="progen3 package not installed"
)

CHECKPOINT = "Profluent-Bio/progen3-112m"


@pytest.fixture(scope="module")
def model():
    """Load model once for all tests in this module."""
    m = ProGen3(CHECKPOINT)
    if torch.cuda.is_available():
        m = m.cuda()
    return m


class TestTokenizer:
    def test_vocab_size(self, model):
        assert model.tokenizer.vocab_size == 134

    def test_encode_sequence(self, model):
        tok = model.tokenizer
        ids = tok.encode_sequence("ACD")
        # <bos> 1 A C D 2 <eos> = 7 tokens
        assert len(ids) == 7
        assert ids[0] == tok.bos_token_id
        assert ids[1] == tok.n_to_c_token_id
        assert ids[-2] == tok.c_to_n_token_id
        assert ids[-1] == tok.eos_token_id

    def test_extract_sequence(self, model):
        tok = model.tokenizer
        ids = tok.encode_sequence("ACDEFG")
        recovered = tok.extract_sequence(ids)
        assert recovered == "ACDEFG"

    def test_call_padding(self, model):
        tok = model.tokenizer
        result = tok(["AC", "ACDE"], padding=True)
        input_ids = result["input_ids"]
        assert input_ids.shape[0] == 2
        # "AC" → 6 tokens, "ACDE" → 8 tokens, padded to 8
        assert input_ids.shape[1] == 8
        # Padding should be pad_token_id
        assert input_ids[0, -1].item() == tok.pad_token_id

    def test_special_ids(self, model):
        tok = model.tokenizer
        special = tok.all_special_ids
        assert tok.pad_token_id in special
        assert tok.bos_token_id in special
        assert tok.eos_token_id in special

    def test_aa_token_ids(self, model):
        tok = model.tokenizer
        # Should have at least the 20 standard AAs
        assert len(tok.aa_token_ids) >= 20


class TestForward:
    def test_forward_shape(self, model):
        tok = model.tokenizer
        result = tok(["ACDEFGHIK"], padding=True)
        seq_SP = result["input_ids"].to(model.device)
        logits = model.forward(seq_SP)
        S, P = seq_SP.shape
        assert logits.shape == (S, P, 134)

    def test_forward_batch(self, model):
        tok = model.tokenizer
        result = tok(["ACDEF", "GHIKLM"], padding=True)
        seq_SP = result["input_ids"].to(model.device)
        logits = model.forward(seq_SP)
        assert logits.shape[0] == 2
        assert logits.shape[2] == 134

    def test_forward_dtype(self, model):
        tok = model.tokenizer
        result = tok(["ACDE"], padding=True)
        seq_SP = result["input_ids"].to(model.device)
        logits = model.forward(seq_SP)
        assert logits.dtype == torch.float32


class TestGenerate:
    def test_unconditional(self, model):
        result = model.generate(n=2, max_new_tokens=32, temperature=0.8)
        assert "sequences" in result
        assert len(result["sequences"]) == 2
        for seq in result["sequences"]:
            assert isinstance(seq, str)
            assert len(seq) > 0
            # All characters should be amino acids
            assert all(c in "ACDEFGHIKLMNPQRSTVWYBJOUXZ" for c in seq)

    def test_prompted(self, model):
        prompt = "MKTL"
        result = model.generate(prompt=prompt, n=2, max_new_tokens=32, temperature=0.8)
        for seq in result["sequences"]:
            assert seq.startswith(prompt)


class TestScore:
    def test_score_basic(self, model):
        sequences = ["ACDEFGHIKLMNPQRSTVWY"]
        scores = model.score(sequences)
        assert "log_likelihood" in scores
        assert "perplexity" in scores
        assert scores["log_likelihood"].shape == (1,)
        assert scores["perplexity"].shape == (1,)
        # Log-likelihoods should be negative
        assert scores["log_likelihood"][0] < 0
        # Perplexity should be > 1
        assert scores["perplexity"][0] > 1

    def test_score_batch(self, model):
        sequences = ["ACDEFGHIK", "MKTLLLTLVVVTIVCLD"]
        scores = model.score(sequences)
        assert scores["log_likelihood"].shape == (2,)
