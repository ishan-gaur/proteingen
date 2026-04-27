"""Tests for ProGen3 autoregressive protein model wrapper.

These tests require the progen3 package and a GPU with Flash Attention support.
Skip with: pytest -k "not progen3"
"""

import importlib.util

import pytest
import torch

progen3_available = importlib.util.find_spec("progen3") is not None

if progen3_available:
    from proteingen.models.progen3 import ProGen3


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

    def test_encode_mask_tokens(self, model):
        tok = model.tokenizer
        ids = tok.encode_sequence("<mask><mask><mask>")
        # <bos> 1 <mask> <mask> <mask> 2 <eos> = 7 tokens
        assert len(ids) == 7
        assert ids[2] == tok.mask_token_id
        assert ids[3] == tok.mask_token_id
        assert ids[4] == tok.mask_token_id

    def test_extract_sequence(self, model):
        tok = model.tokenizer
        ids = tok.encode_sequence("ACDEFG")
        recovered = tok.extract_sequence(ids)
        assert recovered == "ACDEFG"

    def test_decode(self, model):
        tok = model.tokenizer
        ids = tok.encode_sequence("ACDEFG")
        decoded = tok.decode(ids)
        assert decoded == "ACDEFG"

    def test_batch_decode(self, model):
        tok = model.tokenizer
        batch = torch.tensor(
            [
                tok.encode_sequence("ACD"),
                tok.encode_sequence("EFG"),
            ]
        )
        decoded = tok.batch_decode(batch)
        assert decoded == ["ACD", "EFG"]

    def test_call_padding(self, model):
        tok = model.tokenizer
        result = tok(["AC", "ACDE"], padding=True)
        input_ids = result["input_ids"]
        assert input_ids.shape[0] == 2
        # "AC" → 6 tokens, "ACDE" → 8 tokens, padded to 8
        assert input_ids.shape[1] == 8
        assert input_ids[0, -1].item() == tok.pad_token_id

    def test_special_ids(self, model):
        tok = model.tokenizer
        special = tok.all_special_ids
        assert tok.pad_token_id in special
        assert tok.bos_token_id in special
        assert tok.eos_token_id in special
        assert tok.mask_token_id in special

    def test_aa_token_ids(self, model):
        tok = model.tokenizer
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


class TestLogitFormatter:
    def test_non_mask_positions_one_hot(self, model):
        """Non-mask positions should predict only themselves."""
        tok = model.tokenizer
        result = tok(["ACDE"], padding=True)
        seq_SP = result["input_ids"].to(model.device)
        log_probs = model.get_log_probs(seq_SP)
        probs = torch.exp(log_probs)

        # BOS position (0) should predict BOS with prob ~1
        assert probs[0, 0, tok.bos_token_id].item() > 0.99

        # Direction token position (1) should predict "1" with prob ~1
        assert probs[0, 1, tok.n_to_c_token_id].item() > 0.99

    def test_mask_position_aa_only(self, model):
        """First mask position should only have non-zero prob for AAs."""
        tok = model.tokenizer
        result = tok(["<mask><mask>"], padding=True)
        seq_SP = result["input_ids"].to(model.device)
        log_probs = model.get_log_probs(seq_SP)
        probs = torch.exp(log_probs)

        # Position 2 is the first mask — should have AA probs
        aa_ids = tok.aa_token_ids
        aa_prob_sum = probs[0, 2, aa_ids].sum().item()
        assert aa_prob_sum > 0.99, f"AA probs at first mask: {aa_prob_sum}"

        # Special tokens should have ~0 probability at mask position
        for sid in tok.all_special_ids:
            assert probs[0, 2, sid].item() < 1e-6

    def test_second_mask_blocked(self, model):
        """Second mask position should have all -inf logits (no valid prediction)."""
        tok = model.tokenizer
        result = tok(["<mask><mask>"], padding=True)
        seq_SP = result["input_ids"].to(model.device)
        log_probs = model.get_log_probs(seq_SP)

        # Position 3 is the second mask — should be all -inf
        # log_softmax of all -inf → NaN (no valid distribution)
        assert log_probs[0, 3].isnan().all() or log_probs[0, 3].isinf().all()


class TestEmbedding:
    def test_embed_shape(self, model):
        tok = model.tokenizer
        seq_SP = tok(["ACDEFGHIK"])["input_ids"].to(model.device)
        emb = model.embed(seq_SP)
        S, P = seq_SP.shape
        assert emb.shape == (S, P, model.EMB_DIM)

    def test_embedding_matches_forward(self, model):
        """embedding_to_outputs(embed(x)) should match forward(x)."""
        tok = model.tokenizer
        seq_SP = tok(["ACDEFGHIK"])["input_ids"].to(model.device)
        emb = model.embed(seq_SP)
        emb_logits = model.embedding_to_outputs(emb)
        fwd_logits = model.forward(seq_SP)
        assert torch.allclose(emb_logits, fwd_logits, atol=1e-4), (
            f"Max diff: {(emb_logits - fwd_logits).abs().max().item()}"
        )

    def test_gradients_flow(self, model):
        """Gradients should flow through embed()."""
        tok = model.tokenizer
        seq_SP = tok(["ACDE"])["input_ids"].to(model.device)
        emb = model.embed(seq_SP)
        loss = emb.sum()
        loss.backward()
        # If backward didn't crash, gradients flow


class TestSample:
    def test_sample_left_to_right(self, model):
        """sample() with in_order='left_to_right' should produce valid sequences."""
        from proteingen.sampling import sample

        init_x = ["<mask>" * 10 for _ in range(2)]
        traj = sample(model, init_x, in_order="left_to_right")

        assert len(traj["sequences"]) == 2
        for seq in traj["sequences"]:
            assert len(seq) == 10
            assert all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq)

    def test_sample_trajectory_logged(self, model):
        """sample() should log per-step data."""
        from proteingen.sampling import sample

        init_x = ["<mask>" * 5]
        traj = sample(model, init_x, in_order="left_to_right")

        assert traj["step_log_probs"].shape[1] == 5  # 5 steps
        assert traj["step_tokens"].shape[1] == 5
        # Log probs should be finite (not nan) for real steps
        assert torch.isfinite(traj["step_log_probs"]).all()


class TestGenerate:
    def test_unconditional(self, model):
        result = model.generate(n=2, max_new_tokens=32, temperature=0.8)
        assert "sequences" in result
        assert len(result["sequences"]) == 2
        for seq in result["sequences"]:
            assert isinstance(seq, str)
            assert len(seq) > 0

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
        assert scores["log_likelihood"][0] < 0
        assert scores["perplexity"][0] > 1

    def test_score_batch(self, model):
        sequences = ["ACDEFGHIK", "MKTLLLTLVVVTIVCLD"]
        scores = model.score(sequences)
        assert scores["log_likelihood"].shape == (2,)
