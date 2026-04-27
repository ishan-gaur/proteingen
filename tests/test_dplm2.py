import torch
import pytest
from proteingen.modeling import DPLM2


@pytest.fixture(scope="module")
def model():
    m = DPLM2("airkingbd/dplm2_150m")
    m.to(m.device)
    return m


# ── Tokenizer ────────────────────────────────────────────────────────


class TestDPLM2Tokenizer:
    @pytest.fixture
    def tokenizer(self, model):
        return model.tokenizer

    def test_special_tokens(self, tokenizer):
        assert tokenizer.cls_token_id == 0
        assert tokenizer.pad_token_id == 1
        assert tokenizer.eos_token_id == 2
        assert tokenizer.mask_token_id == 32

    def test_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size == 8229

    def test_encode_decode(self, tokenizer):
        seq = "ACDEF"
        encoded = tokenizer.encode(seq)
        assert encoded[0] == 0  # cls_aa
        assert encoded[-1] == 2  # eos_aa
        assert len(encoded) == len(seq) + 2

        decoded = tokenizer.decode(encoded)
        assert decoded == seq

    def test_encode_no_special(self, tokenizer):
        seq = "ACDEF"
        encoded = tokenizer.encode(seq, add_special_tokens=False)
        assert len(encoded) == len(seq)
        assert encoded[0] != 0  # no cls_aa

    def test_standard_aa_tokens(self, tokenizer):
        """All 20 standard AAs map to tokens 4-23."""
        standard_aas = "LAGVSERTIDPKQNFYMHWC"
        for aa in standard_aas:
            token_id = tokenizer.vocab[aa]
            assert 4 <= token_id <= 23, f"{aa} has token_id {token_id}"

    def test_call_padding(self, tokenizer):
        result = tokenizer(["ACDEF", "GH"], padding=True, return_tensors="pt")
        ids = result["input_ids"]
        assert ids.shape == (2, 7)  # longer seq has 5+2=7 tokens
        assert ids[1, -1] == tokenizer.pad_token_id  # shorter seq padded

    def test_added_tokens_decoder(self, tokenizer):
        atd = tokenizer.added_tokens_decoder
        assert 0 in atd  # cls_aa
        assert 1 in atd  # pad
        assert 2 in atd  # eos_aa
        assert 32 in atd  # mask_aa


# ── Construction & forward ───────────────────────────────────────────


def test_construction(model):
    assert hasattr(model, "model")
    assert hasattr(model, "tokenizer")
    assert hasattr(model, "logit_formatter")
    assert model.OUTPUT_DIM == 8229
    assert model.EMB_DIM > 0


def test_forward_pass(model):
    seq_str = "ACDEF"
    encoded = model.tokenizer.encode(seq_str)
    seq_SP = torch.tensor([encoded], dtype=torch.long, device=model.device)

    log_probs = model.get_log_probs(seq_SP)

    S, P = seq_SP.shape
    assert log_probs.shape == (S, P, model.OUTPUT_DIM)
    assert log_probs.dtype == torch.float32
    assert not torch.any(torch.isnan(log_probs))
    assert torch.all(log_probs <= 0.0)

    # Normalized
    probs = torch.exp(log_probs)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_get_log_probs_from_string(model):
    seq1 = "ACDEF"
    seq2 = "GH"

    output_method = model.get_log_probs_from_string([seq1, seq2])

    # Manual
    enc1 = model.tokenizer.encode(seq1)
    enc2 = model.tokenizer.encode(seq2)
    max_len = max(len(enc1), len(enc2))
    pad = model.tokenizer.pad_token_id

    enc1_pad = enc1 + [pad] * (max_len - len(enc1))
    enc2_pad = enc2 + [pad] * (max_len - len(enc2))

    seq_SP = torch.tensor([enc1_pad, enc2_pad], dtype=torch.long, device=model.device)
    output_manual = model.get_log_probs(seq_SP)

    assert torch.allclose(output_method, output_manual)


# ── Embedding path ───────────────────────────────────────────────────


def test_embed_shape(model):
    seq_str = "ACDEF"
    encoded = model.tokenizer.encode(seq_str)
    seq_SP = torch.tensor([encoded], dtype=torch.long, device=model.device)

    emb = model.embed(seq_SP)
    S, P = seq_SP.shape
    assert emb.shape == (S, P, model.EMB_DIM)


def test_embed_matches_forward(model):
    """embedding_to_outputs(embed(seq)) ≈ forward(seq)."""
    seq_str = "ACDEF"
    encoded = model.tokenizer.encode(seq_str)
    seq_SP = torch.tensor([encoded], dtype=torch.long, device=model.device)

    with torch.no_grad():
        # Forward path
        raw_forward = model.forward(seq_SP)
        logits_forward = raw_forward.logits

        # Embedding path
        emb = model.embed(seq_SP)
        logits_emb = model.embedding_to_outputs(emb)

    assert torch.allclose(logits_forward, logits_emb, atol=1e-4)


def test_embed_gradient_flows(model):
    """Gradients flow through embed() back to the OHE input."""
    seq_str = "ACD"
    encoded = model.tokenizer.encode(seq_str)
    seq_SP = torch.tensor([encoded], dtype=torch.long, device=model.device)

    ohe = torch.nn.functional.one_hot(
        seq_SP, num_classes=model.tokenizer.vocab_size
    ).float()
    ohe.requires_grad_(True)

    emb = model.differentiable_embedding(ohe)
    loss = emb.sum()
    loss.backward()

    assert ohe.grad is not None
    assert ohe.grad.abs().sum() > 0


# ── Logit formatting ────────────────────────────────────────────────


def test_mask_token_predicts_standard_aas(model):
    """Mask token should only predict standard AA tokens (4-23)."""
    mask_input = torch.tensor(
        [[0, 32, 2]], dtype=torch.long, device=model.device
    )  # cls_aa, mask_aa, eos_aa

    log_probs = model.get_log_probs(mask_input)

    # At the mask position (index 1), only tokens 4-23 should have finite log probs
    mask_lp = log_probs[0, 1]
    finite_mask = torch.isfinite(mask_lp)

    finite_ids = torch.where(finite_mask)[0].tolist()
    assert set(finite_ids) == set(range(4, 24)), (
        f"Expected mask to predict tokens 4-23, got {finite_ids}"
    )


def test_special_tokens_predict_themselves(model):
    """Special tokens (CLS, EOS, PAD) should only predict themselves."""
    input_ids = torch.tensor([[0, 5, 23, 2]], dtype=torch.long, device=model.device)

    log_probs = model.get_log_probs(input_ids)

    # CLS position (0): only token 0 should be finite
    cls_lp = log_probs[0, 0]
    finite_cls = torch.where(torch.isfinite(cls_lp))[0].tolist()
    assert finite_cls == [0]

    # EOS position (3): only token 2 should be finite
    eos_lp = log_probs[0, 3]
    finite_eos = torch.where(torch.isfinite(eos_lp))[0].tolist()
    assert finite_eos == [2]


# ── Batching ─────────────────────────────────────────────────────────


def test_batching_consistency(model):
    """Batched log-probs approximately match individual passes."""
    seq1 = "ACDEF"
    seq2 = "GHIKL"

    batch_out = model.get_log_probs_from_string([seq1, seq2])
    out1 = model.get_log_probs_from_string([seq1])
    out2 = model.get_log_probs_from_string([seq2])

    # Probability-space comparison (more stable)
    p_batch = torch.softmax(batch_out, dim=-1)
    p1 = torch.softmax(out1, dim=-1)
    p2 = torch.softmax(out2, dim=-1)
    assert torch.allclose(p_batch[0], p1[0], atol=5e-2)
    assert torch.allclose(p_batch[1], p2[0], atol=5e-2)


# ── Temperature ──────────────────────────────────────────────────────


def test_temperature(model):
    """Higher temperature → flatter distribution at mask positions."""
    # Use mask token so the distribution is non-degenerate
    mask_id = model.tokenizer.mask_token_id
    seq_SP = torch.tensor(
        [[0, mask_id, mask_id, 2]], dtype=torch.long, device=model.device
    )  # cls_aa, mask, mask, eos_aa

    lp_cold = model.get_log_probs(seq_SP)

    with model.with_temp(2.0):
        lp_hot = model.get_log_probs(seq_SP)

    # Compare entropy at mask positions (indices 1, 2) over valid AA outputs (4-23)
    aa_slice = slice(4, 24)
    for pos in [1, 2]:
        p_cold = torch.exp(lp_cold[0, pos, aa_slice])
        p_hot = torch.exp(lp_hot[0, pos, aa_slice])
        H_cold = -(p_cold * torch.log(p_cold)).sum()
        H_hot = -(p_hot * torch.log(p_hot)).sum()
        assert H_hot > H_cold, (
            f"Position {pos}: H_hot={H_hot:.4f} <= H_cold={H_cold:.4f}"
        )


# ── Save args ────────────────────────────────────────────────────────


def test_save_args(model):
    args = model._save_args()
    assert "checkpoint" in args
    assert args["checkpoint"] == "airkingbd/dplm2_150m"


# ── Regression against upstream EsmForDPLM2 ──────────────────────────
# Reference logits generated from bytedance/dplm commit 8a2e15e
# (https://github.com/bytedance/dplm/commit/8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d)
# using EsmForDPLM2.from_pretrained("airkingbd/dplm2_150m") with type_ids
# for AA-only input. Our wrapper uses standard EsmForMaskedLM which produces
# identical outputs for sequence-only inputs (no struct tokens).


# fmt: off
# [cls_aa=0, A=5, C=23, D=13, E=9, F=18, eos_aa=2], logits[0, 1, :10] (position "A")
_REF_CASE1_POS1 = [-10.07499885559082, -11.170969009399414, -10.37201976776123, -11.116668701171875, -3.151176691055298, 2.3720688819885254, 2.439086675643921, 0.5355821847915649, 0.13429437577724457, -3.4578349590301514]
# same input, logits[0, 4, :10] (position "E")
_REF_CASE1_POS4 = [-8.860397338867188, -14.673979759216309, -11.043608665466309, -14.655830383300781, -0.7790907621383667, 1.0284886360168457, 1.3158804178237915, 0.889191746711731, 1.075209379196167, 6.461520195007324]
# [cls_aa=0, mask_aa=32, C=23, D=13, mask_aa=32, eos_aa=2], logits[0, 1, :10] (first mask)
_REF_CASE2_POS1 = [-7.893060207366943, -9.15469741821289, -10.027251243591309, -9.117449760437012, 3.103973865509033, 3.129284381866455, 3.3332717418670654, 4.177502632141113, 4.824081897735596, 1.9000431299209595]
# same input, logits[0, 4, :10] (second mask)
_REF_CASE2_POS4 = [-7.497910022735596, -9.141180992126465, -8.793717384338379, -9.170397758483887, -3.136813163757324, -3.0305776596069336, -0.2570188045501709, -2.720017910003662, -0.8788449168205261, -4.456288814544678]
# fmt: on


def test_regression_plain_sequence(model):
    """Raw logits match upstream EsmForDPLM2 on a plain AA sequence."""
    input_ids = torch.tensor(
        [[0, 5, 23, 13, 9, 18, 2]], dtype=torch.long, device=model.device
    )
    with torch.no_grad():
        logits = model.forward(input_ids).logits

    ref1 = torch.tensor(_REF_CASE1_POS1, device=model.device)
    ref4 = torch.tensor(_REF_CASE1_POS4, device=model.device)
    assert torch.allclose(logits[0, 1, :10], ref1, atol=1e-4), (
        f"pos1 diff: {(logits[0, 1, :10] - ref1).abs().max():.6f}"
    )
    assert torch.allclose(logits[0, 4, :10], ref4, atol=1e-4), (
        f"pos4 diff: {(logits[0, 4, :10] - ref4).abs().max():.6f}"
    )


def test_regression_masked_sequence(model):
    """Raw logits match upstream EsmForDPLM2 on a masked sequence."""
    input_ids = torch.tensor(
        [[0, 32, 23, 13, 32, 2]], dtype=torch.long, device=model.device
    )
    with torch.no_grad():
        logits = model.forward(input_ids).logits

    ref1 = torch.tensor(_REF_CASE2_POS1, device=model.device)
    ref4 = torch.tensor(_REF_CASE2_POS4, device=model.device)
    assert torch.allclose(logits[0, 1, :10], ref1, atol=1e-4), (
        f"pos1 diff: {(logits[0, 1, :10] - ref1).abs().max():.6f}"
    )
    assert torch.allclose(logits[0, 4, :10], ref4, atol=1e-4), (
        f"pos4 diff: {(logits[0, 4, :10] - ref4).abs().max():.6f}"
    )
