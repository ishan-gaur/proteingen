import torch
import pytest
from proteingen.modeling import ESMC


@pytest.fixture
def model():
    m = ESMC()
    m.to(m.device)
    return m


def test_construction(model):
    """Model loads with expected attributes and properties."""
    assert hasattr(model, "model")
    assert hasattr(model, "tokenizer")
    assert hasattr(model, "logit_formatter")
    assert model.logit_formatter.output_dim == 64
    assert model.EMB_DIM in (960, 1152)  # 300m or 600m


def test_tokenizer_behavior(model):
    """CLS=0 at position 0, EOS=2 at end, PAD=1 for padding."""
    tokenizer = model.tokenizer
    seq = "ACDEF"
    encoded = tokenizer.encode(seq)

    assert encoded[0] == 0  # CLS
    assert encoded[-1] == 2  # EOS
    assert tokenizer.pad_token_id == 1
    assert len(encoded) == len(seq) + 2

    special_ids = {0, 1, 2}
    for t in encoded[1:-1]:
        assert t not in special_ids


def test_forward_pass(model):
    """get_log_probs returns normalized log-probs with expected shape."""
    seq_str = "ACDEF"
    encoded = model.tokenizer.encode(seq_str)
    seq_SP = torch.tensor([encoded], dtype=torch.long, device=model.device)

    log_probs = model.get_log_probs(seq_SP)

    S, P = seq_SP.shape
    assert log_probs.shape == (S, P, 64)
    assert log_probs.dtype == torch.float32

    assert not torch.any(torch.isnan(log_probs))
    assert torch.all(log_probs <= 0.0)

    # Normalized
    probs = torch.exp(log_probs)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_get_log_probs_from_string(model):
    """get_log_probs_from_string matches manual tokenize → get_log_probs."""
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


def test_batching(model):
    """Batched log-probs approximately match individual passes."""
    seq1 = "ACDEF"
    seq2 = "GHACC"

    batch_out = model.get_log_probs_from_string([seq1, seq2])
    out1 = model.get_log_probs_from_string([seq1])
    out2 = model.get_log_probs_from_string([seq2])

    # Log-prob space comparison
    assert torch.allclose(batch_out[0], out1[0], atol=1)

    len2 = out2.shape[1]
    assert torch.allclose(batch_out[1, :len2], out2[0], atol=1)

    # Probability space comparison (tighter tolerance)
    p_out_batch = torch.softmax(batch_out, dim=-1)
    p_out1 = torch.softmax(out1, dim=-1)
    p_out2 = torch.softmax(out2, dim=-1)
    assert torch.allclose(p_out_batch[0], p_out1[0], atol=5e-2)
    assert torch.allclose(p_out_batch[1, :len2], p_out2[0], atol=5e-2)
