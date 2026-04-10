"""Tests for ESMForgeAPI model wrapper.

These tests require network access and a valid FORGE_TOKEN environment variable.
They are skipped automatically when the token is not set.
"""

import os

import torch
import pytest

from protstar.models.esm import ESMForgeAPI

FORGE_TOKEN = os.environ.get("FORGE_TOKEN", "")
skip_no_key = pytest.mark.skipif(not FORGE_TOKEN, reason="FORGE_TOKEN not set")


# ── ESMC via Forge ───────────────────────────────────────────────────────


@pytest.fixture
def esmc_api():
    return ESMForgeAPI("esmc-300m-2024-12", token=FORGE_TOKEN)


@skip_no_key
def test_esmc_construction(esmc_api):
    assert esmc_api.device == torch.device("cpu")
    assert esmc_api._model_name == "esmc-300m-2024-12"
    assert not esmc_api._is_esm3


@skip_no_key
def test_esmc_forward(esmc_api):
    seq_str = "ACDEF"
    encoded = esmc_api.tokenizer.encode(seq_str)
    seq_SP = torch.tensor([encoded], dtype=torch.long)

    with torch.no_grad():
        logits = esmc_api.forward(seq_SP)

    S, P = seq_SP.shape
    assert logits.shape == (S, P, 64)
    assert logits.dtype == torch.float32
    assert not torch.any(torch.isnan(logits))


@skip_no_key
def test_esmc_get_log_probs(esmc_api):
    seq_str = "ACDEF"
    encoded = esmc_api.tokenizer.encode(seq_str)
    seq_SP = torch.tensor([encoded], dtype=torch.long)

    log_probs = esmc_api.get_log_probs(seq_SP)

    S, P = seq_SP.shape
    assert log_probs.shape == (S, P, 64)
    assert not torch.any(torch.isnan(log_probs))
    assert torch.all(log_probs <= 0.0)

    probs = torch.exp(log_probs)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


@skip_no_key
def test_esmc_get_log_probs_from_string(esmc_api):
    log_probs = esmc_api.get_log_probs_from_string(["ACDEF"])
    assert log_probs.shape[0] == 1
    assert log_probs.shape[2] == 64
    assert torch.all(log_probs <= 0.0)


@skip_no_key
def test_esmc_batched(esmc_api):
    """Batched input with padding."""
    log_probs = esmc_api.get_log_probs_from_string(["ACDEF", "GH"])
    assert log_probs.shape[0] == 2
    assert log_probs.shape[1] == len(esmc_api.tokenizer.encode("ACDEF"))


@skip_no_key
def test_esmc_temperature(esmc_api):
    mask_id = esmc_api.tokenizer.mask_token_id
    ids = esmc_api.tokenizer.encode("ACDEF")
    ids[2] = mask_id  # mask one position so distribution is non-degenerate
    seq_SP = torch.tensor([ids], dtype=torch.long)

    lp_default = esmc_api.get_log_probs(seq_SP)
    with esmc_api.with_temp(0.5):
        lp_cold = esmc_api.get_log_probs(seq_SP)

    # Only compare at the masked position — unmasked positions are one-hot
    mask_pos = (seq_SP[0] == mask_id).nonzero(as_tuple=True)[0]
    assert lp_cold[0, mask_pos].max() > lp_default[0, mask_pos].max()


@skip_no_key
def test_esmc_no_structure_conditioning(esmc_api):
    coords = torch.randn(5, 37, 3)
    with pytest.raises(AssertionError, match="Structure conditioning not supported"):
        esmc_api.set_condition_({"coords_RAX": coords})


# ── ESM3 via Forge ───────────────────────────────────────────────────────


@pytest.fixture
def esm3_api():
    return ESMForgeAPI("esm3-sm-open-v1", token=FORGE_TOKEN)


@skip_no_key
def test_esm3_construction(esm3_api):
    assert esm3_api.device == torch.device("cpu")
    assert esm3_api._model_name == "esm3-sm-open-v1"
    assert esm3_api._is_esm3


@skip_no_key
def test_esm3_forward(esm3_api):
    seq_str = "ACDEF"
    encoded = esm3_api.tokenizer.encode(seq_str)
    seq_SP = torch.tensor([encoded], dtype=torch.long)

    with torch.no_grad():
        logits = esm3_api.forward(seq_SP)

    S, P = seq_SP.shape
    assert logits.shape == (S, P, 64)
    assert logits.dtype == torch.float32
    assert not torch.any(torch.isnan(logits))


@skip_no_key
def test_esm3_get_log_probs(esm3_api):
    seq_str = "ACDEF"
    encoded = esm3_api.tokenizer.encode(seq_str)
    seq_SP = torch.tensor([encoded], dtype=torch.long)

    log_probs = esm3_api.get_log_probs(seq_SP)

    S, P = seq_SP.shape
    assert log_probs.shape == (S, P, 64)
    assert not torch.any(torch.isnan(log_probs))
    assert torch.all(log_probs <= 0.0)

    probs = torch.exp(log_probs)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


@skip_no_key
def test_esm3_get_log_probs_from_string(esm3_api):
    log_probs = esm3_api.get_log_probs_from_string(["ACDEF"])
    assert log_probs.shape[0] == 1
    assert log_probs.shape[2] == 64
    assert torch.all(log_probs <= 0.0)


@skip_no_key
def test_esm3_batched(esm3_api):
    log_probs = esm3_api.get_log_probs_from_string(["ACDEF", "GH"])
    assert log_probs.shape[0] == 2
    assert log_probs.shape[1] == len(esm3_api.tokenizer.encode("ACDEF"))


@skip_no_key
def test_esm3_structure_conditioning(esm3_api):
    coords = torch.randn(5, 37, 3)

    esm3_api.set_condition_({"coords_RAX": coords})
    assert esm3_api.observations is not None
    assert "structure_tokens" in esm3_api.observations
    assert "coordinates" in esm3_api.observations

    seq_str = "ACDEF"
    encoded = esm3_api.tokenizer.encode(seq_str)
    seq_SP = torch.tensor([encoded], dtype=torch.long)
    log_probs = esm3_api.get_log_probs(seq_SP)
    assert log_probs.shape == (1, len(encoded), 64)


@skip_no_key
def test_esm3_conditioned_on_context_manager(esm3_api):
    coords = torch.randn(5, 37, 3)

    assert esm3_api.observations is None
    with esm3_api.conditioned_on({"coords_RAX": coords}):
        assert esm3_api.observations is not None
    assert esm3_api.observations is None


# ── Shared: unsupported operations ───────────────────────────────────────


@skip_no_key
def test_no_lora():
    model = ESMForgeAPI("esmc-300m-2024-12", token=FORGE_TOKEN)
    with pytest.raises(NotImplementedError):
        model.apply_lora()


@skip_no_key
def test_no_checkpointing():
    model = ESMForgeAPI("esmc-300m-2024-12", token=FORGE_TOKEN)
    with pytest.raises(NotImplementedError):
        model._save_args()
