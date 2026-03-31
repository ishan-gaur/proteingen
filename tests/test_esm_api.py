"""Tests for ESM Forge API model wrappers.

These tests require network access and a valid ESM_API_KEY environment variable.
They are skipped automatically when the API key is not set.
"""

import os

import torch
import pytest

from proteingen.models.esm import ESM3API, ESMCAPI

FORGE_TOKEN = os.environ.get("FORGE_TOKEN", "")
skip_no_key = pytest.mark.skipif(not FORGE_TOKEN, reason="FORGE_TOKEN not set")


# ── ESMCAPI tests ────────────────────────────────────────────────────────


@pytest.fixture
def esmc_api():
    return ESMCAPI("esmc-300m-2024-12", token=ESM_API_KEY)


@skip_no_key
def test_esmc_api_construction(esmc_api):
    assert esmc_api.device == torch.device("cpu")
    assert esmc_api._model_name == "esmc-300m-2024-12"
    assert esmc_api.tokenizer is not None
    assert esmc_api.logit_formatter is not None


@skip_no_key
def test_esmc_api_forward(esmc_api):
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
def test_esmc_api_get_log_probs(esmc_api):
    seq_str = "ACDEF"
    encoded = esmc_api.tokenizer.encode(seq_str)
    seq_SP = torch.tensor([encoded], dtype=torch.long)

    log_probs = esmc_api.get_log_probs(seq_SP)

    S, P = seq_SP.shape
    assert log_probs.shape == (S, P, 64)
    assert not torch.any(torch.isnan(log_probs))
    assert torch.all(log_probs <= 0.0)

    # Normalized
    probs = torch.exp(log_probs)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


@skip_no_key
def test_esmc_api_get_log_probs_from_string(esmc_api):
    log_probs = esmc_api.get_log_probs_from_string(["ACDEF"])
    assert log_probs.shape[0] == 1
    assert log_probs.shape[2] == 64
    assert torch.all(log_probs <= 0.0)


@skip_no_key
def test_esmc_api_batched(esmc_api):
    """Batched input with padding."""
    log_probs = esmc_api.get_log_probs_from_string(["ACDEF", "GH"])
    assert log_probs.shape[0] == 2
    # Longer sequence determines P
    assert log_probs.shape[1] == len(esmc_api.tokenizer.encode("ACDEF"))


@skip_no_key
def test_esmc_api_temperature(esmc_api):
    seq_SP = torch.tensor([esmc_api.tokenizer.encode("ACDEF")], dtype=torch.long)

    lp_default = esmc_api.get_log_probs(seq_SP)
    with esmc_api.with_temp(0.5):
        lp_cold = esmc_api.get_log_probs(seq_SP)

    # Colder temperature → sharper distribution → higher max log prob
    assert lp_cold.max() > lp_default.max()


@skip_no_key
def test_esmc_api_no_lora():
    model = ESMCAPI("esmc-300m-2024-12", token=ESM_API_KEY)
    with pytest.raises(NotImplementedError):
        model.apply_lora()


@skip_no_key
def test_esmc_api_no_checkpointing():
    model = ESMCAPI("esmc-300m-2024-12", token=ESM_API_KEY)
    with pytest.raises(NotImplementedError):
        model._save_args()


# ── ESM3API tests ────────────────────────────────────────────────────────


@pytest.fixture
def esm3_api():
    return ESM3API("esm3-sm-open-v1", token=ESM_API_KEY)


@skip_no_key
def test_esm3_api_construction(esm3_api):
    assert esm3_api.device == torch.device("cpu")
    assert esm3_api._model_name == "esm3-sm-open-v1"
    assert esm3_api.tokenizer is not None


@skip_no_key
def test_esm3_api_forward(esm3_api):
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
def test_esm3_api_get_log_probs(esm3_api):
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
def test_esm3_api_get_log_probs_from_string(esm3_api):
    log_probs = esm3_api.get_log_probs_from_string(["ACDEF"])
    assert log_probs.shape[0] == 1
    assert log_probs.shape[2] == 64
    assert torch.all(log_probs <= 0.0)


@skip_no_key
def test_esm3_api_batched(esm3_api):
    log_probs = esm3_api.get_log_probs_from_string(["ACDEF", "GH"])
    assert log_probs.shape[0] == 2
    assert log_probs.shape[1] == len(esm3_api.tokenizer.encode("ACDEF"))


@skip_no_key
def test_esm3_api_structure_conditioning(esm3_api):
    """Structure conditioning via set_condition_."""
    # 5 residues → atom37 coords (5, 37, 3)
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
def test_esm3_api_conditioned_on_context_manager(esm3_api):
    """conditioned_on reverts state."""
    coords = torch.randn(5, 37, 3)

    assert esm3_api.observations is None
    with esm3_api.conditioned_on({"coords_RAX": coords}):
        assert esm3_api.observations is not None
    assert esm3_api.observations is None


@skip_no_key
def test_esm3_api_no_lora():
    model = ESM3API("esm3-sm-open-v1", token=ESM_API_KEY)
    with pytest.raises(NotImplementedError):
        model.apply_lora()
