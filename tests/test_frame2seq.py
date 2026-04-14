from pathlib import Path
import urllib.request

import pytest
import torch

from frame2seq.utils.pdb2input import get_inference_inputs

from protstar.models import Frame2seq


PDB_1YCR = Path("/tmp/test_1YCR.pdb")


@pytest.fixture(scope="module")
def pdb_path() -> Path:
    if not PDB_1YCR.exists():
        urllib.request.urlretrieve(
            "https://files.rcsb.org/download/1YCR.pdb", str(PDB_1YCR)
        )
    return PDB_1YCR


@pytest.fixture(scope="module")
def model() -> Frame2seq:
    return Frame2seq()


@pytest.fixture(scope="module")
def conditioning(pdb_path: Path) -> dict:
    return Frame2seq.condition_from_pdb(str(pdb_path), "A")


@pytest.fixture
def conditioned_model(model: Frame2seq, conditioning: dict):
    with model.conditioned_on(conditioning):
        yield model


def test_construction(model: Frame2seq):
    assert model.OUTPUT_DIM == 22
    assert model.EMB_DIM == 384
    assert len(model.model.models) == 3


def test_tokenizer(model: Frame2seq):
    tok = model.tokenizer
    assert tok.vocab["A"] == 0
    assert tok.vocab["Y"] == 19
    assert tok.vocab["X"] == 20
    assert tok.mask_token_id == 21


def test_requires_conditioning(model: Frame2seq):
    tokens_SP = torch.full((1, 10), model.tokenizer.mask_token_id, dtype=torch.long)
    with pytest.raises(ValueError, match="requires structure conditioning"):
        model.get_log_probs(tokens_SP)


def test_preprocess_observations(conditioned_model: Frame2seq):
    obs = conditioned_model.observations
    assert obs is not None
    assert obs["X_LA3"].shape[-2:] == (5, 3)
    assert obs["seq_mask_L"].dtype == torch.bool


def test_forward_shape(conditioned_model: Frame2seq):
    L = conditioned_model.observations["X_LA3"].shape[0]
    tokens_SP = torch.full(
        (1, L), conditioned_model.tokenizer.mask_token_id, dtype=torch.long
    )

    with torch.no_grad():
        logits_SPT = conditioned_model.forward(tokens_SP)

    assert logits_SPT.shape == (1, L, conditioned_model.OUTPUT_DIM)


def test_embed_matches_forward(conditioned_model: Frame2seq):
    L = conditioned_model.observations["X_LA3"].shape[0]
    tokens_SP = torch.full(
        (1, L), conditioned_model.tokenizer.mask_token_id, dtype=torch.long
    )

    with torch.no_grad():
        emb_SPE = conditioned_model.embed(tokens_SP)
        logits_from_emb_SPT = conditioned_model.embedding_to_outputs(emb_SPE)
        logits_forward_SPT = conditioned_model.forward(tokens_SP)

    finite = torch.isfinite(logits_from_emb_SPT) & torch.isfinite(logits_forward_SPT)
    assert torch.allclose(
        logits_from_emb_SPT[finite], logits_forward_SPT[finite], atol=1e-5
    )


def test_gradient_flow(conditioned_model: Frame2seq):
    L = conditioned_model.observations["X_LA3"].shape[0]
    tokens_SP = torch.full(
        (1, L), conditioned_model.tokenizer.vocab["A"], dtype=torch.long
    )

    ohe_SPT = torch.nn.functional.one_hot(
        tokens_SP, num_classes=conditioned_model.tokenizer.vocab_size
    ).float()
    ohe_SPT.requires_grad_(True)

    emb_SPE = conditioned_model.differentiable_embedding(ohe_SPT)
    loss = emb_SPE.sum()
    loss.backward()

    assert ohe_SPT.grad is not None
    assert ohe_SPT.grad.norm() > 0


def test_log_probs_valid(conditioned_model: Frame2seq):
    L = conditioned_model.observations["X_LA3"].shape[0]
    tokens_SP = torch.full(
        (1, L), conditioned_model.tokenizer.mask_token_id, dtype=torch.long
    )

    with torch.no_grad():
        log_probs_SPT = conditioned_model.get_log_probs(tokens_SP)

    assert log_probs_SPT.shape == (1, L, conditioned_model.OUTPUT_DIM)
    assert torch.all(log_probs_SPT <= 0.0)
    assert not torch.isnan(log_probs_SPT).any()

    probs_SPT = torch.exp(log_probs_SPT)
    assert torch.allclose(
        probs_SPT.sum(dim=-1), torch.ones_like(probs_SPT[..., 0]), atol=1e-4
    )

    assert (log_probs_SPT[..., 20] == float("-inf")).all()
    assert (log_probs_SPT[..., 21] == float("-inf")).all()


def test_temperature(conditioned_model: Frame2seq):
    L = conditioned_model.observations["X_LA3"].shape[0]
    tokens_SP = torch.full(
        (1, L), conditioned_model.tokenizer.mask_token_id, dtype=torch.long
    )

    with torch.no_grad():
        lp_cold_SPT = conditioned_model.get_log_probs(tokens_SP)
        with conditioned_model.with_temp(2.0):
            lp_hot_SPT = conditioned_model.get_log_probs(tokens_SP)

    aa_slice = slice(0, 20)
    p_cold_SPT = torch.exp(lp_cold_SPT[:, :, aa_slice])
    p_hot_SPT = torch.exp(lp_hot_SPT[:, :, aa_slice])

    H_cold = -(p_cold_SPT * torch.log(p_cold_SPT)).sum(-1).mean()
    H_hot = -(p_hot_SPT * torch.log(p_hot_SPT)).sum(-1).mean()
    assert H_hot > H_cold


def test_batching_consistency(conditioned_model: Frame2seq):
    L = conditioned_model.observations["X_LA3"].shape[0]
    mask_id = conditioned_model.tokenizer.mask_token_id

    single_SP = torch.full((1, L), mask_id, dtype=torch.long)
    batch_SP = torch.full((2, L), mask_id, dtype=torch.long)
    batch_SP[1, :5] = conditioned_model.tokenizer.vocab["A"]

    with torch.no_grad():
        lp_single_SPT = conditioned_model.get_log_probs(single_SP)
        lp_batch_SPT = conditioned_model.get_log_probs(batch_SP)

    assert torch.allclose(lp_single_SPT[0], lp_batch_SPT[0], atol=1e-5)


def test_conditioned_on_reverts(model: Frame2seq, conditioning: dict):
    assert model.observations is None
    with model.conditioned_on(conditioning):
        assert model.observations is not None
    assert model.observations is None


def test_matches_upstream_frame2seq_logits(
    conditioned_model: Frame2seq, conditioning: dict
):
    seq_mask_BP, _, X_BPA3 = get_inference_inputs(
        str(conditioning["pdb_path"]), conditioning["chain_id"]
    )

    input_ohe_BPU = torch.zeros(1, seq_mask_BP.shape[1], 21, dtype=torch.float32)
    input_ohe_BPU[:, :, 20] = 1.0

    with torch.no_grad():
        ref_logits_BPU = sum(
            model_i(X_BPA3, seq_mask_BP, input_ohe_BPU)
            for model_i in conditioned_model.model.models
        ) / len(conditioned_model.model.models)

    mask_tokens_SP = torch.full(
        (1, seq_mask_BP.shape[1]),
        conditioned_model.tokenizer.mask_token_id,
        dtype=torch.long,
    )

    with torch.no_grad():
        our_logits_BPT = conditioned_model.forward(mask_tokens_SP)

    assert torch.allclose(our_logits_BPT[:, :, :21], ref_logits_BPU, atol=1e-5)
