"""Tests for TAG guidance projection across mismatched token/OHE spaces."""

from types import SimpleNamespace

import torch
from torch import nn

from proteingen.modeling import GenerativeModel, PassThroughLogitFormatter
from proteingen.modeling import PredictiveModel, binary_logits
from proteingen.modeling import TAG, LinearGuidanceProjection


def make_tokenizer(
    vocab: dict[str, int],
    *,
    mask_token_id: int | None = None,
    cls_token_id: int | None = None,
    eos_token_id: int | None = None,
    unk_token_id: int | None = None,
):
    return SimpleNamespace(
        vocab=vocab,
        vocab_size=len(vocab),
        mask_token_id=mask_token_id,
        cls_token_id=cls_token_id,
        eos_token_id=eos_token_id,
        unk_token_id=unk_token_id,
    )


class ConstantBackbone(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.bias_T = nn.Parameter(torch.zeros(output_dim))

    def forward(self, seq_SP: torch.LongTensor, **kwargs) -> torch.FloatTensor:
        S, P = seq_SP.shape
        return self.bias_T.view(1, 1, -1).expand(S, P, -1)


class ToyPredictor(PredictiveModel):
    def __init__(self, tokenizer, token_ohe_basis_TK: torch.FloatTensor):
        super().__init__(tokenizer=tokenizer)
        self.target = True
        self.register_buffer("_token_ohe_basis_TK", token_ohe_basis_TK)
        self.w_K = nn.Parameter(torch.tensor([1.0, -0.5]))

    def token_ohe_basis(self) -> torch.FloatTensor:
        return self._token_ohe_basis_TK

    def forward(self, ohe_seq_SPK: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return (ohe_seq_SPK * self.w_K).sum(dim=(1, 2))

    def format_raw_to_logits(self, raw_output, seq_SPK, **kwargs) -> torch.FloatTensor:
        return binary_logits(raw_output, self.target)


def _make_projection_setup():
    gen_tokenizer = make_tokenizer(
        {"<cls>": 0, "A": 1, "B": 2, "<mask>": 3, "<eos>": 4},
        mask_token_id=3,
        cls_token_id=0,
        eos_token_id=4,
    )
    pred_tokenizer = make_tokenizer(
        {"A": 0, "B": 1, "<mask>": 2},
        mask_token_id=2,
    )
    # Predictor OHE basis has an explicit mask token that maps to all zeros.
    pred_token_ohe_basis_TK = torch.tensor(
        [
            [1.0, 0.0],  # A
            [0.0, 1.0],  # B
            [0.0, 0.0],  # <mask>
        ]
    )
    projection = LinearGuidanceProjection(
        tokenizer_gen=gen_tokenizer,
        tokenizer_pred=pred_tokenizer,
        pred_token_ohe_basis_TK=pred_token_ohe_basis_TK,
    )
    return gen_tokenizer, pred_tokenizer, projection


def test_prepare_strips_special_positions_and_maps_tokens():
    _, _, projection = _make_projection_setup()
    seq_gen_SP = torch.tensor([[0, 1, 3, 4]])  # <cls> A <mask> <eos>
    logp_gen_SPT = torch.zeros(1, 4, 5)
    logp_gen_SPT[0, 2, 2] = 10.0  # fill masked position with B in clean-classifier mode

    prepared = projection.prepare(
        seq_gen_SP,
        logp_gen_SPT,
        use_clean_classifier=True,
    )

    assert torch.equal(prepared.pred_pos_to_gen_pos_P, torch.tensor([1, 2]))
    assert torch.equal(prepared.baseline_gen_SP, torch.tensor([[1, 2]]))
    assert torch.equal(prepared.seq_pred_SP, torch.tensor([[0, 1]]))


def test_grad_to_gen_delta_implements_linear_taylor_term():
    _, _, projection = _make_projection_setup()
    seq_gen_SP = torch.tensor([[0, 1, 2, 4]])  # <cls> A B <eos>
    prepared = projection.prepare(
        seq_gen_SP,
        torch.zeros(1, 4, 5),
        use_clean_classifier=False,
    )
    grad_pred_SPK = torch.tensor(
        [
            [
                [2.0, 3.0],
                [5.0, 7.0],
            ]
        ]
    )

    delta_gen_SPT = projection.grad_to_gen_delta(
        grad_pred_SPK,
        prepared,
        gen_output_dim=7,
    )

    # Positions not scored by predictor (CLS/EOS) remain zero.
    assert torch.equal(delta_gen_SPT[:, 0, :], torch.zeros(1, 7))
    assert torch.equal(delta_gen_SPT[:, 3, :], torch.zeros(1, 7))

    # At position 1 (baseline=A): delta(A)=0, delta(B)=+1.
    assert torch.isclose(delta_gen_SPT[0, 1, 1], torch.tensor(0.0))
    assert torch.isclose(delta_gen_SPT[0, 1, 2], torch.tensor(1.0))

    # At position 2 (baseline=B): delta(A)=-2, delta(B)=0.
    assert torch.isclose(delta_gen_SPT[0, 2, 1], torch.tensor(-2.0))
    assert torch.isclose(delta_gen_SPT[0, 2, 2], torch.tensor(0.0))

    # Output dims beyond generator vocab are untouched.
    assert torch.equal(delta_gen_SPT[..., 5:], torch.zeros_like(delta_gen_SPT[..., 5:]))


def test_tag_with_projection_returns_gen_space_logits():
    gen_tokenizer, pred_tokenizer, projection = _make_projection_setup()
    gen_model = GenerativeModel(
        model=ConstantBackbone(output_dim=6),
        tokenizer=gen_tokenizer,
        logit_formatter=PassThroughLogitFormatter(),
    )
    pred_model = ToyPredictor(
        tokenizer=pred_tokenizer,
        token_ohe_basis_TK=projection.pred_token_ohe_basis_TK,
    )
    tag = TAG(
        gen_model=gen_model,
        pred_model=pred_model,
        use_clean_classifier=True,
        projection=projection,
    )
    seq_gen_SP = torch.tensor([[0, 1, 3, 4]])

    guided_logp_SPT = tag(seq_gen_SP)
    base_logp_SPT = gen_model.get_log_probs(seq_gen_SP)

    assert guided_logp_SPT.shape == base_logp_SPT.shape == (1, 4, 6)
    assert torch.isfinite(guided_logp_SPT).all()
    assert torch.allclose(guided_logp_SPT[:, [0, 3], :], base_logp_SPT[:, [0, 3], :])
