import torch
from torch import nn
from torch.nn import functional as F
from dfm.generative_modeling import TransitionModel
from dfm.predictive_modeling import PredictiveModel
from transformers import PreTrainedTokenizerBase
from contextlib import contextmanager
from typing import Optional, List
import warnings


class TokenizerTranslator(nn.Module):
    def __init__(
        self,
        tokenizer_src: PreTrainedTokenizerBase,
        tokenizer_tgt: PreTrainedTokenizerBase,
        target_output_dim: Optional[int] = None,
    ):
        super().__init__()
        if target_output_dim is None:
            target_output_dim = tokenizer_tgt.vocab_size
        convert_SrcTgt = torch.zeros(tokenizer_src.vocab_size, target_output_dim)
        src_vocab = set(tokenizer_src.vocab.keys())
        tgt_vocab = set(tokenizer_tgt.vocab.keys())
        shared_vocab = src_vocab & tgt_vocab
        if not shared_vocab:
            raise ValueError(
                f"Source and target tokenizer vocabs have no tokens in common"
            )
        self.vocab_eq = tgt_vocab.issubset(src_vocab)

        for tok in shared_vocab:
            i = tokenizer_src.vocab[tok]
            j = tokenizer_tgt.vocab[tok]
            convert_SrcTgt[i, j] = 1.0
        self.register_buffer("convert_SrcTgt", convert_SrcTgt)

        # Detect whether src tokenizer adds CLS/EOS that tgt tokenizer doesn't
        src_has_cls = getattr(tokenizer_src, "cls_token_id", None) is not None
        src_has_eos = getattr(tokenizer_src, "eos_token_id", None) is not None
        tgt_has_cls = getattr(tokenizer_tgt, "cls_token_id", None) is not None
        tgt_has_eos = getattr(tokenizer_tgt, "eos_token_id", None) is not None
        self._strip_prefix = 1 if (src_has_cls and not tgt_has_cls) else 0
        self._strip_suffix = 1 if (src_has_eos and not tgt_has_eos) else 0

    def forward(self, x_SPSrc: torch.Tensor) -> torch.Tensor:
        # Strip CLS/EOS positions if src tokenizer adds them but tgt doesn't
        if self._strip_suffix > 0:
            x_SPSrc = x_SPSrc[:, self._strip_prefix : -self._strip_suffix, :]
        elif self._strip_prefix > 0:
            x_SPSrc = x_SPSrc[:, self._strip_prefix :, :]
        x_SPTgt = torch.matmul(x_SPSrc, self.convert_SrcTgt[None, ...])
        return x_SPTgt.to(x_SPSrc.dtype)

    def reverse(self, x_SPTgt: torch.Tensor) -> torch.Tensor:
        if not self.vocab_eq:
            warnings.warn(
                "Tokens are being converted to a target vocab that does not include the src as a subset",
                UserWarning,
            )
        x_SPSrc = torch.matmul(x_SPTgt, self.convert_SrcTgt.T[None, ...])
        return x_SPSrc.to(x_SPTgt.dtype)


def _fill_masked_with_argmax(seq_SP, logp_gen_SPT, mask_token_id, vocab_size):
    """Replace mask tokens with gen model's argmax predictions."""
    mask = seq_SP == mask_token_id
    if not mask.any():
        return seq_SP
    gen_argmax = logp_gen_SPT[..., :vocab_size].argmax(dim=-1)
    filled = seq_SP.clone()
    filled[mask] = gen_argmax[mask]
    return filled


class TAG(TransitionModel):
    """Token-level Autoregressive Guidance.

    Combines a generative transition model with a predictive model via Bayes' rule.
    Uses gradients through the predictive model's OHE to shift transition logits.

    Currently requires gen and pred models to share the same tokenizer.
    """
    # TODO[pi] cross-tokenizer support: when gen and pred tokenizers differ,
    # need to convert token IDs discretely, get grad in pred space, then
    # TokenizerTranslator.reverse() back to gen space.

    def __init__(
        self,
        gen_model: TransitionModel,
        pred_model: PredictiveModel,
        argmax_masked_positions: bool = False,
    ):
        super().__init__(
            model=gen_model.model,
            tokenizer=gen_model.tokenizer,
            logit_formatter=gen_model.logit_formatter,
        )
        self.gen_model = gen_model
        self.pred_model = pred_model
        self.argmax_masked_positions = argmax_masked_positions
        assert gen_model.tokenizer is pred_model.tokenizer or \
            gen_model.tokenizer.vocab == pred_model.tokenizer.vocab, \
            "TAG currently requires gen and pred models to share the same tokenizer"

    def forward(self, seq_SP: torch.LongTensor):
        logp_xtilde_g_x_SPT = self.gen_model.get_log_probs(seq_SP)
        pred_input = seq_SP
        if self.argmax_masked_positions:
            pred_input = _fill_masked_with_argmax(
                seq_SP, logp_xtilde_g_x_SPT,
                self.gen_model.tokenizer.mask_token_id,
                self.gen_model.tokenizer.vocab_size,
            )
        grad_SPV = self.pred_model.grad_log_prob(pred_input)
        # TODO[pi] handle cross-tokenizer grad dimension mismatch properly —
        # zero-padding assumes vocab tokens align at the same indices, which is
        # only true when gen and pred share the same tokenizer. For the
        # cross-tokenizer case, need TokenizerTranslator.reverse() on the grad.
        if grad_SPV.shape[-1] < logp_xtilde_g_x_SPT.shape[-1]:
            grad_SPV = F.pad(grad_SPV, (0, logp_xtilde_g_x_SPT.shape[-1] - grad_SPV.shape[-1]))
        return grad_SPV + logp_xtilde_g_x_SPT


class DEG(TransitionModel):
    # TAG and DEG are basically ways to efficiently compute the vector p(y|x_const)
    # Therefore, we don't make the predictive models deal with that kind of query
    def __init__(
        self,
        gen_model: TransitionModel,
        pred_model: PredictiveModel,
        argmax_masked_positions: bool = False,
    ):
        super().__init__(
            model=gen_model.model,
            tokenizer=gen_model.tokenizer,
            logit_formatter=gen_model.logit_formatter,
        )
        # main stipulation here is that the predictive model has to take OHE as input
        self.gen_model = gen_model
        self.pred_model = pred_model
        self.argmax_masked_positions = argmax_masked_positions
        self.positions_to_score_S = None

    # TODO[pi] with all these different things we have to mix in, I'm wondering if
    # the context manager approach was really the right one
    @contextmanager
    def at_position(self, positions_to_score_S: List[int]):
        """
        positions_to_score_S is a list that gives the index at each sequence to try to sample
        If a sequence does not need to be sampled, pass None for that index in the list
        """
        old = self.positions_to_score_S
        self.positions_to_score_S = positions_to_score_S
        try:
            yield self
        finally:
            self.positions_to_score_S = old

    # TODO[pi] need forward that is fully batched for predictor, sequence-wise batched, and does things one at a time
    def forward(self, seq_SP: torch.LongTensor):
        if self.positions_to_score_S is None:
            raise ValueError(
                "Need to call ``self.at_position(positions_to_score_S)`` to provide the position to score for each sequence"
            )
        logp_xtilde_g_x_SPT = self.gen_model.get_log_probs(seq_SP)
        logp_y_g_xtilde_SPT = torch.zeros_like(logp_xtilde_g_x_SPT)
        n_tok = self.tokenizer.vocab_size
        mask_token_id = self.gen_model.tokenizer.mask_token_id
        for s, p in enumerate(self.positions_to_score_S):
            if p is None:
                continue
            base = seq_SP[s].clone()
            if self.argmax_masked_positions:
                base = _fill_masked_with_argmax(
                    base.unsqueeze(0), logp_xtilde_g_x_SPT[s].unsqueeze(0),
                    mask_token_id, n_tok,
                ).squeeze(0)
            # for simplicity just try all possible tokens--including special ones TODO[pi] we could instead use the
            # logitformatter mask in order to only try the valid transitions here
            seq_XP = (
                base.unsqueeze(0).repeat(n_tok, 1)
            )  # X is the index over tokens we're trying
            seq_XP[:, p] = torch.arange(n_tok, device=seq_SP.device)
            with torch.no_grad():
                logp_y_g_xtilde_X = self.pred_model.get_log_probs(seq_XP)
            logp_y_g_xtilde_SPT[s, p, :n_tok] = logp_y_g_xtilde_X
            # Don't need to take care of making the others -inf since the logit_formatter will take care of the invalid ones (including the invalid ones we tested lol)
        return logp_y_g_xtilde_SPT + logp_xtilde_g_x_SPT
