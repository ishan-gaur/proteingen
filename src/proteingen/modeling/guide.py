import torch
from torch import nn
from .generative_modeling import GenerativeModel
from .predictive_modeling import PredictiveModel
from transformers import PreTrainedTokenizerBase
from contextlib import contextmanager
from typing import Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class PreparedGuidanceInput:
    """Inputs needed to convert predictor-space gradients into gen-logit deltas."""

    seq_pred_SP: torch.LongTensor
    pred_pos_to_gen_pos_P: torch.LongTensor
    baseline_gen_SP: torch.LongTensor
    gen_length: int


class GuidanceProjection(nn.Module, ABC):
    """Maps between predictor OHE-gradient space and generative model logit space."""

    @abstractmethod
    def prepare(
        self,
        seq_gen_SP: torch.LongTensor,
        logp_gen_SPT: torch.FloatTensor,
        *,
        use_clean_classifier: bool,
        n_samples: int = 1,
    ) -> PreparedGuidanceInput:
        """Build predictor input tokens and Taylor baseline in gen token space."""
        ...

    @abstractmethod
    def grad_to_gen_delta(
        self,
        grad_pred_SPK: torch.FloatTensor,
        prepared: PreparedGuidanceInput,
        *,
        gen_output_dim: int,
    ) -> torch.FloatTensor:
        """Project predictor-space gradients into generator logit space."""
        ...


class LinearGuidanceProjection(GuidanceProjection):
    """Linear token-space projection for TAG.

    Uses a fixed map ``M[T_gen, K_pred]`` where each gen token's row is the
    predictor-OHE representation of that token. Given predictor gradient ``g``,
    TAG's first-order term at each position is:

        ``delta(t) = g · (M[t] - M[baseline])``

    where ``baseline`` is the current token (or argmax-filled token when using
    clean-classifier mode).
    """

    def __init__(
        self,
        tokenizer_gen: PreTrainedTokenizerBase,
        tokenizer_pred: PreTrainedTokenizerBase,
        pred_token_ohe_basis_TK: torch.FloatTensor,
        fallback_pred_token_id: Optional[int] = None,
        strip_prefix: Optional[int] = None,
        strip_suffix: Optional[int] = None,
    ):
        super().__init__()
        self.tokenizer_gen = tokenizer_gen
        self.tokenizer_pred = tokenizer_pred

        if pred_token_ohe_basis_TK.ndim != 2:
            raise ValueError(
                "pred_token_ohe_basis_TK must have shape (pred_vocab_size, pred_ohe_dim)"
            )
        if pred_token_ohe_basis_TK.shape[0] != tokenizer_pred.vocab_size:
            raise ValueError(
                "pred_token_ohe_basis_TK first dimension must match predictor tokenizer vocab_size"
            )

        fallback_id = fallback_pred_token_id
        if fallback_id is None:
            fallback_id = getattr(tokenizer_pred, "mask_token_id", None)
        if fallback_id is None:
            fallback_id = getattr(tokenizer_pred, "unk_token_id", None)

        src_vocab = tokenizer_gen.vocab
        tgt_vocab = tokenizer_pred.vocab
        gen_to_pred_idx_T = torch.full(
            (tokenizer_gen.vocab_size,),
            fill_value=-1,
            dtype=torch.long,
        )
        for tok, i_gen in src_vocab.items():
            if tok in tgt_vocab:
                gen_to_pred_idx_T[i_gen] = tgt_vocab[tok]

        unmapped_mask = gen_to_pred_idx_T < 0
        if unmapped_mask.any():
            if fallback_id is None:
                idx_to_tok = {idx: tok for tok, idx in src_vocab.items()}
                missing = [
                    idx_to_tok.get(i, f"<idx:{i}>")
                    for i in torch.where(unmapped_mask)[0].tolist()[:10]
                ]
                raise ValueError(
                    "No fallback predictor token available for unmapped generator tokens. "
                    f"Example unmapped tokens: {missing}"
                )
            gen_to_pred_idx_T[unmapped_mask] = int(fallback_id)

        self.register_buffer("gen_to_pred_idx_T", gen_to_pred_idx_T)
        self.register_buffer("pred_token_ohe_basis_TK", pred_token_ohe_basis_TK.float())
        self.register_buffer(
            "gen_to_pred_ohe_TK",
            self.pred_token_ohe_basis_TK[self.gen_to_pred_idx_T],
        )

        if strip_prefix is None:
            src_has_cls = getattr(tokenizer_gen, "cls_token_id", None) is not None
            tgt_has_cls = getattr(tokenizer_pred, "cls_token_id", None) is not None
            strip_prefix = 1 if (src_has_cls and not tgt_has_cls) else 0
        if strip_suffix is None:
            src_has_eos = getattr(tokenizer_gen, "eos_token_id", None) is not None
            tgt_has_eos = getattr(tokenizer_pred, "eos_token_id", None) is not None
            strip_suffix = 1 if (src_has_eos and not tgt_has_eos) else 0

        self._strip_prefix = int(strip_prefix)
        self._strip_suffix = int(strip_suffix)

    def _pred_window(self, seq_gen_SP: torch.LongTensor) -> tuple[int, int]:
        start = self._strip_prefix
        end = seq_gen_SP.size(1) - self._strip_suffix
        if end < start:
            raise ValueError(
                f"Invalid strip configuration: start={start}, end={end}, sequence_length={seq_gen_SP.size(1)}"
            )
        return start, end

    def prepare(
        self,
        seq_gen_SP: torch.LongTensor,
        logp_gen_SPT: torch.FloatTensor,
        *,
        use_clean_classifier: bool,
        n_samples: int = 1,
    ) -> PreparedGuidanceInput:
        seq_for_grad_SP = seq_gen_SP
        if use_clean_classifier:
            if n_samples > 1:
                raise NotImplementedError("n_samples > 1 not implemented for TAG yet")
            seq_for_grad_SP = _fill_masked_with_argmax(
                seq_gen_SP,
                logp_gen_SPT,
                getattr(self.tokenizer_gen, "mask_token_id", None),
                self.tokenizer_gen.vocab_size,
            )

        start, end = self._pred_window(seq_gen_SP)
        seq_for_pred_SP = seq_for_grad_SP[:, start:end]
        seq_pred_SP = self.gen_to_pred_idx_T[seq_for_pred_SP]
        pred_pos_to_gen_pos_P = torch.arange(
            start, end, device=seq_gen_SP.device, dtype=torch.long
        )
        return PreparedGuidanceInput(
            seq_pred_SP=seq_pred_SP,
            pred_pos_to_gen_pos_P=pred_pos_to_gen_pos_P,
            baseline_gen_SP=seq_for_pred_SP,
            gen_length=seq_gen_SP.size(1),
        )

    def grad_to_gen_delta(
        self,
        grad_pred_SPK: torch.FloatTensor,
        prepared: PreparedGuidanceInput,
        *,
        gen_output_dim: int,
    ) -> torch.FloatTensor:
        pred_ohe_dim = self.gen_to_pred_ohe_TK.shape[1]
        if grad_pred_SPK.shape[-1] != pred_ohe_dim:
            raise ValueError(
                f"Predictor grad dim ({grad_pred_SPK.shape[-1]}) does not match projection pred_ohe_dim ({pred_ohe_dim})"
            )

        gen_vocab_size = self.tokenizer_gen.vocab_size
        if gen_output_dim < gen_vocab_size:
            raise ValueError(
                f"gen_output_dim ({gen_output_dim}) must be >= generator vocab_size ({gen_vocab_size})"
            )

        # score_gen[s, p, t] = g[s, p, :] · M[t, :]
        score_gen_vocab_SPT = torch.einsum(
            "spk,tk->spt", grad_pred_SPK, self.gen_to_pred_ohe_TK
        )

        # TAG Taylor delta: g · (M_t - M_baseline)
        baseline_score_SP = score_gen_vocab_SPT.gather(
            dim=-1, index=prepared.baseline_gen_SP.unsqueeze(-1)
        ).squeeze(-1)
        delta_vocab_SPT = score_gen_vocab_SPT - baseline_score_SP.unsqueeze(-1)

        S = grad_pred_SPK.size(0)
        delta_gen_SPT = grad_pred_SPK.new_zeros(
            (S, prepared.gen_length, gen_output_dim)
        )
        delta_gen_SPT[:, prepared.pred_pos_to_gen_pos_P, :gen_vocab_size] = (
            delta_vocab_SPT
        )
        return delta_gen_SPT


def _fill_masked_with_argmax(seq_SP, logp_gen_SPT, mask_token_id, vocab_size, n_samples=1):
    """Replace mask tokens with gen model's argmax predictions.
    If n_samples > 1, samples from the distribution instead.
    """
    if mask_token_id is None:
        return seq_SP.repeat(n_samples, 1)
    mask = seq_SP == mask_token_id
    if not mask.any():
        return seq_SP.repeat(n_samples, 1)
    
    filled = seq_SP.repeat(n_samples, 1)
    mask_expanded = mask.repeat(n_samples, 1)
    
    if n_samples == 1:
        gen_preds = logp_gen_SPT[..., :vocab_size].argmax(dim=-1)
        filled[mask] = gen_preds[mask]
    else:
        # logp_gen_SPT is [1, P, V] here
        probs = torch.softmax(logp_gen_SPT[0, ..., :vocab_size], dim=-1) # [P, V]
        gen_preds = torch.multinomial(probs, num_samples=n_samples, replacement=True).transpose(0, 1) # [n_samples, P]
        filled[mask_expanded] = gen_preds[mask_expanded]
        
    return filled


class TAG(GenerativeModel):
    """Token-level Autoregressive Guidance.

    Combines a generative model with a predictive model via Bayes' rule.
    Uses gradients through the predictive model's OHE to shift transition logits.

    Guidance projection is handled by a ``GuidanceProjection`` object, keeping
    TAG's core update rule focused on Bayes composition in gen-logit space.
    """

    def __init__(
        self,
        gen_model: GenerativeModel,
        pred_model: PredictiveModel,
        use_clean_classifier: bool = False,
        projection: Optional[GuidanceProjection] = None,
        n_fill_samples: int = 1,
    ):
        super().__init__(
            model=gen_model.model,
            tokenizer=gen_model.tokenizer,
            logit_formatter=gen_model.logit_formatter,
        )
        self.gen_model = gen_model
        self.pred_model = pred_model
        self.argmax_masked_positions = use_clean_classifier
        self.n_fill_samples = n_fill_samples
        if projection is None:
            projection = LinearGuidanceProjection(
                tokenizer_gen=gen_model.tokenizer,
                tokenizer_pred=pred_model.tokenizer,
                pred_token_ohe_basis_TK=pred_model.token_ohe_basis().detach(),
            )
        self.projection = projection.to(self.gen_model.device)

    def forward(self, seq_SP: torch.LongTensor):
        if self.gen_model.device != self.pred_model.device:
            raise ValueError(
                "TAG requires gen_model and pred_model to be on the same device"
            )
        logp_xtilde_g_x_SPT = self.gen_model.get_log_probs(seq_SP)
        prepared = self.projection.prepare(
            seq_SP,
            logp_xtilde_g_x_SPT,
            use_clean_classifier=self.argmax_masked_positions,
            n_samples=self.n_fill_samples,
        )
        # Compute gradient at temp=1 for natural gradient shape (no sigmoid
        # saturation), then use the predictor's temperature purely as a linear
        # guidance strength multiplier on the Taylor delta.
        guidance_temp = self.pred_model.temp
        self.pred_model.set_temp_(1.0)
        grad_pred_SPK = self.pred_model.grad_log_prob(prepared.seq_pred_SP)
        self.pred_model.set_temp_(guidance_temp)
        delta_gen_SPT = self.projection.grad_to_gen_delta(
            grad_pred_SPK,
            prepared,
            gen_output_dim=logp_xtilde_g_x_SPT.shape[-1],
        )
        return logp_xtilde_g_x_SPT + delta_gen_SPT / guidance_temp


class DEG(GenerativeModel):
    # TAG and DEG are basically ways to efficiently compute the vector p(y|x_const)
    # Therefore, we don't make the predictive models deal with that kind of query
    def __init__(
        self,
        gen_model: GenerativeModel,
        pred_model: PredictiveModel,
        argmax_masked_positions: bool = False,
        n_fill_samples: int = 1,
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
        self.n_fill_samples = n_fill_samples
        self.positions_to_score_S = None

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
                base_M = _fill_masked_with_argmax(
                    base.unsqueeze(0),
                    logp_xtilde_g_x_SPT[s].unsqueeze(0),
                    mask_token_id,
                    n_tok,
                    n_samples=self.n_fill_samples,
                ) # [n_samples, P]
            else:
                base_M = base.unsqueeze(0)
            
            n_samples = base_M.shape[0]
            # Create [n_samples, n_tok, P] by expanding base_M
            seq_MXP = base_M.unsqueeze(1).repeat(1, n_tok, 1)
            # Set the current position p to all possible tokens
            seq_MXP[:, :, p] = torch.arange(n_tok, device=seq_SP.device).unsqueeze(0).expand(n_samples, -1)
            # Flatten to [n_samples * n_tok, P]
            seq_flat = seq_MXP.view(-1, seq_SP.size(1))
            
            with torch.no_grad():
                logp_y_g_xtilde_flat = self.pred_model.get_log_probs(seq_flat)
            # Reshape back to [n_samples, n_tok]
            logp_y_g_xtilde_MX = logp_y_g_xtilde_flat.view(n_samples, n_tok)
            # Log-mean-exp over the samples: log( (1/M) sum_m exp(logp_m) )
            logp_y_g_xtilde_X = torch.logsumexp(logp_y_g_xtilde_MX, dim=0) - torch.log(torch.tensor(n_samples, dtype=torch.float32, device=seq_SP.device))
            
            logp_y_g_xtilde_SPT[s, p, :n_tok] = logp_y_g_xtilde_X
            # Don't need to take care of making the others -inf since the logit_formatter will take care of the invalid ones (including the invalid ones we tested lol)
        return logp_y_g_xtilde_SPT + logp_xtilde_g_x_SPT
