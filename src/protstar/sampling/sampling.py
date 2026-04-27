import math
import shutil
import sys

import torch
from torch.nn import functional as F
from typing import Callable, Optional, List, TypedDict, TextIO
from ..modeling.generative_modeling import GenerativeModel, TransitionFunc
from tqdm import tqdm


class SamplingTrajectory(TypedDict):
    """Per-step data recorded during ancestral sampling.

    sequences: (S,) list of generated sequences (strings).
    step_log_probs: (S, n_total) — log p(sampled_token) at each sampling step.
        Padding entries (from order padding with position 0) will have the
        log-prob of re-sampling the existing BOS token (0.0 if formatted correctly).
    step_positions: (S, n_total) — which position was sampled at each step.
        Padding entries are 0 (the BOS position).
    step_tokens: (S, n_total) — which token was sampled at each step.
    step_p_y_gt_t: (S, n_total) — optional tracking of property probabilities.
    """

    sequences: list[str]
    step_log_probs: torch.Tensor
    step_positions: torch.Tensor
    step_tokens: torch.Tensor
    step_p_y_gt_t: torch.Tensor | None


SamplingStep = Callable[
    [TransitionFunc, torch.LongTensor, ...],
    [torch.FloatTensor, Optional[List[float]]],
]  # takes in transition kernel and current state and returns updated state. List of floats is the list of timepoints.

Integrator = Callable[
    [GenerativeModel, List[str]], List[str]
]  # Takes in generative model, tokenizes the input and then runs sampling to completion; this will need other args for e.g. uniform sampling


def tensor_to_string(x_SP, tokenizer):
    if hasattr(tokenizer, "batch_decode"):
        seq_SP = tokenizer.batch_decode(x_SP)
        seq_SP = [s.replace(" ", "") for s in seq_SP]
    else:
        # Fallback for tokenizers without batch_decode (e.g. DPLM2Tokenizer)
        seq_SP = [tokenizer.decode(row) for row in x_SP]
    seq_SP = [s.replace("<mask>", "") for s in seq_SP]
    seq_SP = [s.replace("<cls>", "") for s in seq_SP]
    seq_SP = [s.replace("<eos>", "") for s in seq_SP]
    return seq_SP


def _tensor_to_preview_strings(x_SP: torch.LongTensor, tokenizer) -> list[str]:
    if hasattr(tokenizer, "batch_decode"):
        seq_SP = tokenizer.batch_decode(x_SP)
        seq_SP = [s.replace(" ", "") for s in seq_SP]
    else:
        seq_SP = [tokenizer.decode(row) for row in x_SP]
    seq_SP = [s.replace("<cls>", "") for s in seq_SP]
    seq_SP = [s.replace("<eos>", "") for s in seq_SP]
    seq_SP = [s.replace("<pad>", "") for s in seq_SP]
    seq_SP = [s.replace("<mask>", "_") for s in seq_SP]
    return seq_SP


def _truncate_for_terminal(text: str, max_width: int) -> str:
    if max_width <= 0:
        return ""
    if len(text) <= max_width:
        return text
    if max_width <= 3:
        return "." * max_width
    return text[: max_width - 3] + "..."


def _build_live_preview_lines(
    x_SP: torch.LongTensor, tokenizer, max_lines: int, max_width: int
) -> list[str]:
    if max_lines <= 0:
        return []

    n_sequences = x_SP.size(0)
    if n_sequences > max_lines:
        n_decode = max(max_lines - 1, 0)
    else:
        n_decode = n_sequences

    preview = _tensor_to_preview_strings(x_SP[:n_decode], tokenizer)
    lines = [_truncate_for_terminal(seq, max_width) for seq in preview]

    if n_sequences > max_lines:
        lines.append(_truncate_for_terminal("...", max_width))

    while len(lines) < max_lines:
        lines.append("")

    return lines


def _render_live_preview(
    lines: list[str], previous_line_count: int, stream: TextIO
) -> int:
    line_count = max(previous_line_count, len(lines))
    for i in range(line_count):
        stream.write("\n")
        stream.write("\x1b[2K")
        if i < len(lines):
            stream.write(lines[i])

    stream.flush()
    return len(lines)


class LiveSamplingPreview:
    """Terminal renderer for in-place sequence previews below tqdm.

    Interface:
    - ``update(x_SP)``: render current sampled sequences in-place
    - ``close()``: flush output (no-op cleanup)

    Intended usage inside sampling loops:

    ```python
    preview = LiveSamplingPreview(model.tokenizer, enabled=True)
    for step in ...:
        ...  # update x_SP
        pbar.update(1)
        preview.update(x_SP)
    preview.close()
    ```
    """

    def __init__(
        self,
        tokenizer,
        *,
        enabled: bool,
        reserve_lines: int = 2,
        stream: Optional[TextIO] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self._reserve_lines = reserve_lines
        self._stream = sys.stdout if stream is None else stream
        self._enabled = bool(
            enabled
            and hasattr(self._stream, "isatty")
            and self._stream.isatty()
        )
        self._line_count = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def update(self, x_SP: torch.LongTensor) -> None:
        if not self._enabled:
            return

        if self._line_count > 0:
            self._stream.write(f"\x1b[{self._line_count}A")
            self._stream.flush()

        terminal_size = shutil.get_terminal_size(fallback=(80, 24))
        max_lines = max(terminal_size.lines - self._reserve_lines, 1)
        max_width = max(terminal_size.columns, 1)
        preview_lines = _build_live_preview_lines(
            x_SP=x_SP,
            tokenizer=self.tokenizer,
            max_lines=max_lines,
            max_width=max_width,
        )
        self._line_count = _render_live_preview(
            preview_lines,
            previous_line_count=self._line_count,
            stream=self._stream,
        )

    def close(self) -> None:
        if self._enabled:
            self._stream.flush()

    def __enter__(self) -> "LiveSamplingPreview":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def sample_ctmc_linear_interpolation(
    model: GenerativeModel,
    x_SP: torch.LongTensor | List[str],
    n_steps: int,
    return_string=True,
    live_preview: bool = True,
):
    if isinstance(x_SP, list):
        x_SP = model.tokenizer(x_SP, padding=True, return_tensors="pt")["input_ids"]
    x_device = x_SP.device
    x_SP = x_SP.to(model.device)

    preview = LiveSamplingPreview(model.tokenizer, enabled=live_preview)

    with preview, tqdm(total=n_steps) as pbar:
        for step in range(n_steps):
            logp_x_SPT = model.get_log_probs(
                x_SP
            )  # TODO[pi] I'm not seeing any -infs here--which I should with ESM3 for a masked logit formatted output
            # Euler in this case is just a linear interpolation
            output_dim = logp_x_SPT.size(2)
            X_0_SPT = F.one_hot(x_SP, num_classes=output_dim)
            X_n_SPT = torch.exp(logp_x_SPT)
            steps_left = n_steps - step
            X_1_SPT = ((steps_left - 1) / steps_left) * X_0_SPT + (1 / steps_left) * X_n_SPT
            X_1_SxPT = X_1_SPT.reshape(-1, output_dim)
            x_SP = torch.multinomial(X_1_SxPT, num_samples=1).reshape(*(x_SP.shape))

            pbar.update(1)
            preview.update(x_SP)
    assert (x_SP == model.tokenizer.mask_token_id).sum() == 0
    if return_string:
        return tensor_to_string(x_SP, model.tokenizer)
    else:
        return x_SP.to(x_device)


@torch.no_grad()
def sample(
    model: GenerativeModel,
    x_SP: torch.LongTensor | List[str],
    n_parallel: int = 1,
    in_order: Optional[list[torch.LongTensor] | str] = None,
    live_preview: bool = True,
    record_p_y_gt_t: bool = False,
) -> SamplingTrajectory:
    """Ancestral sampling for masked generative models.

    Unmasks positions n_parallel at a time, sampling from the model's predicted
    distribution at each step. If ``in_order`` is not provided, a random
    permutation of masked positions is generated for each sequence.

    **Sharp edge**: unmask orders are padded to uniform length across sequences
    with position 0 (typically BOS/CLS). At padding steps, ALL sequences are
    still sampled at their designated positions — including the padding position.
    If the model's logit formatter is correctly configured, special-token
    positions predict only themselves, making the padding a no-op. If logits
    are NOT properly formatted, the token at position 0 may be mutated.

    Args:
        model: a GenerativeModel (or guided TAG/DEG model).
        x_SP: (S, P) partially masked token IDs, or list of strings.
        n_parallel: number of positions to unmask per step. Default 1.
        in_order: controls the unmask order. Can be:
            - None (default): random permutation of masked positions per sequence.
            - ``"left_to_right"``: masked positions in ascending index order.
            - ``list[LongTensor]``: one tensor per sequence giving explicit
              positions to unmask (first element = first revealed).
        live_preview: if True, render in-place sequence updates below tqdm in a
            real terminal. Automatically disabled when stdout is not a TTY.

    Returns:
        SamplingTrajectory with generated sequences and per-step data.
    """
    mask_token_id = model.tokenizer.mask_token_id

    if isinstance(x_SP, list):
        x_SP = model.tokenizer(x_SP, padding=True, return_tensors="pt")["input_ids"]
    x_SP = x_SP.clone().to(model.device)
    S, P = x_SP.shape

    # Build unmask orders
    if in_order is None:
        in_order = []
        for s in range(S):
            masked = (x_SP[s] == mask_token_id).nonzero().flatten()
            in_order.append(masked[torch.randperm(len(masked))])
    elif in_order == "left_to_right":
        in_order = []
        for s in range(S):
            in_order.append((x_SP[s] == mask_token_id).nonzero().flatten())

    assert len(in_order) == S, f"Expected {S} orders, got {len(in_order)}"

    # Pad orders to uniform length with position 0, then chunk by n_parallel
    max_positions = max((len(o) for o in in_order), default=0)
    n_steps = math.ceil(max_positions / n_parallel) if max_positions > 0 else 0
    padded_len = n_steps * n_parallel

    # order_flat: (S, padded_len) — padded with position 0
    order_flat = torch.zeros(S, padded_len, dtype=torch.long)
    for s, order in enumerate(in_order):
        order_flat[s, : len(order)] = order

    # (S, n_steps, n_parallel) — positions to unmask at each step
    if n_steps > 0:
        order_steps = order_flat.reshape(S, n_steps, n_parallel).to(model.device)
    else:
        order_steps = order_flat.reshape(S, 0, n_parallel).to(model.device)

    # Trajectory storage — flat (S, padded_len)
    step_log_probs = torch.full((S, padded_len), float("nan"))
    step_positions = order_flat.clone()
    step_tokens = torch.full((S, padded_len), -1, dtype=torch.long)
    step_p_y_gt_t = torch.full((S, padded_len), float("nan")) if record_p_y_gt_t else None

    preview = LiveSamplingPreview(model.tokenizer, enabled=live_preview)

    with preview, tqdm(total=n_steps) as pbar:
        for step in range(n_steps):
            positions = order_steps[:, step, :]  # (S, n_parallel)

            # DEG needs position info before computing log probs
            if hasattr(model, "at_position"):
                if n_parallel > 1:
                    raise NotImplementedError("DEG with n_parallel > 1 not implemented")
                positions_per_seq: List[Optional[int]] = [
                    pos[0].item() for pos in positions
                ]
                with model.at_position(positions_per_seq):
                    log_probs_SPT = model.get_log_probs(x_SP)
            else:
                log_probs_SPT = model.get_log_probs(x_SP)

            T = log_probs_SPT.size(-1)
            probs_SPT = torch.exp(log_probs_SPT)

            # Gather probs at selected positions
            pos_expanded = positions.unsqueeze(-1).expand(S, n_parallel, T)  # (S, n_parallel, T)
            probs_at_pos = probs_SPT.gather(1, pos_expanded)  # (S, n_parallel, T)

            # Sample tokens
            tokens = torch.multinomial(
                probs_at_pos.reshape(S * n_parallel, T), num_samples=1
            ).reshape(S, n_parallel)  # (S, n_parallel)

            # Update sequences
            x_SP.scatter_(1, positions, tokens)

            # Record trajectory
            flat_start = step * n_parallel
            flat_end = flat_start + n_parallel
            log_probs_at_pos = log_probs_SPT.gather(1, pos_expanded)  # (S, n_parallel, T)
            token_log_probs = log_probs_at_pos.gather(
                2, tokens.unsqueeze(-1)
            ).squeeze(-1)  # (S, n_parallel)
            step_log_probs[:, flat_start:flat_end] = token_log_probs.cpu()
            step_tokens[:, flat_start:flat_end] = tokens.cpu()

            if record_p_y_gt_t and hasattr(model, "pred_model"):
                pred_log_probs = model.pred_model.get_log_probs(x_SP)
                p_y_gt_t = torch.exp(pred_log_probs).cpu()
                if step_p_y_gt_t is not None:
                    step_p_y_gt_t[:, flat_start:flat_end] = p_y_gt_t.unsqueeze(-1).expand(S, n_parallel)

            pbar.update(1)

            preview.update(x_SP)

    assert (x_SP == mask_token_id).sum() == 0, "Some positions remain masked"
    sequences = tensor_to_string(x_SP, model.tokenizer)

    return SamplingTrajectory(
        sequences=sequences,
        step_log_probs=step_log_probs,
        step_positions=step_positions,
        step_tokens=step_tokens,
        step_p_y_gt_t=step_p_y_gt_t,
    )


def any_order_ancestral_step(
    model: GenerativeModel,
    x_SP: torch.LongTensor,
    n_parallel: int,
    mask_token_id: int,
    next_pos_idx_SP: Optional[torch.LongTensor] = None,
) -> torch.FloatTensor:
    # TODO[pi] actually drop sequences that are finished and then place them in their original position at the end
    # so can we actually reduce the effective batch size if we can

    # Pick positions to sample BEFORE calling the model — DEG needs to know
    # which position to enumerate before computing log probs.
    if next_pos_idx_SP is None:
        next_pos_idx_SP = []  # idx tensor for the sequence and pos dimensions, doesn't actually have full SxP shape
        for s in range(x_SP.size(0)):
            masked_positions_S = (x_SP[s] == mask_token_id).nonzero().flatten()
            rand_idxs = torch.randperm(len(masked_positions_S))[:n_parallel]
            for p in masked_positions_S[rand_idxs]:
                next_pos_idx_SP.append([s, p.item()])
        next_pos_idx_SP = torch.LongTensor(next_pos_idx_SP)

    if next_pos_idx_SP.numel() == 0:
        return x_SP

    # Get log probs — if model is DEG, pass position info via at_position
    if hasattr(model, "at_position"):
        B = x_SP.size(0)
        positions_per_seq: List[Optional[int]] = [None] * B
        if n_parallel > 1:
            raise NotImplementedError("DEG with n_parallel>1 not implemented yet")
        for s_idx, p_idx in next_pos_idx_SP:
            # With n_parallel=1 this is one position per sequence.
            # With n_parallel>1, last position wins — DEG with n_parallel>1
            # would need extension to enumerate multiple positions per sequence.
            positions_per_seq[s_idx.item()] = p_idx.item()
        with model.at_position(positions_per_seq):
            p_x_SPT = torch.exp(model.get_log_probs(x_SP))
    else:
        p_x_SPT = torch.exp(model.get_log_probs(x_SP))

    # Actually sample from the selected positions
    p_x_ST = torch.stack(
        [p_x_SPT[i, j, :] for i, j in next_pos_idx_SP]
    )  # note that some sequences (S dimension) might be missing
    x_new_S = torch.multinomial(p_x_ST, num_samples=1)
    for i_for_pos_idx, (i, j) in enumerate(next_pos_idx_SP):
        x_SP[i, j] = x_new_S[i_for_pos_idx]
    return x_SP


def generate_unmask_orders(
    seq_lengths: list[int],
    n_orders: int,
    special_positions: Optional[list[set[int]]] = None,
    seed: Optional[int] = None,
) -> list[list[torch.LongTensor]]:
    """Generate random unmask orders for a set of sequences.

    Returns orders[s][k] = 1-D LongTensor of maskable position indices for
    sequence s, order k. Positions are listed in the order they should be
    unmasked (first element = first position revealed).

    Args:
        seq_lengths: length of each tokenized sequence (including special tokens).
        n_orders: how many random orders to generate per sequence.
        special_positions: per-sequence set of position indices that are NOT
            maskable (BOS, EOS, PAD). If None, positions 0 and L-1 are treated
            as special (BOS/EOS convention for ESM-family tokenizers).
        seed: optional RNG seed for reproducibility.
    """
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    orders: list[list[torch.LongTensor]] = []
    for s, L in enumerate(seq_lengths):
        if special_positions is not None:
            maskable = sorted(set(range(L)) - special_positions[s])
        else:
            # Default: skip first and last (BOS/EOS)
            maskable = list(range(1, L - 1))
        maskable_t = torch.LongTensor(maskable)
        seq_orders = []
        for _ in range(n_orders):
            perm = torch.randperm(len(maskable_t), generator=rng)
            seq_orders.append(maskable_t[perm])
        orders.append(seq_orders)
    return orders


def mask_by_order(
    token_ids: torch.LongTensor,
    order: torch.LongTensor,
    mask_fraction: float,
    mask_token_id: int,
) -> torch.LongTensor:
    """Mask positions according to a decoding order.

    Positions appearing LAST in the order (the tail) are masked. Specifically,
    the last ``ceil(mask_fraction * len(order))`` positions in the order are
    set to mask_token_id.

    Args:
        token_ids: (P,) token IDs for a single sequence.
        order: (M,) position indices in unmask order (first = first revealed).
        mask_fraction: fraction of maskable positions to mask (0 to 1).
        mask_token_id: token ID to use for masking.

    Returns:
        (P,) masked token IDs. The order of unmasking during generation should
        be order[n_keep:] (i.e. the masked positions, in unmask order).
    """
    n_maskable = len(order)
    n_to_mask = math.ceil(mask_fraction * n_maskable)
    n_keep = n_maskable - n_to_mask

    masked = token_ids.clone()
    positions_to_mask = order[n_keep:]
    masked[positions_to_mask] = mask_token_id
    return masked


# For legacy code comparison
def _formatted_logits(
    model: GenerativeModel, x_SP: torch.LongTensor
) -> torch.FloatTensor:
    """Run forward + format_raw_to_logits without applying log_softmax."""
    if model.observations is not None:
        obs = model.collate_observations(x_SP, model.observations)
        raw_output = model.forward(x_SP, **obs)
        return model.format_raw_to_logits(raw_output, x_SP, **obs)
    raw_output = model.forward(x_SP)
    return model.format_raw_to_logits(raw_output, x_SP)


# For legacy code comparison
def _predictive_log_prob_from_ohe(
    pred_model,
    ohe_seq_SPK: torch.FloatTensor,
) -> torch.FloatTensor:
    """Evaluate predictor log p(target|x) from already-projected OHE features."""
    if pred_model.target is None:
        raise ValueError(
            "Predictive model target not set. Call set_target_() or use with_target()."
        )
    if pred_model.observations is not None:
        obs = pred_model.collate_observations(ohe_seq_SPK, pred_model.observations)
        raw_output = pred_model.forward(ohe_seq_SPK, **obs)
        logits = pred_model.format_raw_to_logits(raw_output, ohe_seq_SPK, **obs)
    else:
        raw_output = pred_model.forward(ohe_seq_SPK)
        logits = pred_model.format_raw_to_logits(raw_output, ohe_seq_SPK)
    log_probs = F.log_softmax(logits / pred_model.temp, dim=-1)
    return log_probs[:, 1]


# For legacy code comparison
def build_legacy_predictor_log_prob(tag_model):
    """Build old-demo style predictor_log_prob closure from DFM TAG components.

    Returns a closure compatible with the original flow-matching guidance loop:
    - integer token input: ``(B, P)`` in generator token space
    - one-hot input: ``(B, P, T_gen)`` for TAG Taylor guidance
    """
    from ..modeling.guide import LinearGuidanceProjection

    projection = tag_model.projection
    if not isinstance(projection, LinearGuidanceProjection):
        raise ValueError(
            "Legacy predictor_log_prob currently supports LinearGuidanceProjection only"
        )
    pred_model = tag_model.pred_model
    gen_vocab_size = projection.tokenizer_gen.vocab_size
    start = projection._strip_prefix
    suffix = projection._strip_suffix

    def _pred_window(x):
        end = x.shape[1] - suffix
        if end < start:
            raise ValueError(
                f"Invalid strip window: start={start}, end={end}, length={x.shape[1]}"
            )
        return x[:, start:end]

    def predictor_log_prob(xt, t, **kwargs):
        if xt.is_floating_point():
            xt_inner = _pred_window(xt)
            xt_inner = xt_inner[..., :gen_vocab_size]
            pred_ohe = xt_inner @ projection.gen_to_pred_ohe_TK.to(xt.device)
            return _predictive_log_prob_from_ohe(pred_model, pred_ohe)

        xt = xt.long()
        xt_inner = _pred_window(xt)
        pred_tokens = projection.gen_to_pred_idx_T[xt_inner]
        return pred_model.get_log_probs(pred_tokens)

    return predictor_log_prob


# For legacy code comparison
def _legacy_get_guided_rates(
    predictor_log_prob,
    xt,
    t,
    R_t,
    S,
    use_tag=False,
    guide_temp=1.0,
    log_prob_ratio_cutoff=80.0,
):
    """Original demo guidance-ratio update used by flow-matching sampler."""
    B, D = xt.shape
    device = xt.device
    t_tensor = t * torch.ones((B,), device=device)

    if not use_tag:
        log_prob_xt = predictor_log_prob(xt, t_tensor)
        log_prob_xt_jumps = torch.zeros(B, D, S, device=device)
        for d_idx in range(D):
            for s_idx in range(S):
                xt_jump = xt.clone()
                xt_jump[:, d_idx] = s_idx
                log_prob_xt_jumps[:, d_idx, s_idx] = predictor_log_prob(
                    xt_jump, t_tensor
                )
        log_prob_ratio = log_prob_xt_jumps - log_prob_xt.view(B, 1, 1)
    else:
        xt_ohe = F.one_hot(xt.long(), num_classes=S).to(torch.float)
        with torch.enable_grad():
            xt_ohe.requires_grad_(True)
            log_prob_xt = predictor_log_prob(xt_ohe, t_tensor)
            log_prob_xt.sum().backward()
            grad_log_prob = xt_ohe.grad
        log_prob_ratio = grad_log_prob - (xt_ohe * grad_log_prob).sum(
            dim=-1, keepdim=True
        )

    log_prob_ratio /= guide_temp
    log_prob_ratio = torch.clamp(log_prob_ratio, max=log_prob_ratio_cutoff)
    prob_ratio = torch.exp(log_prob_ratio)
    return R_t * prob_ratio


# For legacy code comparison
@torch.no_grad()
def sample_flow_matching_legacy(
    model: GenerativeModel,
    x_SP: torch.LongTensor | List[str],
    dt: float = 0.01,
    predictor_log_prob=None,
    guide_temp: float = 1.0,
    use_tag: bool = False,
    x1_temp: float = 1.0,
    stochasticity: float = 0.0,
    argmax_final: bool = True,
    logits_postprocess: Optional[
        Callable[[torch.FloatTensor, torch.LongTensor], torch.FloatTensor]
    ] = None,
    return_string: bool = True,
    live_preview: bool = True,
) -> torch.LongTensor | list[str]:
    """Legacy flow-matching Euler sampler (old stability demo numerics).

    This reproduces the original rate-matrix integration loop and guidance-ratio
    update so DFM models can be compared head-to-head against old behavior.

    live_preview shows in-place sequence updates below tqdm in real terminals.
    """
    if isinstance(x_SP, list):
        x_SP = model.tokenizer(x_SP, padding=True, return_tensors="pt")["input_ids"]
    x_device = x_SP.device
    xt = x_SP.to(model.device)

    S = model.tokenizer.vocab_size
    mask_idx = model.tokenizer.mask_token_id
    if mask_idx is None:
        raise ValueError("sample_flow_matching_legacy requires tokenizer.mask_token_id")

    t = 0.0
    n_steps = int(1.0 / dt)
    mask_one_hot = torch.zeros((S,), device=xt.device)
    mask_one_hot[mask_idx] = 1.0

    preview = LiveSamplingPreview(model.tokenizer, enabled=live_preview)

    with preview, tqdm(total=n_steps) as pbar:
        for _ in range(n_steps):
            logits = _formatted_logits(model, xt)[..., :S]
            if logits_postprocess is not None:
                logits = logits_postprocess(logits, xt)
            pt_x1_probs = F.softmax(logits / x1_temp, dim=-1)

            xt_is_mask = (xt == mask_idx).view(*xt.shape, 1).float()
            R_t = xt_is_mask * pt_x1_probs * ((1 + stochasticity * t) / (1 - t))
            remask_rates = (1 - xt_is_mask) * mask_one_hot.view(1, 1, -1) * stochasticity
            R_t += remask_rates

            if predictor_log_prob is not None:
                R_t = _legacy_get_guided_rates(
                    predictor_log_prob,
                    xt,
                    t,
                    R_t,
                    S,
                    use_tag=use_tag,
                    guide_temp=guide_temp,
                )

            R_t.scatter_(-1, xt[:, :, None], 0.0)
            R_t.scatter_(-1, xt[:, :, None], (-R_t.sum(dim=-1, keepdim=True)))

            step_probs = (R_t * dt).clamp(min=0.0, max=1.0)
            step_probs.scatter_(-1, xt[:, :, None], 0.0)
            step_probs.scatter_(
                -1,
                xt[:, :, None],
                (1.0 - torch.sum(step_probs, dim=-1, keepdim=True)).clamp(min=0.0),
            )
            step_probs = torch.clamp(step_probs, min=0.0, max=1.0)

            xt = torch.distributions.Categorical(step_probs).sample()

            pbar.update(1)
            preview.update(xt)

            t += dt
            if t > 1.0:
                break

        if argmax_final:
            xt_is_mask = (xt == mask_idx).view(*xt.shape).float()
            logits = _formatted_logits(model, xt)[..., :S]
            if logits_postprocess is not None:
                logits = logits_postprocess(logits, xt)
            xt = (torch.argmax(logits, dim=-1) * xt_is_mask + xt * (1 - xt_is_mask)).long()
            preview.update(xt)

    if return_string:
        return tensor_to_string(xt, model.tokenizer)
    return xt.to(x_device)
