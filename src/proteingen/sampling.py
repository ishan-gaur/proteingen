import torch
from torch.nn import functional as F
from typing import Callable, Optional, List
from proteingen.generative_modeling import TransitionModel, TransitionFunc
from tqdm import tqdm

SamplingStep = Callable[
    [TransitionFunc, torch.LongTensor, ...],
    [torch.FloatTensor, Optional[List[float]]],
]  # takes in transition kernel and current state and returns updated state. List of floats is the list of timepoints.

Integrator = Callable[
    [TransitionModel, List[str]], List[str]
]  # Takes in transition model, tokenizes the input and then runs sampling to completion; this will need other args for e.g. uniform sampling


def tensor_to_string(x_SP, tokenizer):
    seq_SP = tokenizer.batch_decode(x_SP)
    seq_SP = [s.replace(" ", "") for s in seq_SP]
    seq_SP = [s.replace("<mask>", "") for s in seq_SP]
    seq_SP = [s.replace("<cls>", "") for s in seq_SP]
    seq_SP = [s.replace("<eos>", "") for s in seq_SP]
    return seq_SP


def sample_linear_interpolation(
    model: TransitionModel,
    x_SP: torch.LongTensor | List[str],
    n_steps: int,
    return_string=True,
):
    if isinstance(x_SP, list):
        x_SP = model.tokenizer(x_SP, padding=True, return_tensors="pt")["input_ids"]
    x_device = x_SP.device
    x_SP = x_SP.to(model.device)

    for step in tqdm(range(n_steps)):
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
    assert (x_SP == model.tokenizer.mask_token_id).sum() == 0
    if return_string:
        return tensor_to_string(x_SP, model.tokenizer)
    else:
        return x_SP.to(x_device)


def _formatted_logits(model: TransitionModel, x_SP: torch.LongTensor) -> torch.FloatTensor:
    """Run forward + format_raw_to_logits without applying log_softmax."""
    if model.observations is not None:
        obs = model.collate_observations(x_SP, model.observations)
        raw_output = model.forward(x_SP, **obs)
        return model.format_raw_to_logits(raw_output, x_SP, **obs)
    raw_output = model.forward(x_SP)
    return model.format_raw_to_logits(raw_output, x_SP)


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


def build_legacy_predictor_log_prob(tag_model):
    """Build old-demo style predictor_log_prob closure from DFM TAG components.

    Returns a closure compatible with the original flow-matching guidance loop:
    - integer token input: ``(B, P)`` in generator token space
    - one-hot input: ``(B, P, T_gen)`` for TAG Taylor guidance
    """
    from dfm.guide import LinearGuidanceProjection

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
                log_prob_xt_jumps[:, d_idx, s_idx] = predictor_log_prob(xt_jump, t_tensor)
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


@torch.no_grad()
def sample_flow_matching_legacy(
    model: TransitionModel,
    x_SP: torch.LongTensor | List[str],
    dt: float = 0.01,
    predictor_log_prob=None,
    guide_temp: float = 1.0,
    use_tag: bool = False,
    x1_temp: float = 1.0,
    stochasticity: float = 0.0,
    argmax_final: bool = True,
    logits_postprocess: Optional[Callable[[torch.FloatTensor, torch.LongTensor], torch.FloatTensor]] = None,
    return_string: bool = True,
) -> torch.LongTensor | list[str]:
    """Legacy flow-matching Euler sampler (old stability demo numerics).

    This reproduces the original rate-matrix integration loop and guidance-ratio
    update so DFM models can be compared head-to-head against old behavior.
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

    for _ in tqdm(range(n_steps)):
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

        t += dt
        if t > 1.0:
            break

    if argmax_final:
        xt_is_mask = (xt == mask_idx).view(*xt.shape).float()
        logits = _formatted_logits(model, xt)[..., :S]
        if logits_postprocess is not None:
            logits = logits_postprocess(logits, xt)
        xt = (torch.argmax(logits, dim=-1) * xt_is_mask + xt * (1 - xt_is_mask)).long()

    if return_string:
        return tensor_to_string(xt, model.tokenizer)
    return xt.to(x_device)


def sample_any_order_ancestral(
    model: TransitionModel,
    x_SP: torch.LongTensor | List[str],
    n_parallel: int = 1,
    return_string: bool = True,
):
    mask_token_id = model.tokenizer.mask_token_id
    pad_token_id = model.tokenizer.pad_token_id

    # TODO[pi] generally we won't do this kind of coddling in our code, but this is the
    # main high level interface most users will interact with, so we're making an exception
    if isinstance(x_SP, list):
        x_SP = model.tokenizer(x_SP, padding=True, return_tensors="pt")["input_ids"]
    x_device = x_SP.device
    x_SP = x_SP.to(model.device)

    t_st, t_end = (
        0.0,
        1.0,
    )  # TODO[pi] t_st should be the min of the fraction positions masked
    pbar = tqdm(total=t_end)

    t = t_st
    while t != t_end:
        x_SP = any_order_ancestral_step(
            model, x_SP, n_parallel, mask_token_id
        )
        len_S = x_SP.size(1) - (x_SP == pad_token_id).sum(dim=1)
        t_S = 1 - (x_SP == mask_token_id).sum(dim=1) / len_S
        t_new = t_S.min().item()
        pbar.update(t_new - t)
        # TODO[pi]: print out the matrix with the pbar updates so you can see it changing in place (this shouldn't print the matrix again an again causing vertical scrolling)
        t = t_new
    pbar.close()
    assert (x_SP == model.tokenizer.mask_token_id).sum() == 0
    if return_string:
        return tensor_to_string(x_SP, model.tokenizer)
    else:
        return x_SP.to(x_device)


def any_order_ancestral_step(
    model: TransitionModel,
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
