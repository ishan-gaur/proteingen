import torch
from torch.nn import functional as F
from typing import Callable, Optional, List
from dfm.generative_modeling import TransitionModel, TransitionFunc
from tqdm import tqdm
import random
import math

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


def sample_euler(
    model: TransitionModel,
    x_SP: torch.LongTensor | List[str],
    n_steps: int,
    return_string=True,
):
    if isinstance(x_SP, list):
        x_SP = model.tokenizer(x_SP, padding=True, return_tensors="pt")["input_ids"]
    x_device = x_SP.device
    x_SP = x_SP.to(model.device)

    t_st, t_end = 0.0, 1.0
    dt = (t_end - t_st) / n_steps
    for step in tqdm(range(n_steps)):
        logp_x_SPT = model.get_log_probs(
            x_SP
        )  # TODO[pi] I'm not seeing any -infs here--which I should with ESM3IF for a masked logit formatted output
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
            model.get_log_probs, x_SP, n_parallel, mask_token_id
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
    transition_log_prob_fn: TransitionFunc,  # is there any reason not to just pass the model here?
    x_SP: torch.LongTensor,
    n_parallel: int,
    mask_token_id: int,
    next_pos_idx_SP: Optional[torch.LongTensor] = None,
) -> torch.FloatTensor:
    # TODO[pi] actually drop sequences that are finished and then place them in their original position at the end
    # so can we actually reduce the effective batch size if we can

    # If the caller doesn't specify which positions to sample next, sample a random masked index (if possible)
    # for each sequence
    if next_pos_idx_SP is None:
        next_pos_idx_SP = []  # idx tensor for the sequence and pos dimensions, doesn't actually have full SxP shape
        for s in range(x_SP.size(0)):
            masked_positions_S = (x_SP[s] == mask_token_id).nonzero().flatten()
            rand_idxs = torch.randperm(len(masked_positions_S))[:n_parallel]
            for p in masked_positions_S[rand_idxs]
                p = masked_positions_S[rand_idx]
                next_pos_idx_SP.append([s, p])
        next_pos_idx_SP = torch.LongTensor(next_pos_idx_SP)

    if next_pos_idx_SP.numel() == 0:
        return x_SP

    # Actually get the log probs and sample the change
    p_x_SPT = torch.exp(transition_log_prob_fn(x_SP))
    p_x_ST = torch.stack(
        [p_x_SPT[i, j, :] for i, j in next_pos_idx_SP]
    )  # note that some sequences (S dimension) might be missing
    x_new_S = torch.multinomial(p_x_ST, num_samples=1)
    for i_for_pos_idx, (i, j) in enumerate(next_pos_idx_SP):
        x_SP[i, j] = x_new_S[i_for_pos_idx]
    return x_SP
