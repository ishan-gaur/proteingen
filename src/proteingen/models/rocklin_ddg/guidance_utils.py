"""
Guidance utilities for stability-guided protein inverse folding.

Contains:
- Predictor log prob (accepts ESM3 tokens, converts internally to PMPNN)
- Stability prediction and ddG evaluation
- Flow matching sampling with TAG guidance
- ESM3 inverse folding wrappers
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from tqdm import tqdm

from proteingen.models.rocklin_ddg.data_utils import (
    PMPNN_ALPHABET,
    ESM_ALPHABET,
    ESM3_ALPHABET_SIZE,
    esm_tokens_to_pmpnn_tokens_batch,
    esm_ohe_to_pmpnn_ohe,
    esm_tok_to_pmpnn_tok,
    format_coords_to_esm3,
)


# ---------------------------------------------------------------------------
# Predictor log prob (guidance signal)
# ---------------------------------------------------------------------------


def get_predictor_log_prob(classifier, cond_inputs):
    """Build a predictor_log_prob closure that accepts ESM3-alphabet tokens.

    Pre-encodes structure once (expensive), returns a fast closure that only
    runs the decoder at each sampling step.

    The returned function handles both:
    - Integer tokens (B, D): used in exact guidance
    - One-hot float tensors (B, D, S): used in TAG (Taylor-Approximated Guidance)

    Args:
        classifier: StabilityPMPNN model loaded with one_hot_encode_input=True
        cond_inputs: dict from prepare_conditioning_inputs (PMPNN alphabet, no cls/eos)

    Returns:
        predictor_log_prob(xt, t) -> log_probs (B,)
    """
    with torch.no_grad():
        h_V, h_E, E_idx, mask_attend = classifier.encode_structure(
            cond_inputs["X"],
            cond_inputs["mask"],
            cond_inputs["chain_M"],
            cond_inputs["residue_idx"],
            cond_inputs["chain_encoding_all"],
        )

    def predictor_log_prob(xt, t, **kwargs):
        B = xt.shape[0]

        # Replicate pre-encoded structure for batch
        h_V_rep = h_V[0:1].repeat(B, 1, 1)
        h_E_rep = h_E[0:1].repeat(B, 1, 1, 1)
        E_idx_rep = E_idx[0:1].repeat(B, 1, 1)
        mask_attend_rep = mask_attend[0:1].repeat(B, 1, 1)
        mask_rep = cond_inputs["mask"][0:1].repeat(B, 1)
        chain_M_rep = cond_inputs["chain_M"][0:1].repeat(B, 1)

        if xt.is_floating_point():
            # TAG mode: xt is one-hot (B, D, 33)
            # Convert ESM3 one-hot to PMPNN one-hot, preserving gradients
            xt_pmpnn = esm_ohe_to_pmpnn_ohe(xt)
        else:
            # Integer token mode: xt is (B, D) ESM3 indices
            # Strip cls/eos and convert to PMPNN indices
            xt_pmpnn_idx = esm_tokens_to_pmpnn_tokens_batch(xt)
            xt_pmpnn = F.one_hot(xt_pmpnn_idx, num_classes=len(PMPNN_ALPHABET)).float()

        logit = classifier.decode(
            h_V_rep, h_E_rep, E_idx_rep,
            xt_pmpnn, mask_attend_rep, mask_rep, chain_M_rep,
        )
        return torch.log(F.sigmoid(logit.reshape(-1)))

    return predictor_log_prob


# ---------------------------------------------------------------------------
# Stability prediction and evaluation
# ---------------------------------------------------------------------------


def predict_stability(seq_tokens, stability_model, cond_inputs, device="cuda"):
    """Predict stability (dG) for a batch of sequences.

    Args:
        seq_tokens: (N, L) tensor of PMPNN-alphabet tokens (no special tokens)
        stability_model: StabilityPMPNN oracle model
        cond_inputs: conditioning inputs dict

    Returns:
        numpy array of dG predictions (N,)
    """
    num_samples = seq_tokens.shape[0]
    batch_size = cond_inputs["X"].shape[0]
    preds = []

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            curr_batch = seq_tokens[i: i + batch_size].to(device)
            curr_batch_size = curr_batch.shape[0]

            if curr_batch_size < batch_size:
                batch_cond = {
                    key: val[:curr_batch_size] if isinstance(val, torch.Tensor) else val
                    for key, val in cond_inputs.items()
                }
            else:
                batch_cond = cond_inputs

            pred = (
                stability_model(
                    batch_cond["X"],
                    curr_batch,
                    batch_cond["mask"],
                    batch_cond["chain_M"],
                    batch_cond["residue_idx"],
                    batch_cond["chain_encoding_all"],
                )
                .reshape(-1)
                .detach()
                .cpu()
                .numpy()
            )
            preds.append(pred)

    return np.concatenate(preds)


def compute_ddg(sequences, oracle_model, cond_inputs, device="cuda"):
    """Compute predicted ddG for a list of amino acid sequences.

    Args:
        sequences: list of amino acid strings (PMPNN alphabet)
        oracle_model: StabilityPMPNN regression model (vocab=21)
        cond_inputs: conditioning inputs dict

    Returns:
        numpy array of ddG values (N,)
    """
    seq_tokens = torch.tensor(
        [[PMPNN_ALPHABET.index(aa) for aa in seq] for seq in sequences],
        dtype=torch.long,
    )

    preds = predict_stability(seq_tokens, oracle_model, cond_inputs, device)
    wt_pred = predict_stability(
        cond_inputs["S1"][0:1], oracle_model, cond_inputs, device
    )
    ddg = preds - wt_pred[0]
    return ddg


# ---------------------------------------------------------------------------
# Flow matching sampling (extracted from pdg/utils/fm_utils.py)
# ---------------------------------------------------------------------------


def _normalize_rates(R_t, xt):
    """Set diagonal of rate matrix to negative row sum."""
    R_t = torch.scatter(R_t, -1, xt[:, :, None], 0.0)
    R_t = torch.scatter(R_t, -1, xt[:, :, None], (-R_t.sum(dim=-1, keepdim=True)))
    return R_t


def _normalize_step_probs(step_probs, xt):
    """Normalize transition probabilities."""
    step_probs = torch.scatter(step_probs, -1, xt[:, :, None], 0.0)
    step_probs = torch.scatter(
        step_probs, -1, xt[:, :, None],
        (1.0 - torch.sum(step_probs, dim=-1, keepdim=True)).clamp(min=0.0),
    )
    step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
    return step_probs


def get_guided_rates(
    predictor_log_prob,
    xt, t, R_t, S,
    use_tag=False,
    guide_temp=1.0,
    log_prob_ratio_cutoff=80.0,
    **kwargs,
):
    """Compute guide-adjusted rates for predictor guidance.

    Supports TAG (Taylor-Approximated Guidance) via one-hot + gradient.

    Args:
        predictor_log_prob: function(xt, t) -> log p(y|x,t) of shape (B,)
        xt: current states (B, D)
        t: current time (float)
        R_t: unconditional rates (B, D, S)
        S: vocab size
        use_tag: use Taylor approximation
        guide_temp: guidance temperature (lower = stronger)
    """
    B, D = xt.shape
    device = xt.device
    t_tensor = t * torch.ones((B,), device=device)

    if not use_tag:
        # Exact guidance: evaluate predictor at all possible single-position transitions
        log_prob_xt = predictor_log_prob(xt, t_tensor)

        # Build all jump transitions
        xt_expanded = xt.unsqueeze(2).expand(B, D, S)  # (B, D, S)
        arange_S = torch.arange(S, device=device).view(1, 1, S).expand(B, D, S)
        # For each (b, d, s), replace position d with token s
        xt_jumps_flat = xt.unsqueeze(2).expand(B, D, S).clone()  # (B, D, S)
        # We need to iterate since we can't easily vectorize this
        log_prob_xt_jumps = torch.zeros(B, D, S, device=device)
        for d_idx in range(D):
            for s_idx in range(S):
                xt_jump = xt.clone()
                xt_jump[:, d_idx] = s_idx
                log_prob_xt_jumps[:, d_idx, s_idx] = predictor_log_prob(xt_jump, t_tensor)

        log_prob_ratio = log_prob_xt_jumps - log_prob_xt.view(B, 1, 1)
    else:
        # TAG: Taylor-approximated guidance
        xt_ohe = F.one_hot(xt.long(), num_classes=S).to(torch.float)
        with torch.enable_grad():
            xt_ohe.requires_grad_(True)
            log_prob_xt = predictor_log_prob(xt_ohe, t_tensor)
            log_prob_xt.sum().backward()
            grad_log_prob = xt_ohe.grad  # (B, D, S)

        # 1st order Taylor approximation of log ratio
        log_prob_ratio = grad_log_prob - (xt_ohe * grad_log_prob).sum(
            dim=-1, keepdim=True
        )

    # Scale by temperature and clamp
    log_prob_ratio /= guide_temp
    log_prob_ratio = torch.clamp(log_prob_ratio, max=log_prob_ratio_cutoff)

    # Multiply rates by probability ratio
    prob_ratio = torch.exp(log_prob_ratio)
    R_t = R_t * prob_ratio

    return R_t


@torch.no_grad()
def flow_matching_sampling_masking_euler(
    denoising_model,
    batch_size,
    S, D, device,
    dt=0.01,
    mask_idx=None,
    predictor_log_prob=None,
    guide_temp=1.0,
    stochasticity=0.0,
    use_tag=False,
    argmax_final=True,
    x1_temp=1.0,
    eps=1e-9,
    **kwargs,
):
    """Single-particle Euler integration for discrete flow matching.

    Args:
        denoising_model: function(xt, t) -> logits (B, D, S)
        batch_size: number of samples
        S: vocabulary size
        D: sequence length (including cls/eos for ESM3)
        device: torch device
        dt: time step
        mask_idx: mask token index
        predictor_log_prob: optional guidance function
        guide_temp: guidance temperature
        stochasticity: remasking rate
        use_tag: use Taylor approximation for guidance
        argmax_final: argmax remaining masks at end
        x1_temp: temperature for model logits
    """
    if mask_idx is None:
        mask_idx = S - 1

    B = batch_size
    xt = mask_idx * torch.ones((B, D), dtype=torch.long, device=device)

    t = 0.0
    num_steps = int(1 / dt)
    mask_one_hot = torch.zeros((S,), device=device)
    mask_one_hot[mask_idx] = 1.0

    for _ in tqdm(range(num_steps)):
        logits = denoising_model(xt, t * torch.ones((B,), device=device))
        pt_x1_probs = F.softmax(logits / x1_temp, dim=-1)

        # Unmasking rates
        xt_is_mask = (xt == mask_idx).view(B, D, 1).float()
        R_t = xt_is_mask * pt_x1_probs * ((1 + stochasticity * t) / (1 - t))

        # Remasking rates
        remask_rates = (1 - xt_is_mask) * mask_one_hot.view(1, 1, -1) * stochasticity
        R_t += remask_rates

        # Apply predictor guidance
        if predictor_log_prob is not None:
            R_t = get_guided_rates(
                predictor_log_prob, xt, t, R_t, S,
                use_tag=use_tag, guide_temp=guide_temp,
            )

        # Normalize rates and convert to probabilities
        R_t.scatter_(-1, xt[:, :, None], 0.0)
        R_t.scatter_(-1, xt[:, :, None], (-R_t.sum(dim=-1, keepdim=True)))

        step_probs = (R_t * dt).clamp(min=0.0, max=1.0)
        step_probs.scatter_(-1, xt[:, :, None], 0.0)
        step_probs.scatter_(
            -1, xt[:, :, None],
            (1.0 - torch.sum(step_probs, dim=-1, keepdim=True)).clamp(min=0.0),
        )
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)

        # Sample next state
        xt = torch.distributions.Categorical(step_probs).sample()

        t += dt
        if t > 1.0:
            break

    # Argmax remaining masks
    if argmax_final:
        xt_is_mask = (xt == mask_idx).view(B, D).float()
        logits = denoising_model(xt, t * torch.ones((B,), device=device))
        xt = (torch.argmax(logits, dim=-1) * xt_is_mask + xt * (1 - xt_is_mask)).long()

    return xt.detach().cpu().numpy()


def flow_matching_sampling(
    num_samples, denoising_model,
    S, D, device,
    dt=0.01, mask_idx=None,
    predictor_log_prob=None,
    guide_temp=1.0,
    stochasticity=0.0,
    use_tag=False,
    batch_size=100,
    x1_temp=1.0,
    **kwargs,
):
    """Generate samples using flow matching in batches.

    Returns:
        numpy array of shape (num_samples, D)
    """
    print(
        f"Generating {num_samples} samples: dt={dt}, "
        f"stochasticity={stochasticity}, guide_temp={guide_temp}, "
        f"x1_temp={x1_temp}, guided={predictor_log_prob is not None}, "
        f"use_tag={use_tag}"
    )
    if batch_size > num_samples:
        batch_size = num_samples

    samples = []
    counter = 0
    while counter < num_samples:
        curr_batch = min(batch_size, num_samples - counter)
        x1 = flow_matching_sampling_masking_euler(
            denoising_model=denoising_model,
            batch_size=curr_batch,
            S=S, D=D, device=device,
            dt=dt, mask_idx=mask_idx,
            predictor_log_prob=predictor_log_prob,
            guide_temp=guide_temp,
            stochasticity=stochasticity,
            use_tag=use_tag,
            x1_temp=x1_temp,
            **kwargs,
        )
        samples.append(x1)
        counter += curr_batch
        print(f"{counter} out of {num_samples} generated")

    return np.concatenate(samples, axis=0)[:num_samples]


# ---------------------------------------------------------------------------
# ESM3 inverse folding wrappers
# ---------------------------------------------------------------------------


def esm_if_forward(model, xt, input_tokens):
    """Forward pass of ESM3 as inverse folding model.

    Args:
        model: ESM3 model
        xt: partially masked sequence tokens (B, D)
        input_tokens: list of ESMProteinTensor (batch templates with structure)

    Returns:
        logits (B, D, S)
    """
    import attr
    from esm.utils import generation
    from esm.sdk.api import LogitsConfig

    tokenizers = model.tokenizers
    sampled_tokens = [attr.evolve(tokens) for tokens in input_tokens]
    device = sampled_tokens[0].device

    sequence_lengths = [len(tokens) for tokens in sampled_tokens]
    batched_tokens = generation._stack_protein_tensors(
        sampled_tokens, sequence_lengths, tokenizers, device
    )

    xt_copy = xt.clone()
    xt_copy[..., 0] = tokenizers.sequence.cls_token_id
    xt_copy[..., -1] = tokenizers.sequence.eos_token_id
    setattr(batched_tokens, "sequence", xt_copy)

    forward_output = model.logits(batched_tokens, LogitsConfig(sequence=True))
    logits = forward_output.logits.sequence[..., : model.tokenizers.sequence.vocab_size]
    return logits


def esm_if_fm_sample(
    model, coords,
    num_samples=100, dt=0.01, x1_temp=0.1,
    stochasticity=0.0,
    predictor_log_prob=None,
    guide_temp=1.0,
    use_tag=True,
    batch_size=100,
    **kwargs,
):
    """Sample sequences via ESM3 inverse folding with optional guidance.

    Args:
        model: ESM3 model
        coords: atom37 coordinates (L, 37, 3)
        num_samples: number of sequences to generate
        dt: Euler step size
        x1_temp: temperature for model logits
        stochasticity: remasking noise
        predictor_log_prob: optional guidance function(xt, t) -> log_prob (B,)
        guide_temp: guidance temperature
        use_tag: use Taylor-Approximated Guidance
        batch_size: batch size for generation

    Returns:
        dict with 'proteins' (list of ESMProtein) and 'tokens' (stacked tensor)
    """
    import attr
    from esm.sdk.api import ESMProtein
    from esm.utils import generation
    from esm.utils.constants.esm3 import SEQUENCE_MASK_TOKEN

    if coords.shape[-1] != 3 or coords.shape[-2] != 37:
        raise ValueError(f"coords should be (L, 37, 3), got {coords.shape}")

    # Encode structure to get input tokens (adds cls/eos)
    batch_input_tokens = [
        model.encode(ESMProtein(coordinates=coords)) for _ in range(batch_size)
    ]
    input_tokens = batch_input_tokens[0]
    D = len(input_tokens)

    # Define denoising model wrapper
    def denoising_model(xt, t, **kw):
        xt[:, 0] = 0   # <cls>
        xt[:, -1] = 2   # <eos>
        logits = esm_if_forward(model, xt, batch_input_tokens)
        # Mask out special tokens and non-standard amino acids for inner positions
        logits[:, 0, 0] = 0.0
        logits[:, 0, 1:] = -float("inf")
        logits[:, -1, 0] = 0.0
        logits[:, -1, 1:] = -float("inf")
        logits[:, 1:-1, 0:4] = -float("inf")    # special tokens
        logits[:, 1:-1, 24:] = -float("inf")     # non-standard AAs and mask
        return logits

    # Run flow matching sampling
    samples = flow_matching_sampling(
        num_samples=num_samples,
        denoising_model=denoising_model,
        S=ESM3_ALPHABET_SIZE,
        D=D,
        device=torch.device("cuda"),
        dt=dt,
        mask_idx=SEQUENCE_MASK_TOKEN,
        batch_size=batch_size,
        stochasticity=stochasticity,
        x1_temp=x1_temp,
        predictor_log_prob=predictor_log_prob,
        guide_temp=guide_temp,
        use_tag=use_tag,
        **kwargs,
    )

    # Convert sampled tokens to ESMProtein objects
    sampled_proteins = dict(proteins=[], tokens=[])
    for i in range(num_samples):
        sampled_tokens = attr.evolve(input_tokens)
        setattr(
            sampled_tokens, "sequence", torch.tensor(samples[i].flatten(), dtype=int)
        )
        if len(sampled_tokens.sequence) != D:
            print(f"Skip sample {i} due to length mismatch..")
            continue

        # Set special tokens
        sampled_tokens.sequence[0] = model.tokenizers.sequence.cls_token_id
        sampled_tokens.sequence[-1] = model.tokenizers.sequence.eos_token_id

        sampled_proteins["tokens"].append(sampled_tokens.sequence)
        if torch.any(sampled_tokens.sequence == 32):
            raise ValueError("Mask token found in sampled sequence!")
        protein_sequence = model.decode(sampled_tokens)
        sampled_proteins["proteins"].append(protein_sequence)

    if sampled_proteins["tokens"]:
        sampled_proteins["tokens"] = torch.vstack(sampled_proteins["tokens"])
    return sampled_proteins
