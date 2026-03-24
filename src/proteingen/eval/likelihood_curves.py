"""Evaluate transition models by measuring log-likelihood along noising trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import matplotlib
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from proteingen.generative_modeling import TransitionModel

matplotlib.use("Agg")


class LogProbTrajectory(TypedDict):
    """Result of plot_p_x_trajectory.

    time_points: (n_time_points,) — fraction of positions unmasked at each step.
    avg_log_probs: (n_sequences, n_time_points) — per-sequence average log p(x_true)
        at masked positions. NaN where a sequence had no masked positions.
    """

    time_points: torch.Tensor
    avg_log_probs: torch.Tensor


@torch.no_grad()
def plot_p_x_trajectory(
    sequences: list[str],
    model: TransitionModel,
    n_time_points: int,
    output_path: str | Path,
    batch_size: int = 32,
) -> LogProbTrajectory:
    """Compute and plot average log-probability trajectories under progressive unmasking.

    For each of n_time_points evenly-spaced noise levels t in [0, 1), randomly
    masks each sequence position with probability (1 - t), then measures the
    model's average log p(true token) at the masked positions.

    At t ≈ 0: nearly everything is masked (little context → low log prob).
    At t ≈ 1: nearly everything is revealed (rich context → high log prob).

    Args:
        sequences: protein sequences to evaluate.
        model: a TransitionModel (e.g. ESMC wrapped with MaskedModelLogitFormatter).
        n_time_points: number of evenly-spaced noise levels to evaluate.
        output_path: file path for the saved plot (e.g. "trajectory.png").
        batch_size: sequences per forward pass.

    Returns:
        LogProbTrajectory with time_points and per-sequence avg log probs.
    """
    tokenizer = model.tokenizer
    mask_token_id = tokenizer.mask_token_id
    assert mask_token_id is not None, "Tokenizer must have a mask_token_id"

    # Tokenize all sequences (CPU)
    tokenized = tokenizer(sequences, padding=True, return_tensors="pt")
    true_tokens_SP = tokenized["input_ids"]  # (S, P)
    S, P = true_tokens_SP.shape

    # Identify maskable positions (non-special tokens in the original sequence)
    special_ids: set[int] = set()
    for attr in ("cls_token_id", "eos_token_id", "pad_token_id", "mask_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_ids.add(tid)
    if hasattr(tokenizer, "added_tokens_decoder"):
        special_ids |= set(tokenizer.added_tokens_decoder.keys())

    maskable_SP = torch.ones(S, P, dtype=torch.bool)
    for sid in special_ids:
        maskable_SP &= true_tokens_SP != sid

    # n_time_points points from 0 to just below 1
    time_points = torch.linspace(0, 1, n_time_points + 1)[:-1]
    avg_log_probs = torch.full((S, n_time_points), float("nan"))

    for t_idx, t in enumerate(tqdm(time_points, desc="Evaluating noise levels")):
        # Each maskable position is kept (unmasked) with probability t
        keep_SP = torch.rand(S, P) < t
        to_mask_SP = maskable_SP & ~keep_SP

        noised_SP = true_tokens_SP.clone()
        noised_SP[to_mask_SP] = mask_token_id

        # Forward pass in batches → collect log probs on CPU
        log_prob_chunks: list[torch.Tensor] = []
        for start in range(0, S, batch_size):
            end = min(start + batch_size, S)
            lp = model.get_log_probs(noised_SP[start:end].to(model.device))
            log_prob_chunks.append(lp.cpu())
        log_probs_SPT = torch.cat(log_prob_chunks, dim=0)  # (S, P, T)

        # Gather log prob of the true token at every position
        T = log_probs_SPT.shape[-1]
        safe_idx = true_tokens_SP.clamp(max=T - 1)  # safety for special-token indices
        true_lp_SP = log_probs_SPT.gather(2, safe_idx.unsqueeze(2)).squeeze(2)  # (S, P)

        # Average only over masked positions
        true_lp_SP[~to_mask_SP] = 0.0
        n_masked_S = to_mask_SP.sum(dim=1).float()
        seq_avg = true_lp_SP.sum(dim=1) / n_masked_S.clamp(min=1)
        seq_avg[n_masked_S == 0] = float("nan")
        avg_log_probs[:, t_idx] = seq_avg

    # ── Plot ─────────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    t_np = time_points.numpy()

    # Individual trajectories (cap drawn curves to keep the plot readable)
    for s in range(min(S, 200)):
        ax.plot(
            t_np, avg_log_probs[s].numpy(), alpha=0.1, color="steelblue", linewidth=0.5
        )

    # Mean ± std (NaN-safe)
    mean_lp = torch.nanmean(avg_log_probs, dim=0).numpy()
    centered = avg_log_probs - torch.nanmean(avg_log_probs, dim=0, keepdim=True)
    n_valid = (~torch.isnan(avg_log_probs)).sum(dim=0).float().clamp(min=1)
    std_lp = (torch.nan_to_num(centered**2).sum(dim=0) / n_valid).sqrt().numpy()

    ax.plot(t_np, mean_lp, color="navy", linewidth=2, label="Mean")
    ax.fill_between(t_np, mean_lp - std_lp, mean_lp + std_lp, alpha=0.2, color="navy")

    ax.set_xlabel("Fraction unmasked (t)")
    ax.set_ylabel("Avg log p(x_true) at masked positions")
    ax.set_title("Log-likelihood trajectory under progressive unmasking")
    ax.legend()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return LogProbTrajectory(time_points=time_points, avg_log_probs=avg_log_probs)
