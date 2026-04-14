"""Evaluate generative models by measuring log-likelihood along noising trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from protstar.generative_modeling import GenerativeModel

matplotlib.use("Agg")

# Color palette for multi-condition plots (up to 10 conditions)
_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


class LogProbTrajectory(TypedDict):
    """Result of compute_log_prob_trajectory.

    time_points: (n_time_points,) — fraction of positions unmasked at each step.
    avg_log_probs: (n_sequences, n_time_points) — per-sequence average log p(x_true)
        at masked positions. NaN where a sequence had no masked positions.
    """

    time_points: torch.Tensor
    avg_log_probs: torch.Tensor


class DecodingLogProbTrajectory(TypedDict):
    """Per-sequence teacher-forced decode trajectories.

    percent_unmasked: list[(n_steps_s,)] — per-sequence fraction of positions
        already unmasked before each decode step.
    decoded_position_log_probs: list[(n_steps_s,)] — log p(true token) at the
        position decoded at each step, in the same order.
    """

    percent_unmasked: list[torch.Tensor]
    decoded_position_log_probs: list[torch.Tensor]


def _get_maskable_positions(
    true_tokens_SP: torch.LongTensor, tokenizer
) -> torch.BoolTensor:
    """Return maskability mask for tokenized sequences."""
    S, P = true_tokens_SP.shape
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
    return maskable_SP


@torch.no_grad()
def compute_decoding_log_prob_trajectory(
    sequences: list[str],
    model: GenerativeModel,
    orders: list[torch.LongTensor],
    batch_size: int = 32,
) -> DecodingLogProbTrajectory:
    """Teacher-forced log p(true token) along fixed decoding orders.

    Each sequence starts with all order positions masked. At decode step k, the
    model predicts the true residue at order[k] given the partially unmasked
    context from earlier steps (order[:k]). This returns one trajectory per
    sequence at native step resolution.

    Args:
        sequences: protein sequences to evaluate.
        model: a GenerativeModel with a tokenizer mask token.
        orders: one token-position order per sequence (same tokenizer space as model).
        batch_size: number of decode steps to score per forward pass.

    Returns:
        DecodingLogProbTrajectory with per-sequence step fractions and log probs.
    """
    if len(sequences) != len(orders):
        raise ValueError(
            f"Got {len(sequences)} sequences but {len(orders)} orders; expected one order per sequence"
        )

    tokenizer = model.tokenizer
    mask_token_id = tokenizer.mask_token_id
    assert mask_token_id is not None, "Tokenizer must have a mask_token_id"

    tokenized = tokenizer(sequences, padding=True, return_tensors="pt")
    true_tokens_SP = tokenized["input_ids"]  # [S, P]
    maskable_SP = _get_maskable_positions(true_tokens_SP, tokenizer)  # [S, P]

    percent_unmasked: list[torch.Tensor] = []
    decoded_position_log_probs: list[torch.Tensor] = []

    for s_idx, order in enumerate(tqdm(orders, desc="Scoring decode trajectories")):
        order = order.to(torch.long).cpu()
        if order.numel() == 0:
            percent_unmasked.append(torch.empty(0, dtype=torch.float32))
            decoded_position_log_probs.append(torch.empty(0, dtype=torch.float32))
            continue

        valid_positions = maskable_SP[s_idx]  # [P]
        if not bool(valid_positions[order].all()):
            raise ValueError(
                f"Order for sequence {s_idx} contains non-maskable positions"
            )

        n_steps = order.numel()
        x_masked_P = true_tokens_SP[s_idx].clone()  # [P]
        x_masked_P[order] = mask_token_id

        seq_log_probs = torch.empty(n_steps, dtype=torch.float32)  # [K]

        for start in range(0, n_steps, batch_size):
            end = min(start + batch_size, n_steps)
            batch_states = []
            step_positions = []

            for step_idx in range(start, end):
                state_P = x_masked_P.clone()  # [P]
                if step_idx > 0:
                    revealed = order[:step_idx]  # [step_idx]
                    state_P[revealed] = true_tokens_SP[s_idx, revealed]
                batch_states.append(state_P)
                step_positions.append(int(order[step_idx]))

            state_BP = torch.stack(batch_states, dim=0)  # [B, P]
            log_probs_BPT = model.get_log_probs(
                state_BP.to(model.device)
            ).cpu()  # [B, P, T]
            pos_B = torch.tensor(step_positions, dtype=torch.long)  # [B]
            true_tokens_B = true_tokens_SP[s_idx, pos_B].to(torch.long)  # [B]
            row_B = torch.arange(pos_B.numel(), dtype=torch.long)  # [B]
            seq_log_probs[start:end] = log_probs_BPT[row_B, pos_B, true_tokens_B]

        step_idx_K = torch.arange(n_steps, dtype=torch.float32)  # [K]
        frac_K = step_idx_K / float(n_steps)  # [K]
        percent_unmasked.append(frac_K)
        decoded_position_log_probs.append(seq_log_probs)

    return DecodingLogProbTrajectory(
        percent_unmasked=percent_unmasked,
        decoded_position_log_probs=decoded_position_log_probs,
    )


@torch.no_grad()
def compute_log_prob_trajectory(
    sequences: list[str],
    model: GenerativeModel,
    n_time_points: int,
    batch_size: int = 32,
) -> LogProbTrajectory:
    """Compute average log-probability trajectories under progressive unmasking.

    For each of n_time_points evenly-spaced noise levels t in [0, 1), randomly
    masks each sequence position with probability (1 - t), then measures the
    model's average log p(true token) at the masked positions.

    At t ≈ 0: nearly everything is masked (little context → low log prob).
    At t ≈ 1: nearly everything is revealed (rich context → high log prob).

    Args:
        sequences: protein sequences to evaluate.
        model: a GenerativeModel (e.g. ESMC wrapped with MaskedModelLogitFormatter).
        n_time_points: number of evenly-spaced noise levels to evaluate.
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
    maskable_SP = _get_maskable_positions(true_tokens_SP, tokenizer)

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

    return LogProbTrajectory(time_points=time_points, avg_log_probs=avg_log_probs)


def plot_log_prob_trajectories(
    trajectories: list[LogProbTrajectory],
    labels: list[str],
    output_path: str | Path,
    show_individual: bool = True,
    max_individual: int = 200,
    title: str = "Log-likelihood trajectory under progressive unmasking",
) -> None:
    """Plot one or more log-probability trajectories on a single figure.

    Each trajectory is drawn as a mean ± std band, optionally with individual
    sequence curves behind it.

    Args:
        trajectories: list of LogProbTrajectory dicts to plot.
        labels: display name for each trajectory (must match len(trajectories)).
        output_path: file path for the saved plot.
        show_individual: if True, draw faint per-sequence lines behind the mean.
        max_individual: cap on per-sequence lines drawn per trajectory.
        title: plot title.
    """
    assert len(trajectories) == len(labels), (
        f"Got {len(trajectories)} trajectories but {len(labels)} labels"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (traj, label) in enumerate(zip(trajectories, labels)):
        color = _COLORS[idx % len(_COLORS)]
        t_np = traj["time_points"].numpy()
        avg_lp = traj["avg_log_probs"]
        S = avg_lp.shape[0]

        # Individual trajectories
        if show_individual:
            for s in range(min(S, max_individual)):
                ax.plot(t_np, avg_lp[s].numpy(), alpha=0.08, color=color, linewidth=0.5)

        # Mean ± std (NaN-safe)
        mean_lp = torch.nanmean(avg_lp, dim=0).numpy()
        centered = avg_lp - torch.nanmean(avg_lp, dim=0, keepdim=True)
        n_valid = (~torch.isnan(avg_lp)).sum(dim=0).float().clamp(min=1)
        std_lp = (torch.nan_to_num(centered**2).sum(dim=0) / n_valid).sqrt().numpy()

        ax.plot(t_np, mean_lp, color=color, linewidth=2, label=label)
        ax.fill_between(
            t_np, mean_lp - std_lp, mean_lp + std_lp, alpha=0.15, color=color
        )

    ax.set_xlabel("Fraction unmasked (t)")
    ax.set_ylabel("Avg log p(x_true) at masked positions")
    ax.set_title(title)
    ax.legend()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_decoding_log_prob_trajectories(
    trajectories: list[DecodingLogProbTrajectory],
    labels: list[str],
    output_path: str | Path,
    show_individual: bool = False,
    max_individual: int = 20,
    n_grid_points: int = 100,
    title: str = "Teacher-forced decoding log-likelihood",
) -> None:
    """Plot teacher-forced decode trajectories on a normalized x-axis.

    Sequence lengths differ, so each per-sequence curve is linearly interpolated
    to a shared [0, 1] grid (percent unmasked) before model-level aggregation.
    """
    assert len(trajectories) == len(labels), (
        f"Got {len(trajectories)} trajectories but {len(labels)} labels"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grid = np.linspace(0.0, 1.0, n_grid_points)
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (traj, label) in enumerate(zip(trajectories, labels)):
        color = _COLORS[idx % len(_COLORS)]
        per_seq_interp = []

        for frac_K, lp_K in zip(
            traj["percent_unmasked"],
            traj["decoded_position_log_probs"],
        ):
            if frac_K.numel() == 0:
                continue

            x = frac_K.detach().cpu().numpy()
            y = lp_K.detach().cpu().numpy()
            if x[-1] < 1.0:
                x = np.concatenate([x, np.array([1.0])])
                y = np.concatenate([y, np.array([y[-1]])])

            interp = np.interp(grid, x, y)
            per_seq_interp.append(interp)

            if show_individual and len(per_seq_interp) <= max_individual:
                ax.plot(grid, interp, color=color, alpha=0.1, linewidth=0.5)

        if not per_seq_interp:
            continue

        interp_arr = np.asarray(per_seq_interp)
        mean = interp_arr.mean(axis=0)
        std = interp_arr.std(axis=0)

        ax.plot(grid, mean, color=color, linewidth=2, label=label)
        ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Percent unmasked")
    ax.set_ylabel("Log p(true token at decoded position)")
    ax.set_title(title)
    ax.legend()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
