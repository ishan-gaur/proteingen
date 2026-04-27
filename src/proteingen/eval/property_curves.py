"""Evaluate generative models by measuring property probability trajectories during generation."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import matplotlib
import matplotlib.pyplot as plt
import torch

from ..sampling.sampling import SamplingTrajectory

matplotlib.use("Agg")

_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

class PropertyTrajectory(TypedDict):
    """Result of property tracking during generation.

    time_points: (n_time_points,) — fraction of positions unmasked.
    p_y_gt_t: (n_sequences, n_time_points) — probability of exceeding threshold at each step.
    """
    time_points: torch.Tensor
    p_y_gt_t: torch.Tensor

def compute_property_trajectory_from_sampling(
    trajectory: SamplingTrajectory,
) -> PropertyTrajectory:
    """Extract property trajectories from a SamplingTrajectory.
    
    Args:
        trajectory: A SamplingTrajectory obtained with record_p_y_gt_t=True.
        
    Returns:
        PropertyTrajectory with time_points (>0 to 1) and p_y_gt_t.
    """
    step_p_y_gt_t = trajectory["step_p_y_gt_t"]
    if step_p_y_gt_t is None:
        raise ValueError("Trajectory must be generated with record_p_y_gt_t=True")
    
    S, n_steps = step_p_y_gt_t.shape
    
    # Each step unmasks some tokens. t ranges from >0 to 1.
    time_points = torch.linspace(1/n_steps, 1, n_steps)
    
    return PropertyTrajectory(
        time_points=time_points,
        p_y_gt_t=step_p_y_gt_t
    )

def plot_property_trajectories(
    trajectories: list[PropertyTrajectory],
    labels: list[str],
    output_path: str | Path,
    show_individual: bool = True,
    max_individual: int = 200,
    title: str = "Property probability trajectory during generation",
    ylabel: str = "p(y > t | x)",
) -> None:
    assert len(trajectories) == len(labels), (
        f"Got {len(trajectories)} trajectories but {len(labels)} labels"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (traj, label) in enumerate(zip(trajectories, labels)):
        color = _COLORS[idx % len(_COLORS)]
        t_np = traj["time_points"].numpy()
        p_val = traj["p_y_gt_t"]
        S = p_val.shape[0]

        if show_individual:
            for s in range(min(S, max_individual)):
                ax.plot(t_np, p_val[s].numpy(), alpha=0.08, color=color, linewidth=0.5)

        mean_p = torch.nanmean(p_val, dim=0).numpy()
        centered = p_val - torch.nanmean(p_val, dim=0, keepdim=True)
        n_valid = (~torch.isnan(p_val)).sum(dim=0).float().clamp(min=1)
        std_p = (torch.nan_to_num(centered**2).sum(dim=0) / n_valid).sqrt().numpy()

        ax.plot(t_np, mean_p, color=color, linewidth=2, label=label)
        ax.fill_between(
            t_np, mean_p - std_p, mean_p + std_p, alpha=0.15, color=color
        )

    ax.set_xlabel("Fraction unmasked (t)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
