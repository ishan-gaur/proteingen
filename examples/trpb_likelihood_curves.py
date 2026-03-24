"""Plot ESMC log-likelihood trajectories on the TrpB fitness dataset.

Evaluates how well ESMC predicts masked amino acids as more of the
sequence is revealed, using real TrpB sequences from SaProtHub.

Usage:
    uv run python examples/trpb_likelihood_curves.py --device cuda
    uv run python examples/trpb_likelihood_curves.py --device cpu --n-sequences 10 --n-time-points 5
"""

import argparse

import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from proteingen.eval.likelihood_curves import plot_p_x_trajectory
from proteingen.models.esm import ESMC

DATASET_ID = "SaProtHub/Dataset-TrpB_fitness_landsacpe"


def main():
    parser = argparse.ArgumentParser(description="ESMC likelihood curves on TrpB")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--esmc-checkpoint", default="esmc_300m")
    parser.add_argument("--n-sequences", type=int, default=50)
    parser.add_argument("--n-time-points", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="trpb_likelihood_curves.png")
    args = parser.parse_args()

    print("Loading TrpB dataset...")
    path = hf_hub_download(DATASET_ID, "dataset.csv", repo_type="dataset")
    df = pd.read_csv(path)
    sequences = df["protein"].tolist()[: args.n_sequences]
    print(f"  {len(sequences)} sequences, length {len(sequences[0])}")

    print(f"Loading ESMC ({args.esmc_checkpoint})...")
    model = ESMC(args.esmc_checkpoint).to(args.device).eval()

    print(f"Computing likelihood trajectories ({args.n_time_points} time points)...")
    result = plot_p_x_trajectory(
        sequences=sequences,
        model=model,
        n_time_points=args.n_time_points,
        output_path=args.output,
        batch_size=args.batch_size,
    )

    print(f"\nSaved plot to {args.output}")
    mean_lp = torch.nanmean(result["avg_log_probs"], dim=0)
    for t, lp in zip(result["time_points"], mean_lp):
        print(f"  t={t:.2f}  mean log p = {lp:.3f}")


if __name__ == "__main__":
    main()
