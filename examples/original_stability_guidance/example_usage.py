"""
Example: Stability-guided ESM3 inverse folding on Rocklin cluster 146 (5KPH).

Runs unguided vs. guided ESM3 sampling and evaluates with a stability oracle.
Produces a ddG histogram comparing the two approaches.

Requirements:
    - ESM3 (pip install esm)
    - GPU with enough memory for ESM3 + classifier
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add this directory to path so local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predictor_model import load_stability_model
from data_utils import (
    prepare_conditioning_inputs,
    format_coords_to_esm3,
    load_pdb_to_graph_dict,
    ESM_ALPHABET,
    PMPNN_ALPHABET,
    esm_tok_to_pmpnn_tok,
    compute_seq_id,
)
from guidance_utils import (
    get_predictor_log_prob,
    compute_ddg,
    esm_if_fm_sample,
)


def main():
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    device = "cuda"
    num_samples = 100
    batch_size = 50     # reduce if OOM
    dt = 0.01
    x1_temp = 0.1
    guide_temp = 0.01
    stochasticity = 0.0
    use_tag = True      # Taylor-Approximated Guidance

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdb_path = os.path.join(base_dir, "data", "structures", "5KPH.pdb")
    oracle_path = os.path.join(base_dir, "weights", "stability_regression.pt")
    classifier_path = os.path.join(
        base_dir, "weights", "noisy_classifier_146_iter_1.pt"
    )
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Load ESM3 model
    # -------------------------------------------------------------------------
    print("Loading ESM3 model...")
    from esm.models.esm3 import ESM3

    esm_model = ESM3.from_pretrained("esm3-open").to(device)
    esm_model.eval()

    # -------------------------------------------------------------------------
    # 2. Load stability models
    # -------------------------------------------------------------------------
    print("Loading stability models...")
    # Noisy classifier for guidance (one_hot_encode_input=True enables TAG)
    classifier = load_stability_model(
        classifier_path, vocab_size=21, device=device, one_hot_encode_input=True
    )
    classifier.eval()

    # Oracle regression model for evaluation
    oracle = load_stability_model(
        oracle_path, vocab_size=21, device=device, one_hot_encode_input=False
    )
    oracle.eval()

    # -------------------------------------------------------------------------
    # 3. Prepare conditioning inputs (PMPNN alphabet for predictors)
    # -------------------------------------------------------------------------
    print("Preparing conditioning inputs...")
    cond_inputs = prepare_conditioning_inputs(pdb_path, batch_size, device=device)
    wt_seq = cond_inputs["wt_seq"]
    print(f"Wildtype sequence ({len(wt_seq)} aa): {wt_seq}")

    # -------------------------------------------------------------------------
    # 4. Format coordinates for ESM3 (atom37)
    # -------------------------------------------------------------------------
    graph = load_pdb_to_graph_dict(pdb_path)
    bb_coords = np.stack(
        [
            graph["coords_chain_A"]["N_chain_A"],
            graph["coords_chain_A"]["CA_chain_A"],
            graph["coords_chain_A"]["C_chain_A"],
            graph["coords_chain_A"]["O_chain_A"],
        ],
        axis=1,
    )  # (L, 4, 3)
    coords_esm3 = format_coords_to_esm3(bb_coords, device=device)  # (L, 37, 3)

    # -------------------------------------------------------------------------
    # 5. Run UNGUIDED ESM3 inverse folding
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Running UNGUIDED ESM3 inverse folding...")
    print(f"{'='*60}")
    unguided_results = esm_if_fm_sample(
        model=esm_model,
        coords=coords_esm3,
        num_samples=num_samples,
        dt=dt,
        x1_temp=x1_temp,
        stochasticity=stochasticity,
        predictor_log_prob=None,
        batch_size=batch_size,
    )

    # -------------------------------------------------------------------------
    # 6. Run GUIDED ESM3 inverse folding
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Running GUIDED ESM3 inverse folding (TAG)...")
    print(f"{'='*60}")
    predictor_log_prob = get_predictor_log_prob(classifier, cond_inputs)
    guided_results = esm_if_fm_sample(
        model=esm_model,
        coords=coords_esm3,
        num_samples=num_samples,
        dt=dt,
        x1_temp=x1_temp,
        stochasticity=stochasticity,
        predictor_log_prob=predictor_log_prob,
        guide_temp=guide_temp,
        use_tag=use_tag,
        batch_size=batch_size,
    )

    # -------------------------------------------------------------------------
    # 7. Extract sequences
    # -------------------------------------------------------------------------
    unguided_seqs = [p.sequence for p in unguided_results["proteins"]]
    guided_seqs = [p.sequence for p in guided_results["proteins"]]

    print(f"\nUnguided: {len(unguided_seqs)} sequences generated")
    print(f"Guided:   {len(guided_seqs)} sequences generated")

    # -------------------------------------------------------------------------
    # 8. Evaluate ddG (predicted stability relative to wildtype)
    # -------------------------------------------------------------------------
    print("\nEvaluating predicted ddG...")
    # Need cond_inputs sized for evaluation batches
    eval_batch_size = max(len(unguided_seqs), len(guided_seqs))
    eval_cond = prepare_conditioning_inputs(pdb_path, eval_batch_size, device=device)

    unguided_ddg_raw = compute_ddg(unguided_seqs, oracle, eval_cond, device=device)
    guided_ddg_raw = compute_ddg(guided_seqs, oracle, eval_cond, device=device)

    # Sign flip: Rocklin convention (higher = more stable) -> manuscript convention (lower = more stable)
    unguided_ddg = -1.0 * unguided_ddg_raw
    guided_ddg = -1.0 * guided_ddg_raw

    # Compute sequence identity to wildtype
    unguided_seq_ids = [compute_seq_id(s, wt_seq) for s in unguided_seqs]
    guided_seq_ids = [compute_seq_id(s, wt_seq) for s in guided_seqs]

    # -------------------------------------------------------------------------
    # 9. Print summary statistics
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Results Summary (lower ddG = more stable)")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'Unguided':>12} {'Guided':>12}")
    print(f"{'-'*54}")
    print(f"{'Mean ddG':.<30} {np.mean(unguided_ddg):>12.3f} {np.mean(guided_ddg):>12.3f}")
    print(f"{'Median ddG':.<30} {np.median(unguided_ddg):>12.3f} {np.median(guided_ddg):>12.3f}")
    print(f"{'Frac ddG <= 0':.<30} {(unguided_ddg <= 0).mean():>12.3f} {(guided_ddg <= 0).mean():>12.3f}")
    print(f"{'Mean seq identity':.<30} {np.mean(unguided_seq_ids):>12.3f} {np.mean(guided_seq_ids):>12.3f}")

    # -------------------------------------------------------------------------
    # 10. Plot ddG histogram with KDE
    # -------------------------------------------------------------------------
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    bins = np.linspace(
        min(unguided_ddg.min(), guided_ddg.min()) - 0.5,
        max(unguided_ddg.max(), guided_ddg.max()) + 0.5,
        30,
    )

    ax.hist(
        unguided_ddg, bins=bins, alpha=0.3, color="tab:blue",
        density=True, edgecolor="white",
    )
    ax.hist(
        guided_ddg, bins=bins, alpha=0.3, color="tab:orange",
        density=True, edgecolor="white",
    )

    # KDE overlay
    x_grid = np.linspace(bins[0], bins[-1], 200)
    kde_unguided = gaussian_kde(unguided_ddg)
    kde_guided = gaussian_kde(guided_ddg)
    ax.plot(x_grid, kde_unguided(x_grid), color="tab:blue", linewidth=2,
            label=f"Unguided (mean={np.mean(unguided_ddg):.2f})")
    ax.plot(x_grid, kde_guided(x_grid), color="tab:orange", linewidth=2,
            label=f"Guided (mean={np.mean(guided_ddg):.2f})")

    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="ddG = 0")
    ax.set_xlabel(r"Predicted $\Delta\Delta G$ (lower $\leftarrow$ more stable)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("Stability-Guided vs Unguided ESM3 Inverse Folding\n(Rocklin cluster 146 / 5KPH)", fontsize=14)
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    hist_path = os.path.join(output_dir, "ddg_histogram.png")
    fig.savefig(hist_path, dpi=150)
    print(f"\nHistogram saved to {hist_path}")

    # -------------------------------------------------------------------------
    # 11. Plot sequence identity vs ddG scatter
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5))
    ax2.scatter(
        unguided_seq_ids, unguided_ddg, alpha=0.5, color="tab:blue",
        label="Unguided", s=30,
    )
    ax2.scatter(
        guided_seq_ids, guided_ddg, alpha=0.5, color="tab:orange",
        label="Guided", s=30,
    )
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_xlabel("Sequence Identity to Wildtype", fontsize=13)
    ax2.set_ylabel(r"Predicted $\Delta\Delta G$ (lower = more stable)", fontsize=13)
    ax2.set_title("Sequence Identity vs Stability", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()

    scatter_path = os.path.join(output_dir, "seqid_vs_ddg.png")
    fig2.savefig(scatter_path, dpi=150)
    print(f"Scatter plot saved to {scatter_path}")

    plt.close("all")
    print("\nDone!")


if __name__ == "__main__":
    main()
