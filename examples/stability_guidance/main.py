"""
Example: Stability-guided ESM3 inverse folding on Rocklin cluster 146 (5KPH).

Runs unguided vs. guided ESM3 sampling and evaluates with a stability oracle.
Produces a ddG histogram comparing the two approaches.

Uses DFM's sample_euler sampler with:
- gen model temp=0.1 (sharpens denoising distribution, equivalent to x1_temp)
- classifier temp=0.005 (guidance strength — lower = stronger)
- n_steps=100 (matching original dt=0.01)

Requirements:
    - ESM3 (pip install esm)
    - GPU with enough memory for ESM3 + classifier
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib
from scipy.stats import gaussian_kde

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from protstar.models.esm import ESM3
from protstar.models.rocklin_ddg.stability_predictor import (
    PreTrainedStabilityPredictor,
    StabilityPMPNN,
)
from protstar.models.rocklin_ddg.data_utils import compute_seq_id
from protstar.models.utils import pdb_to_atom37_and_seq
from protstar.sampling import sample_ctmc_linear_interpolation
from protstar.guide import TAG

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
device = "cuda"
num_samples = 100
batch_size = 50
n_steps = 100
gen_temp = 1.0
guide_temp = 0.03  # lower = stronger guidance

base_dir = Path(__file__).resolve().parent
pdb_path = base_dir / "data" / "structures" / "5KPH.pdb"
oracle_path = base_dir / "weights" / "stability_regression.pt"
classifier_path = base_dir / "weights" / "noisy_classifier_146_iter_1.pt"
output_dir = base_dir / "outputs"
output_dir.mkdir(exist_ok=True)


# -------------------------------------------------------------------------
# Helper: evaluate ddG using oracle model directly
# -------------------------------------------------------------------------
def predict_stability_raw(oracle_model, sequences, cond, device):
    """Get raw stability predictions (not log-sigmoid) for ddG computation."""
    from protstar.generative_modeling import MPNNTokenizer

    tokenizer = MPNNTokenizer()
    tokens = tokenizer(sequences)["input_ids"].to(device)
    B = tokens.shape[0]

    X = cond["X"][:1].expand(B, -1, -1, -1).to(device)
    mask = cond["mask"][:1].expand(B, -1).to(device)
    chain_M = cond["chain_M"][:1].expand(B, -1).to(device)
    residue_idx = cond["residue_idx"][:1].expand(B, -1).to(device)
    chain_encoding_all = cond["chain_encoding_all"][:1].expand(B, -1).to(device)

    with torch.no_grad():
        return (
            oracle_model(X, tokens, mask, chain_M, residue_idx, chain_encoding_all)
            .reshape(-1)
            .cpu()
            .numpy()
        )


def sample_batched(model, init_tokens, *, num_samples, batch_size, n_steps):
    """Run sample_euler in batches, return list of sequence strings."""
    all_seqs = []
    remaining = num_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        batch_init = init_tokens.expand(bs, -1).clone()
        seqs = sample_ctmc_linear_interpolation(model, batch_init, n_steps=n_steps, return_string=True)
        all_seqs.extend(seqs)
        remaining -= bs
        print(f"  {len(all_seqs)} / {num_samples} generated")
    return all_seqs[:num_samples]


# -------------------------------------------------------------------------
# 1. Load ESM3 model
# -------------------------------------------------------------------------
print("Loading ESM3 model...")
esm_model = ESM3().to(device)
esm_model.set_temp_(gen_temp)

# -------------------------------------------------------------------------
# 2. Load stability models
# -------------------------------------------------------------------------
print("Loading stability models...")
classifier = (
    PreTrainedStabilityPredictor(str(classifier_path), one_hot_encode_input=True)
    .to(device)
    .eval()
)

# Oracle: raw StabilityPMPNN for evaluation (not wrapped in PredictiveModel)
oracle_model = StabilityPMPNN.init(num_letters=21, vocab=21)
oracle_model.load_state_dict(torch.load(str(oracle_path), weights_only=False))
oracle_model = oracle_model.to(device).eval()

# -------------------------------------------------------------------------
# 3. Prepare conditioning inputs
# -------------------------------------------------------------------------
print("Preparing conditioning inputs...")
coords_RAX, wt_seq = pdb_to_atom37_and_seq(str(pdb_path), backbone_only=True)
print(f"Wildtype sequence ({len(wt_seq)} aa): {wt_seq}")

# Structure conditioning for ESM3 inverse folding
esm_model.set_condition_({"coords_RAX": coords_RAX})

# Structure conditioning for stability classifier (PMPNN featurization)
stability_cond = PreTrainedStabilityPredictor.prepare_conditioning(
    str(pdb_path), device=device
)
classifier.set_condition_(stability_cond)
classifier.set_temp_(guide_temp)

# Initial all-mask tokens
tokenizer = esm_model.tokenizer
init_tokens = tokenizer(
    ["<mask>" * len(wt_seq)], return_tensors="pt",
)["input_ids"].to(device)

# -------------------------------------------------------------------------
# 4. Run UNGUIDED ESM3 inverse folding
# -------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("Running UNGUIDED ESM3 inverse folding...")
print(f"{'=' * 60}")
unguided_seqs = sample_batched(
    esm_model,
    init_tokens,
    num_samples=num_samples,
    batch_size=batch_size,
    n_steps=n_steps,
)
print(f"Unguided: {len(unguided_seqs)} sequences generated")
for i, s in enumerate(unguided_seqs[:3]):
    print(f"  {i}: {s}")

# -------------------------------------------------------------------------
# 5. Run GUIDED ESM3 inverse folding (TAG)
# -------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("Running GUIDED ESM3 inverse folding (TAG)...")
print(f"{'=' * 60}")
guided_model = TAG(
    esm_model,
    classifier,
    use_clean_classifier=False,
).to(device)

guided_seqs = sample_batched(
    guided_model,
    init_tokens,
    num_samples=num_samples,
    batch_size=batch_size,
    n_steps=n_steps,
)
print(f"Guided: {len(guided_seqs)} sequences generated")
for i, s in enumerate(guided_seqs[:3]):
    print(f"  {i}: {s}")

# -------------------------------------------------------------------------
# 6. Evaluate ddG (predicted stability relative to wildtype)
# -------------------------------------------------------------------------
print("\nEvaluating predicted ddG...")
unguided_preds = predict_stability_raw(oracle_model, unguided_seqs, stability_cond, device)
guided_preds = predict_stability_raw(oracle_model, guided_seqs, stability_cond, device)
wt_pred = predict_stability_raw(oracle_model, [str(wt_seq)], stability_cond, device)

# ddG = pred - wt_pred, sign-flipped for manuscript convention (lower = more stable)
unguided_ddg = -1.0 * (unguided_preds - wt_pred[0])
guided_ddg = -1.0 * (guided_preds - wt_pred[0])

unguided_seq_ids = [compute_seq_id(s, str(wt_seq)) for s in unguided_seqs]
guided_seq_ids = [compute_seq_id(s, str(wt_seq)) for s in guided_seqs]

# -------------------------------------------------------------------------
# 7. Print summary statistics
# -------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("Results Summary (lower ddG = more stable)")
print(f"{'=' * 60}")
print(f"{'Metric':<30} {'Unguided':>12} {'Guided':>12}")
print(f"{'-' * 54}")
print(f"{'Mean ddG':.<30} {np.mean(unguided_ddg):>12.3f} {np.mean(guided_ddg):>12.3f}")
print(
    f"{'Median ddG':.<30} {np.median(unguided_ddg):>12.3f} {np.median(guided_ddg):>12.3f}"
)
print(
    f"{'Frac ddG <= 0':.<30} {(unguided_ddg <= 0).mean():>12.3f} {(guided_ddg <= 0).mean():>12.3f}"
)
print(
    f"{'Mean seq identity':.<30} {np.mean(unguided_seq_ids):>12.3f} {np.mean(guided_seq_ids):>12.3f}"
)

# -------------------------------------------------------------------------
# 8. Plot ddG histogram with KDE
# -------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

bins = np.linspace(
    min(unguided_ddg.min(), guided_ddg.min()) - 0.5,
    max(unguided_ddg.max(), guided_ddg.max()) + 0.5,
    30,
)

ax.hist(
    unguided_ddg,
    bins=bins,
    alpha=0.3,
    color="tab:blue",
    density=True,
    edgecolor="white",
)
ax.hist(
    guided_ddg,
    bins=bins,
    alpha=0.3,
    color="tab:orange",
    density=True,
    edgecolor="white",
)

x_grid = np.linspace(bins[0], bins[-1], 200)
kde_unguided = gaussian_kde(unguided_ddg)
kde_guided = gaussian_kde(guided_ddg)
ax.plot(
    x_grid,
    kde_unguided(x_grid),
    color="tab:blue",
    linewidth=2,
    label=f"Unguided (mean={np.mean(unguided_ddg):.2f})",
)
ax.plot(
    x_grid,
    kde_guided(x_grid),
    color="tab:orange",
    linewidth=2,
    label=f"Guided (mean={np.mean(guided_ddg):.2f})",
)

ax.axvline(0, color="black", linestyle="--", linewidth=1, label="ddG = 0")
ax.set_xlabel(
    r"Predicted $\Delta\Delta G$ (lower $\leftarrow$ more stable)", fontsize=13
)
ax.set_ylabel("Density", fontsize=13)
ax.set_title(
    "Stability-Guided vs Unguided ESM3 Inverse Folding\n(Rocklin cluster 146 / 5KPH)",
    fontsize=14,
)
ax.legend(fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()

hist_path = output_dir / "ddg_histogram.png"
fig.savefig(hist_path, dpi=150)
print(f"\nHistogram saved to {hist_path}")

# -------------------------------------------------------------------------
# 9. Plot sequence identity vs ddG scatter
# -------------------------------------------------------------------------
fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5))
ax2.scatter(
    unguided_seq_ids,
    unguided_ddg,
    alpha=0.5,
    color="tab:blue",
    label="Unguided",
    s=30,
)
ax2.scatter(
    guided_seq_ids,
    guided_ddg,
    alpha=0.5,
    color="tab:orange",
    label="Guided",
    s=30,
)
ax2.axhline(0, color="black", linestyle="--", linewidth=1)
ax2.set_xlabel("Sequence Identity to Wildtype", fontsize=13)
ax2.set_ylabel(r"Predicted $\Delta\Delta G$ (lower = more stable)", fontsize=13)
ax2.set_title("Sequence Identity vs Stability", fontsize=14)
ax2.legend(fontsize=11)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
plt.tight_layout()

scatter_path = output_dir / "seqid_vs_ddg.png"
fig2.savefig(scatter_path, dpi=150)
print(f"Scatter plot saved to {scatter_path}")

plt.close("all")
print("\nDone!")
