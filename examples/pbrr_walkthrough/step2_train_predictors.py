"""Step 2: Train and compare predictive models on PbrR data.

Trains three predictive models on the same train/test split as the original
guidance paper:
  1. Linear probe on ESMC embeddings
  2. OHE-encoded MLP (OneHotMLP)
  3. XGBoost random forest regressor

Evaluates each on the held-out test set using RMSE, R², and Spearman ρ
for both Pb and Zn fold-change predictions. Produces a comparison bar chart
and scatter plots of predicted vs actual values.

Usage:
    uv run python examples/pbrr_walkthrough/step2_train_predictors.py [--device cuda]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from proteingen.models.esm import ESMC
from proteingen.models.xgboost_model import XGBoostPredictor
from proteingen.predictive_modeling import (
    LinearProbe,
    OneHotMLP,
)

matplotlib.use("Agg")

# Allow running as `uv run python examples/pbrr_walkthrough/step2_train_predictors.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import (
    CHECKPOINTS_FOLDER,
    OUTPUTS_FOLDER,
    load_round1_data,
    load_splits,
)


# ── Concrete predictive model subclasses ─────────────────────────────────────

# We need concrete subclasses that implement format_raw_to_logits.
# For Step 2 we only need predict() (raw regression), but the ABC requires it.


class PbrRLinearProbe(LinearProbe):
    """ESMC linear probe for PbrR Pb/Zn FC regression."""

    def __init__(self, esmc_checkpoint: str = "esmc_300m"):
        esmc = ESMC(esmc_checkpoint)
        super().__init__(embed_model=esmc, output_dim=2)  # predict [Pb, Zn]

    def _save_args(self) -> dict:
        return {"esmc_checkpoint": self.embed_model._esmc_checkpoint}

    def format_raw_to_logits(self, raw_output, ohe_seq_SPT, **kwargs):
        # Not used for training/evaluation, only for guidance
        raise NotImplementedError("Use predict() for regression evaluation")


class PbrROneHotMLP(OneHotMLP):
    """OHE-encoded MLP for PbrR Pb/Zn FC regression."""

    def __init__(
        self,
        sequence_length: int = 147,  # L+2 with BOS/EOS
        model_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

        tokenizer = EsmSequenceTokenizer()
        super().__init__(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            model_dim=model_dim,
            n_layers=n_layers,
            output_dim=2,  # predict [Pb, Zn]
            dropout=dropout,
        )

    def format_raw_to_logits(self, raw_output, ohe_seq_SPT, **kwargs):
        raise NotImplementedError("Use predict() for regression evaluation")


class PbrRXGBoost(XGBoostPredictor):
    """XGBoost regressor for PbrR Pb/Zn FC prediction."""

    def __init__(self, **xgb_kwargs):
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

        tokenizer = EsmSequenceTokenizer()
        super().__init__(tokenizer=tokenizer, output_dim=2, **xgb_kwargs)

    def format_raw_to_logits(self, raw_output, ohe_seq_SPT, **kwargs):
        raise NotImplementedError("Use predict() for regression evaluation")


# ── Training helpers ─────────────────────────────────────────────────────────


def train_linear_probe(
    model: PbrRLinearProbe,
    train_seqs: list[str],
    train_labels: torch.FloatTensor,
    val_seqs: list[str],
    val_labels: torch.FloatTensor,
    device: torch.device,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    embed_batch_size: int = 16,
) -> dict:
    """Train linear probe: precompute embeddings, then train MLP head."""
    print("  Computing train embeddings...")
    train_emb = model.precompute_embeddings(train_seqs, embed_batch_size, device)
    print("  Computing val embeddings...")
    val_emb = model.precompute_embeddings(val_seqs, embed_batch_size, device)

    # Free ESMC from GPU
    model.embed_model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    head = model.w.to(device)
    train_loader = DataLoader(
        TensorDataset(train_emb, train_labels),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        head.train()
        for emb_b, label_b in train_loader:
            pred = head(emb_b.to(device))
            loss = F.mse_loss(pred, label_b.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 20 == 0 or epoch == 1:
            head.eval()
            with torch.no_grad():
                val_pred = head(val_emb.to(device))
                val_loss = F.mse_loss(val_pred, val_labels.to(device)).item()
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in head.state_dict().items()}
            print(
                f"    Epoch {epoch:3d} | Val MSE: {val_loss:.4f} (best: {best_loss:.4f})"
            )

    assert best_state is not None
    head.load_state_dict(best_state)
    head.eval()

    # Get predictions
    with torch.no_grad():
        train_pred = head(train_emb.to(device)).cpu()
        val_pred = head(val_emb.to(device)).cpu()

    return {"train_pred": train_pred, "val_pred": val_pred}


def train_ohe_mlp(
    model: PbrROneHotMLP,
    train_tokens: torch.LongTensor,
    train_labels: torch.FloatTensor,
    val_tokens: torch.LongTensor,
    val_labels: torch.FloatTensor,
    device: torch.device,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> dict:
    """Train OHE MLP on tokenized sequences."""
    model.to(device)

    # Precompute OHE
    V = model.vocab_size
    train_ohe = F.one_hot(train_tokens, V).float().reshape(len(train_tokens), -1)
    val_ohe = F.one_hot(val_tokens, V).float().reshape(len(val_tokens), -1)

    train_loader = DataLoader(
        TensorDataset(train_ohe, train_labels),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        for ohe_b, label_b in train_loader:
            # OHE MLP forward expects (B, P, T) not flattened
            ohe_3d = ohe_b.reshape(ohe_b.size(0), -1, V).to(device)
            pred = model.forward(ohe_3d)
            loss = F.mse_loss(pred, label_b.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 30 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_ohe_3d = val_ohe.reshape(val_ohe.size(0), -1, V).to(device)
                val_pred = model.forward(val_ohe_3d)
                val_loss = F.mse_loss(val_pred, val_labels.to(device)).item()
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(
                f"    Epoch {epoch:3d} | Val MSE: {val_loss:.4f} (best: {best_loss:.4f})"
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        train_ohe_3d = train_ohe.reshape(train_ohe.size(0), -1, V).to(device)
        train_pred = model.forward(train_ohe_3d).cpu()
        val_ohe_3d = val_ohe.reshape(val_ohe.size(0), -1, V).to(device)
        val_pred = model.forward(val_ohe_3d).cpu()

    return {"train_pred": train_pred, "val_pred": val_pred}


def train_xgboost(
    model: PbrRXGBoost,
    train_tokens: torch.LongTensor,
    train_labels: torch.FloatTensor,
    val_tokens: torch.LongTensor,
    val_labels: torch.FloatTensor,
) -> dict:
    """Train XGBoost on flattened OHE features."""
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

    V = EsmSequenceTokenizer().vocab_size

    train_ohe = F.one_hot(train_tokens, V).float().reshape(len(train_tokens), -1)
    val_ohe = F.one_hot(val_tokens, V).float().reshape(len(val_tokens), -1)

    info = model.fit(train_ohe, train_labels, eval_ohe=val_ohe, eval_labels=val_labels)
    print(f"    XGBoost training info: {info}")

    # Get predictions
    with torch.no_grad():
        train_ohe_3d = train_ohe.reshape(train_ohe.size(0), -1, V)
        train_pred = model.forward(train_ohe_3d).cpu()
        val_ohe_3d = val_ohe.reshape(val_ohe.size(0), -1, V)
        val_pred = model.forward(val_ohe_3d).cpu()

    return {"train_pred": train_pred, "val_pred": val_pred}


# ── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_predictions(
    pred: torch.FloatTensor,
    true: torch.FloatTensor,
    label: str,
) -> dict:
    """Compute and print RMSE, R², and Spearman ρ for Pb and Zn."""
    results = {}
    for i, metal in enumerate(["Pb", "Zn"]):
        p = pred[:, i].numpy()
        t = true[:, i].numpy()
        rmse = np.sqrt(mean_squared_error(t, p))
        r2 = r2_score(t, p)
        rho, _ = spearmanr(t, p)
        results[f"{metal}_rmse"] = rmse
        results[f"{metal}_r2"] = r2
        results[f"{metal}_rho"] = rho
        print(f"    {label} {metal}: RMSE={rmse:.4f}, R²={r2:.4f}, ρ={rho:.4f}")
    return results


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_comparison(all_results: dict, output_dir: Path):
    """Plot comparison bar charts and scatter plots for all models."""
    output_dir.mkdir(parents=True, exist_ok=True)

    models = list(all_results.keys())
    metrics = ["Pb_rho", "Zn_rho", "Pb_r2", "Zn_r2", "Pb_rmse", "Zn_rmse"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        vals = [all_results[m]["test_metrics"][metric] for m in models]
        bars = ax.bar(models, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ax.set_ylabel(metric.split("_")[1].upper())
        ax.set_title(f"{metric.split('_')[0]} {metric.split('_')[1].upper()}")
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Predictive Model Comparison on PbrR Test Set", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "step2_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison to {output_dir / 'step2_model_comparison.png'}")

    # Scatter plots: predicted vs actual on test set
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 5 * len(models)))
    if len(models) == 1:
        axes = axes.reshape(1, -1)

    for i, model_name in enumerate(models):
        res = all_results[model_name]
        pred = res["test_pred"]
        true = res["test_true"]

        for j, metal in enumerate(["Pb", "Zn"]):
            ax = axes[i, j]
            ax.scatter(true[:, j].numpy(), pred[:, j].numpy(), alpha=0.6, s=30)
            lims = [
                min(true[:, j].min(), pred[:, j].min()) - 0.1,
                max(true[:, j].max(), pred[:, j].max()) + 0.1,
            ]
            ax.plot(lims, lims, "k--", alpha=0.5, label="y=x")
            ax.set_xlabel(f"Actual {metal} log-FC")
            ax.set_ylabel(f"Predicted {metal} log-FC")
            rho = res["test_metrics"][f"{metal}_rho"]
            ax.set_title(f"{model_name} — {metal} (ρ={rho:.3f})")
            ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "step2_scatter_plots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter plots to {output_dir / 'step2_scatter_plots.png'}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--esmc-checkpoint", default="esmc_300m")
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--mlp-epochs", type=int, default=300)
    parser.add_argument("--mlp-dim", type=int, default=256)
    parser.add_argument("--mlp-layers", type=int, default=2)
    args = parser.parse_args()
    device = torch.device(args.device)

    # ── Load data ────────────────────────────────────────────────────────
    print("Loading PbrR round 1 data...")
    data = load_round1_data()
    splits = load_splits()

    # Prepare labels as log fold change (matching original paper)
    Y_raw = torch.tensor(
        np.stack([data["pb_fc"], data["zn_fc"]], axis=1), dtype=torch.float32
    )
    Y_log_fc = torch.log(Y_raw)

    train_idx = splits["train"]
    test_idx = np.array(sorted(set(splits["test"].tolist()) | set(splits["hard"])))

    train_tokens = data["tokenized"][train_idx]
    test_tokens = data["tokenized"][test_idx]
    train_labels = Y_log_fc[train_idx]
    test_labels = Y_log_fc[test_idx]
    train_seqs = [data["sequences"][i] for i in train_idx]
    test_seqs = [data["sequences"][i] for i in test_idx]

    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    print(
        f"Train labels — Pb: [{train_labels[:, 0].min():.2f}, {train_labels[:, 0].max():.2f}], "
        f"Zn: [{train_labels[:, 1].min():.2f}, {train_labels[:, 1].max():.2f}]"
    )

    all_results = {}

    # ── 1. Linear Probe ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Training Linear Probe on ESMC embeddings...")
    print("=" * 60)

    probe = PbrRLinearProbe(args.esmc_checkpoint)
    probe_preds = train_linear_probe(
        probe,
        train_seqs,
        train_labels,
        test_seqs,
        test_labels,
        device=device,
        epochs=args.probe_epochs,
    )

    print("  Test set metrics:")
    probe_metrics = evaluate_predictions(
        probe_preds["val_pred"], test_labels, "LinearProbe"
    )
    all_results["LinearProbe"] = {
        "test_pred": probe_preds["val_pred"],
        "test_true": test_labels,
        "test_metrics": probe_metrics,
    }

    # Free probe memory
    del probe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── 2. OHE MLP ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Training OHE-encoded MLP...")
    print("=" * 60)

    seq_len = data["tokenized"].shape[1]
    mlp = PbrROneHotMLP(
        sequence_length=seq_len,
        model_dim=args.mlp_dim,
        n_layers=args.mlp_layers,
    )
    mlp_preds = train_ohe_mlp(
        mlp,
        train_tokens,
        train_labels,
        test_tokens,
        test_labels,
        device=device,
        epochs=args.mlp_epochs,
    )

    print("  Test set metrics:")
    mlp_metrics = evaluate_predictions(mlp_preds["val_pred"], test_labels, "OneHotMLP")
    all_results["OneHotMLP"] = {
        "test_pred": mlp_preds["val_pred"],
        "test_true": test_labels,
        "test_metrics": mlp_metrics,
    }

    del mlp
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── 3. XGBoost ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Training XGBoost regressor...")
    print("=" * 60)

    xgb_model = PbrRXGBoost(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    xgb_preds = train_xgboost(
        xgb_model,
        train_tokens,
        train_labels,
        test_tokens,
        test_labels,
    )

    print("  Test set metrics:")
    xgb_metrics = evaluate_predictions(xgb_preds["val_pred"], test_labels, "XGBoost")
    all_results["XGBoost"] = {
        "test_pred": xgb_preds["val_pred"],
        "test_true": test_labels,
        "test_metrics": xgb_metrics,
    }

    # ── Comparison ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON SUMMARY (Test Set)")
    print("=" * 60)
    print(
        f"{'Model':>15s} | {'Pb ρ':>8s} | {'Zn ρ':>8s} | {'Pb R²':>8s} | {'Zn R²':>8s} | {'Pb RMSE':>8s} | {'Zn RMSE':>8s}"
    )
    print("-" * 80)
    for name, res in all_results.items():
        m = res["test_metrics"]
        print(
            f"{name:>15s} | {m['Pb_rho']:>8.4f} | {m['Zn_rho']:>8.4f} | "
            f"{m['Pb_r2']:>8.4f} | {m['Zn_r2']:>8.4f} | {m['Pb_rmse']:>8.4f} | {m['Zn_rmse']:>8.4f}"
        )

    # Find best model
    avg_rho = {
        name: (res["test_metrics"]["Pb_rho"] + res["test_metrics"]["Zn_rho"]) / 2
        for name, res in all_results.items()
    }
    best_model = max(avg_rho, key=avg_rho.get)
    print(
        f"\nBest model (by avg Spearman ρ): {best_model} (ρ_avg = {avg_rho[best_model]:.4f})"
    )

    # Save results
    results_path = CHECKPOINTS_FOLDER / "step2_results.pt"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            name: {"test_metrics": res["test_metrics"]}
            for name, res in all_results.items()
        },
        results_path,
    )
    print(f"Saved results to {results_path}")

    # Plot
    plot_comparison(all_results, OUTPUTS_FOLDER)


if __name__ == "__main__":
    main()
