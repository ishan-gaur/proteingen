"""Train a linear probe on ESMC embeddings to predict TrpB fitness.

Uses frozen ESMC-300m embeddings (mean-pooled over sequence positions) with
a single linear layer to predict continuous fitness values from the
SaProtHub/Dataset-TrpB_fitness_landsacpe dataset (160k protein sequences,
length 397, real-valued fitness labels).

Pre-computes embeddings once, then trains the linear layer with MSE loss.

Usage:
    uv run python examples/trpb_linear_probe.py
    uv run python examples/trpb_linear_probe.py --device cuda --batch-size 64
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from huggingface_hub import hf_hub_download
from dfm.models import ESMCEmbedding
from dfm.predictive_modeling import LinearProbe
import pandas as pd


DATASET_ID = "SaProtHub/Dataset-TrpB_fitness_landsacpe"


# ── 1. Data loading ─────────────────────────────────────────────────────────


def load_trpb_data() -> pd.DataFrame:
    """Download and load the TrpB fitness landscape dataset."""
    path = hf_hub_download(DATASET_ID, "dataset.csv", repo_type="dataset")
    return pd.read_csv(path)


# ── 2. Embedding computation ────────────────────────────────────────────────


@torch.no_grad()
def compute_embeddings(
    sequences: list[str],
    embed_model: ESMCEmbedding,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute mean-pooled ESMC embeddings for all sequences.

    Args:
        sequences: Raw amino acid strings.
        embed_model: Frozen ESMCEmbedding model.
        batch_size: Sequences per forward pass.
        device: Device to run inference on.

    Returns:
        Embeddings tensor, shape (N, 960).
    """
    embed_model = embed_model.to(device)
    all_embeddings = []
    n = len(sequences)
    for start in range(0, n, batch_size):
        batch_seqs = sequences[start : start + batch_size]
        token_ids = embed_model.tokenize(batch_seqs).to(device)
        embeddings = embed_model(token_ids)  # (B, 960)
        all_embeddings.append(embeddings.cpu())
        if (start // batch_size) % 50 == 0:
            print(f"  Embedded {min(start + batch_size, n):>6d} / {n}")
    return torch.cat(all_embeddings, dim=0)


# ── 3. Training loop ────────────────────────────────────────────────────────


def train_epoch(
    probe: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch, return mean MSE loss."""
    probe.train()
    # Only the linear layer is trainable; embed_model is frozen
    total_loss = 0.0
    n_batches = 0
    for emb_batch, label_batch in loader:
        emb_batch = emb_batch.to(device)
        label_batch = label_batch.to(device)

        pred = probe.w(emb_batch).squeeze(-1)  # (B,)
        loss = nn.functional.mse_loss(pred, label_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate(
    probe: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Compute MSE and Spearman correlation on a dataset."""
    probe.eval()
    all_preds = []
    all_labels = []
    for emb_batch, label_batch in loader:
        emb_batch = emb_batch.to(device)
        pred = probe.w(emb_batch).squeeze(-1)
        all_preds.append(pred.cpu())
        all_labels.append(label_batch)

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    mse = nn.functional.mse_loss(preds, labels).item()

    # Spearman rank correlation
    from scipy.stats import spearmanr

    rho, pval = spearmanr(preds.numpy(), labels.numpy())
    return {"mse": mse, "spearman_rho": rho, "spearman_pval": pval}


# ── 4. Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="TrpB fitness linear probe")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    device = torch.device(args.device)

    # ── Load data ────────────────────────────────────────────────────────
    print("Loading TrpB fitness dataset...")
    df = load_trpb_data()
    train_df = df[df["stage"] == "train"].reset_index(drop=True)
    valid_df = df[df["stage"] == "valid"].reset_index(drop=True)
    test_df = df[df["stage"] == "test"].reset_index(drop=True)
    print(f"  Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    # ── Compute embeddings (once) ────────────────────────────────────────
    print("Loading ESMC-300m embedding model...")
    embed_model = ESMCEmbedding("esmc_300m")
    print(f"  Embedding dim: {embed_model.EMB_DIM}")

    print("Computing train embeddings...")
    train_emb = compute_embeddings(
        train_df["protein"].tolist(), embed_model, args.embed_batch_size, device
    )
    print("Computing valid embeddings...")
    valid_emb = compute_embeddings(
        valid_df["protein"].tolist(), embed_model, args.embed_batch_size, device
    )
    print("Computing test embeddings...")
    test_emb = compute_embeddings(
        test_df["protein"].tolist(), embed_model, args.embed_batch_size, device
    )

    # Move embed_model off device to free memory
    embed_model = embed_model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_labels = torch.tensor(train_df["label"].values, dtype=torch.float32)
    valid_labels = torch.tensor(valid_df["label"].values, dtype=torch.float32)
    test_labels = torch.tensor(test_df["label"].values, dtype=torch.float32)

    # ── Build dataloaders from pre-computed embeddings ───────────────────
    train_loader = DataLoader(
        TensorDataset(train_emb, train_labels),
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        TensorDataset(valid_emb, valid_labels),
        batch_size=args.train_batch_size,
    )
    test_loader = DataLoader(
        TensorDataset(test_emb, test_labels),
        batch_size=args.train_batch_size,
    )

    # ── Build linear probe ───────────────────────────────────────────────
    probe = LinearProbe(embed_model=embed_model, output_dim=1).to(device)
    n_params = sum(p.numel() for p in probe.w.parameters())
    print(f"Linear probe: {n_params} trainable parameters")

    optimizer = torch.optim.AdamW(
        probe.w.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ── Train ────────────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs...")
    best_valid_rho = -float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(probe, train_loader, optimizer, device)
        valid_metrics = evaluate(probe, valid_loader, device)

        if valid_metrics["spearman_rho"] > best_valid_rho:
            best_valid_rho = valid_metrics["spearman_rho"]
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in probe.w.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d} | "
                f"Train MSE: {train_loss:.6f} | "
                f"Valid MSE: {valid_metrics['mse']:.6f} | "
                f"Valid ρ: {valid_metrics['spearman_rho']:.4f}"
            )

    # ── Evaluate best model on test set ──────────────────────────────────
    print(f"\nBest validation ρ = {best_valid_rho:.4f} at epoch {best_epoch}")
    assert best_state is not None
    probe.w.load_state_dict(best_state)
    test_metrics = evaluate(probe, test_loader, device)
    print(
        f"Test MSE: {test_metrics['mse']:.6f} | "
        f"Test ρ: {test_metrics['spearman_rho']:.4f} | "
        f"Test p-value: {test_metrics['spearman_pval']:.2e}"
    )


# ── 5. Guidance integration (sketch) ────────────────────────────────────
#
# After training, the probe can be wrapped as a PredictiveModel for use
# with TAG/DEG guided sampling. Since LinearProbe outputs a point estimate
# (no uncertainty), a RealValuedPredictiveModel subclass would define how
# to convert predictions + threshold into binary logits.
#
# Example (requires a RealValuedPredictiveModel subclass):
#
#   from dfm import RealValuedPredictiveModel
#   from dfm.guide import DEG
#   from dfm.models.esm import ESMC as ESMCTransition
#
#   class FitnessPredictor(RealValuedPredictiveModel):
#       def __init__(self, probe, tokenizer):
#           super().__init__(model=probe, tokenizer=tokenizer)
#           self.input_dim = tokenizer.vocab_size
#
#       def forward(self, ohe_seq_SPT, **kwargs):
#           seq_SP = ohe_seq_SPT.argmax(dim=-1)
#           return self.model(seq_SP)
#
#       def format_raw_to_logits(self, raw_output, ohe_seq_SPT, **kwargs):
#           # Convert point estimate + threshold to binary logits
#           pred_B = raw_output.reshape(-1)
#           # ... model-specific uncertainty → CDF conversion ...
#
#   predictor = FitnessPredictor(probe, esm_tokenizer)
#   gen_model = ESMCTransition("esmc_300m")
#
#   with predictor.target(threshold=0.5):
#       guided = DEG(gen_model, predictor)
#       seqs = sample_any_order_ancestral(guided, masked_seqs)


if __name__ == "__main__":
    main()
