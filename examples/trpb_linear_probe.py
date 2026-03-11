"""Train a linear probe on ESMC embeddings to predict TrpB fitness.

Uses frozen ESMC-300m embeddings (mean-pooled over sequence positions) with
a single linear layer to predict continuous fitness values from the
SaProtHub/Dataset-TrpB_fitness_landsacpe dataset (160k protein sequences,
length 397, real-valued fitness labels).

Pre-computes embeddings once (cached to disk), then trains the linear layer
with MSE loss.

Usage:
    uv run python examples/trpb_linear_probe.py
    uv run python examples/trpb_linear_probe.py --device cuda --embed-batch-size 64
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr
from dfm.models.esm import ESMC
from dfm.predictive_modeling import (
    PointEstimatePredictiveModel,
    LinearProbe,
)
import pandas as pd


DATASET_ID = "SaProtHub/Dataset-TrpB_fitness_landsacpe"
CACHE_DIR = Path("data/trpb_embeddings")


# ── TrpB fitness predictor ──────────────────────────────────────────────────


class TrpBFitnessPredictor(PointEstimatePredictiveModel):
    """Predicts TrpB fitness from sequence using ESMC embeddings + linear probe.

    At inference / guidance time, the full pipeline runs:
        ohe_seq → ESMC (differentiable embedding) → pool → linear → scalar

    For training, pre-compute embeddings via ``self.probe.compute_embeddings()``
    and train ``self.probe.w`` directly on those cached embeddings.
    """

    def __init__(self, esmc_checkpoint: str = "esmc_300m"):
        esmc = ESMC(esmc_checkpoint)
        super().__init__(tokenizer=esmc.tokenizer)
        special_ids = set(esmc.tokenizer.all_special_ids)

        def masked_mean_pool(emb_SPD, seq_SP):
            mask_SP = torch.ones_like(seq_SP, dtype=torch.bool)
            for sid in special_ids:
                mask_SP = mask_SP & (seq_SP != sid)
            mask_SP1 = mask_SP.unsqueeze(-1).float()
            return (emb_SPD * mask_SP1).sum(dim=1) / mask_SP1.sum(dim=1).clamp(min=1)

        self.probe = LinearProbe(embed_model=esmc, output_dim=1, pooling_fn=masked_mean_pool)
        self.input_dim = esmc.OUTPUT_DIM


# ── Data loading ─────────────────────────────────────────────────────────────


def load_trpb_data() -> pd.DataFrame:
    path = hf_hub_download(DATASET_ID, "dataset.csv", repo_type="dataset")
    return pd.read_csv(path)


def get_or_compute_embeddings(
    split_name: str,
    sequences: list[str],
    probe: LinearProbe,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    cache_path = CACHE_DIR / f"{split_name}.pt"
    if cache_path.exists():
        print(f"  Loading cached {split_name} embeddings from {cache_path}")
        return torch.load(cache_path, weights_only=True)

    embeddings = probe.compute_embeddings(sequences, batch_size, device)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, cache_path)
    print(f"  Cached to {cache_path}")
    return embeddings


# ── Training ─────────────────────────────────────────────────────────────────


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

    # ── Build predictor & compute embeddings ─────────────────────────────
    print("Building TrpB fitness predictor...")
    predictor = TrpBFitnessPredictor("esmc_300m")

    print("Computing embeddings...")
    train_emb = get_or_compute_embeddings(
        "train", train_df["protein"].tolist(), predictor.probe, args.embed_batch_size, device
    )
    valid_emb = get_or_compute_embeddings(
        "valid", valid_df["protein"].tolist(), predictor.probe, args.embed_batch_size, device
    )
    test_emb = get_or_compute_embeddings(
        "test", test_df["protein"].tolist(), predictor.probe, args.embed_batch_size, device
    )

    # Free ESMC GPU memory
    predictor.probe.embed_model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_labels = torch.tensor(train_df["label"].values, dtype=torch.float32)
    valid_labels = torch.tensor(valid_df["label"].values, dtype=torch.float32)
    test_labels = torch.tensor(test_df["label"].values, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_emb, train_labels),
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        TensorDataset(valid_emb, valid_labels), batch_size=args.train_batch_size
    )
    test_loader = DataLoader(
        TensorDataset(test_emb, test_labels), batch_size=args.train_batch_size
    )

    # ── Train the linear layer ───────────────────────────────────────────
    linear = predictor.probe.w.to(device)
    print(f"Linear probe: {sum(p.numel() for p in linear.parameters())} params")

    optimizer = torch.optim.AdamW(
        linear.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print(f"\nTraining for {args.epochs} epochs...")
    best_valid_rho = -float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        linear.train()
        total_loss = 0.0
        n_batches = 0
        for emb_batch, label_batch in train_loader:
            emb_batch, label_batch = emb_batch.to(device), label_batch.to(device)
            pred = linear(emb_batch).squeeze(-1)
            loss = F.mse_loss(pred, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / n_batches

        linear.eval()
        with torch.no_grad():
            preds = torch.cat(
                [linear(e.to(device)).squeeze(-1).cpu() for e, _ in valid_loader]
            )
            labels = torch.cat([l for _, l in valid_loader])
            valid_mse = F.mse_loss(preds, labels).item()
            valid_rho, _ = spearmanr(preds.numpy(), labels.numpy())

        if valid_rho > best_valid_rho:
            best_valid_rho = valid_rho
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in linear.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d} | "
                f"Train MSE: {train_loss:.6f} | "
                f"Valid MSE: {valid_mse:.6f} | "
                f"Valid ρ: {valid_rho:.4f}"
            )

    # ── Evaluate best model on test set ──────────────────────────────────
    print(f"\nBest validation ρ = {best_valid_rho:.4f} at epoch {best_epoch}")
    assert best_state is not None
    linear.load_state_dict(best_state)
    linear.eval()
    with torch.no_grad():
        preds = torch.cat(
            [linear(e.to(device)).squeeze(-1).cpu() for e, _ in test_loader]
        )
        labels = torch.cat([l for _, l in test_loader])
        test_mse = F.mse_loss(preds, labels).item()
        test_rho, test_pval = spearmanr(preds.numpy(), labels.numpy())

    print(
        f"Test MSE: {test_mse:.6f} | "
        f"Test ρ: {test_rho:.4f} | "
        f"Test p-value: {test_pval:.2e}"
    )

    # ── Guidance usage ───────────────────────────────────────────────────
    # After training, the predictor is ready for guided sampling:
    #
    #   from dfm.guide import DEG
    #   gen_model = ESMC("esmc_300m")  # as TransitionModel
    #   with predictor.with_target(0.5):  # threshold
    #       guided = DEG(gen_model, predictor)
    #       seqs = sample_any_order_ancestral(guided, masked_seqs)


if __name__ == "__main__":
    main()
