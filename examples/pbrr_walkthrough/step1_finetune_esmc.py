"""Step 1: LoRA-finetune ESMC on PbrR Pareto front variants.

Finetunes ESMC-300M using LoRA adapters on the "successful" training variants
(those on the Pareto front: high Pb sensitivity, low Zn response). Then plots
likelihood curves comparing the pretrained and finetuned models on the remaining
(non-training) data, visualized as Pb vs Zn affinity scatter plots colored by
model likelihood.

Usage:
    uv run python examples/pbrr_walkthrough/step1_finetune_esmc.py [--epochs 100] [--device cuda]
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from proteingen.models.esm import ESMC

matplotlib.use("Agg")

# Allow running as `uv run python examples/pbrr_walkthrough/step1_finetune_esmc.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import (
    CHECKPOINTS_FOLDER,
    OUTPUTS_FOLDER,
    load_round1_data,
    load_splits,
)


# ── Noisy sequence dataset ───────────────────────────────────────────────────


class NoisySequenceDataset(Dataset):
    """Dataset that randomly masks tokens for MLM training."""

    def __init__(self, tokenized_sequences: torch.LongTensor, mask_token_id: int):
        self.sequences = tokenized_sequences
        self.mask_token_id = mask_token_id

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        t = random.random()
        mask = torch.rand(seq.shape) < t
        mask[0] = False  # don't mask BOS
        mask[-1] = False  # don't mask EOS
        noisy = seq.clone()
        noisy[mask] = self.mask_token_id
        return noisy, seq


def collate_fn(batch):
    noisy, true = zip(*batch)
    return torch.stack(noisy), torch.stack(true)


# ── Likelihood computation ───────────────────────────────────────────────────


@torch.no_grad()
def compute_mean_log_likelihood(
    model: ESMC,
    tokenized_sequences: torch.LongTensor,
    batch_size: int = 16,
    n_mask_samples: int = 10,
) -> torch.FloatTensor:
    """Compute mean log-likelihood for each sequence via masked pseudo-likelihood.

    For each sequence, randomly masks ~50% of positions n_mask_samples times
    and averages the log p(true token | masked context) at masked positions.

    Returns (N,) tensor of mean log-likelihoods.
    """
    device = model.device
    N = len(tokenized_sequences)
    all_ll = torch.zeros(N)
    mask_token_id = model.tokenizer.mask_token_id

    for sample_idx in range(n_mask_samples):
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = tokenized_sequences[start:end].clone()
            true_tokens = batch.clone()

            # Random mask ~50% of non-special positions
            mask = torch.rand(batch.shape) < 0.5
            mask[:, 0] = False  # BOS
            mask[:, -1] = False  # EOS
            batch[mask] = mask_token_id

            log_probs = model.get_log_probs(batch.to(device))  # (B, L, V)
            T = log_probs.shape[-1]
            true_clamped = true_tokens.to(device).clamp(max=T - 1)
            true_lp = log_probs.gather(2, true_clamped.unsqueeze(2)).squeeze(
                2
            )  # (B, L)

            # Average log-prob at masked positions only
            mask_d = mask.to(device)
            n_masked = mask_d.sum(dim=1).float().clamp(min=1)
            mean_lp = (true_lp * mask_d).sum(dim=1) / n_masked
            all_ll[start:end] += mean_lp.cpu()

    return all_ll / n_mask_samples


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_likelihood_scatter(
    pb_fc: np.ndarray,
    zn_fc: np.ndarray,
    ll_pretrained: np.ndarray,
    ll_finetuned: np.ndarray,
    output_dir: Path,
    train_success_indices: np.ndarray,
):
    """Plot Pb vs Zn FC colored by model likelihood for pretrained vs finetuned."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, ll, title in [
        (axes[0], ll_pretrained, "Pretrained ESMC"),
        (axes[1], ll_finetuned, "LoRA-Finetuned ESMC"),
    ]:
        log_pb = np.log(pb_fc)
        log_zn = np.log(zn_fc)

        sc = ax.scatter(
            log_pb,
            log_zn,
            c=ll,
            cmap="viridis",
            alpha=0.7,
            s=20,
            edgecolors="none",
        )

        # Mark training success samples
        ax.scatter(
            log_pb[train_success_indices],
            log_zn[train_success_indices],
            facecolors="none",
            edgecolors="red",
            s=60,
            linewidths=1.5,
            label="Training (Pareto front)",
        )

        ax.set_xlabel("log(Pb FC)")
        ax.set_ylabel("log(Zn FC)")
        ax.set_title(title)
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        plt.colorbar(sc, ax=ax, label="Mean log-likelihood")

    fig.suptitle("Model Likelihood on Test Data: Pb vs Zn Affinity")
    fig.tight_layout()
    fig.savefig(
        output_dir / "step1_likelihood_scatter.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"Saved likelihood scatter to {output_dir / 'step1_likelihood_scatter.png'}")

    # Also plot the difference
    fig, ax = plt.subplots(figsize=(8, 6))
    log_pb = np.log(pb_fc)
    log_zn = np.log(zn_fc)
    diff = ll_finetuned - ll_pretrained
    sc = ax.scatter(
        log_pb,
        log_zn,
        c=diff,
        cmap="RdBu_r",
        alpha=0.7,
        s=20,
        edgecolors="none",
        vmin=-np.abs(diff).max(),
        vmax=np.abs(diff).max(),
    )
    ax.scatter(
        log_pb[train_success_indices],
        log_zn[train_success_indices],
        facecolors="none",
        edgecolors="red",
        s=60,
        linewidths=1.5,
        label="Training (Pareto front)",
    )
    ax.set_xlabel("log(Pb FC)")
    ax.set_ylabel("log(Zn FC)")
    ax.set_title("Likelihood Difference (Finetuned - Pretrained)")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label="Δ log-likelihood")
    fig.tight_layout()
    fig.savefig(output_dir / "step1_likelihood_diff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved likelihood difference to {output_dir / 'step1_likelihood_diff.png'}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--n-mask-samples",
        type=int,
        default=10,
        help="Number of masking samples for likelihood estimation",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    # ── Load data ────────────────────────────────────────────────────────
    print("Loading PbrR round 1 data...")
    data = load_round1_data()
    splits = load_splits()

    train_success_seqs = data["tokenized"][splits["train_success"]]
    print(f"Training on {len(train_success_seqs)} Pareto front variants")
    print(f"Will evaluate on {len(data['sequences'])} total sequences")

    # ── Build and evaluate pretrained model ──────────────────────────────
    print("\nBuilding pretrained ESMC-300M...")
    pretrained_model = ESMC("esmc_300m")
    pretrained_model.to(device)
    pretrained_model.eval()

    print("Computing pretrained likelihoods...")
    ll_pretrained = compute_mean_log_likelihood(
        pretrained_model,
        data["tokenized"],
        batch_size=16,
        n_mask_samples=args.n_mask_samples,
    )
    print(
        f"  Pretrained mean LL: {ll_pretrained.mean():.4f} ± {ll_pretrained.std():.4f}"
    )

    # Free pretrained model memory before training
    pretrained_model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Finetune with LoRA ───────────────────────────────────────────────
    print(f"\nFinetuning ESMC with LoRA (rank={args.lora_rank})...")
    finetuned_model = ESMC("esmc_300m")
    finetuned_model.to(device)

    lora_targets = [
        "attn.layernorm_qkv.1",
        "attn.out_proj",
        "sequence_head.0",
        "sequence_head.3",
    ]
    finetuned_model.apply_lora(
        target_modules=lora_targets,
        r=args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    n_trainable = sum(
        p.numel() for p in finetuned_model.parameters() if p.requires_grad
    )
    n_total = sum(p.numel() for p in finetuned_model.parameters())
    print(
        f"  Trainable: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f}%)"
    )

    # Training dataset and loader
    mask_token_id = finetuned_model.tokenizer.mask_token_id
    train_dataset = NoisySequenceDataset(train_success_seqs, mask_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        [p for p in finetuned_model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    # Linear warmup scheduler
    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    finetuned_model.train()
    step_count = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for noisy_seqs, true_seqs in train_loader:
            noisy_seqs = noisy_seqs.to(device)
            true_seqs = true_seqs.to(device)

            log_probs = finetuned_model.get_log_probs(noisy_seqs)  # (B, L, V)
            T = log_probs.shape[-1]
            true_clamped = true_seqs.clamp(max=T - 1)
            target_lp = log_probs.gather(2, true_clamped.unsqueeze(2)).squeeze(2)

            # Loss on masked positions only
            mask = noisy_seqs != true_seqs
            n_masked = mask.sum(dim=1).float().clamp(min=1)
            loss = -(target_lp * mask).sum(dim=1) / n_masked
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                finetuned_model.parameters(), args.gradient_clip
            )
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            step_count += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{args.epochs} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    print(f"  Best training loss: {best_loss:.4f}")

    # Save LoRA checkpoint
    ckpt_dir = CHECKPOINTS_FOLDER / "step1_lora"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    finetuned_model.save_lora(ckpt_dir / "lora_adapter")
    print(f"  Saved LoRA adapter to {ckpt_dir / 'lora_adapter'}")

    # ── Evaluate finetuned model ─────────────────────────────────────────
    print("\nComputing finetuned likelihoods...")
    finetuned_model.eval()
    ll_finetuned = compute_mean_log_likelihood(
        finetuned_model,
        data["tokenized"],
        batch_size=16,
        n_mask_samples=args.n_mask_samples,
    )
    print(f"  Finetuned mean LL: {ll_finetuned.mean():.4f} ± {ll_finetuned.std():.4f}")

    # ── Plot results ─────────────────────────────────────────────────────
    # Plot on ALL samples (marking Pareto front training points)
    plot_likelihood_scatter(
        pb_fc=data["pb_fc"],
        zn_fc=data["zn_fc"],
        ll_pretrained=ll_pretrained.numpy(),
        ll_finetuned=ll_finetuned.numpy(),
        output_dir=OUTPUTS_FOLDER,
        train_success_indices=splits["train_success"],
    )

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Compare likelihoods in success vs non-success regions
    success_mask = np.log(data["zn_fc"]) < 3.5 * np.log(data["pb_fc"]) - 2.5
    non_success_mask = ~success_mask

    for name, mask in [
        ("Success region", success_mask),
        ("Non-success region", non_success_mask),
    ]:
        ll_pre = ll_pretrained[mask].mean().item()
        ll_ft = ll_finetuned[mask].mean().item()
        print(
            f"  {name}: Pretrained LL={ll_pre:.4f}, Finetuned LL={ll_ft:.4f}, Δ={ll_ft - ll_pre:+.4f}"
        )


if __name__ == "__main__":
    main()
