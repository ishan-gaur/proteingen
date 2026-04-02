"""Step 3: Oracle training, noisy classifier, and guided generation comparison.

This step:
1. Trains an oracle model on ALL PbrR data (round 1 + later rounds from supplementary)
   using the best predictive model architecture from Step 2 (expected: OHE MLP or XGBoost).
2. Trains a noisy classifier on the full round 1 dataset for DEG guidance.
3. Generates sequences using four methods and scores them with the oracle:
   - Pretrained unconditional ESMC
   - Finetuned unconditional ESMC (LoRA from Step 1)
   - DEG-guided pretrained ESMC
   - DEG-guided finetuned ESMC
4. Compares generations on Pb vs Zn oracle-scored scatter plots.

Only mutated positions (SSM positions from experimental evidence) are masked
and sampled, matching the original experiment's design.

Usage:
    uv run python examples/pbrr_walkthrough/step3_guided_generation.py [--device cuda]
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
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from proteingen.guide import DEG
from proteingen.models.esm import ESMC
from proteingen.models.xgboost_model import XGBoostPredictor
from proteingen.predictive_modeling import (
    OneHotMLP,
)
from proteingen.sampling import sample_any_order_ancestral

matplotlib.use("Agg")

# Allow running as `uv run python examples/pbrr_walkthrough/step3_guided_generation.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import (
    CHECKPOINTS_FOLDER,
    OUTPUTS_FOLDER,
    load_round1_data,
    load_all_rounds_data,
    load_splits,
    find_ssm_positions,
    make_masked_wt,
)


# ── Oracle model (OHE MLP trained on all data) ──────────────────────────────


class PbrROracleOHEMLP(OneHotMLP):
    """OHE MLP oracle trained on ALL PbrR data (all rounds).

    Used purely for evaluation — scores generated sequences to produce
    predicted Pb/Zn fold changes.
    """

    def __init__(
        self,
        sequence_length: int = 147,
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
            output_dim=2,
            dropout=dropout,
        )

    def format_raw_to_logits(self, raw_output, ohe_seq_SPT, **kwargs):
        raise NotImplementedError("Oracle is for scoring, not guidance")


class PbrROracleXGBoost(XGBoostPredictor):
    """XGBoost oracle trained on ALL PbrR data."""

    def __init__(self, **xgb_kwargs):
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

        tokenizer = EsmSequenceTokenizer()
        super().__init__(tokenizer=tokenizer, output_dim=2, **xgb_kwargs)

    def format_raw_to_logits(self, raw_output, ohe_seq_SPT, **kwargs):
        raise NotImplementedError("Oracle is for scoring, not guidance")


# ── Noisy classifier for DEG guidance ────────────────────────────────────────


class PbrRNoisyClassifier(OneHotMLP):
    """Noisy classifier for DEG guidance.

    Trained on round 1 data with random masking at train time so it can
    handle partially-masked sequences during generation. Predicts binary
    "success" (Pb FC > threshold AND Zn FC < threshold) from OHE input.

    format_raw_to_logits converts the 2-output regression to a joint
    binary logit via point_estimate_binary_logits.
    """

    def __init__(
        self,
        sequence_length: int = 147,
        model_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        pb_threshold: float = 2.0,
        zn_threshold: float = -0.6,
        k: float = 50.0,
    ):
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

        tokenizer = EsmSequenceTokenizer()
        super().__init__(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            model_dim=model_dim,
            n_layers=n_layers,
            output_dim=2,  # predict [Pb_logFC, Zn_logFC]
            dropout=dropout,
        )
        self.pb_threshold = pb_threshold
        self.zn_threshold = zn_threshold
        self.k = k

    def format_raw_to_logits(self, raw_output, ohe_seq_SPT, **kwargs):
        """Convert Pb/Zn predictions to joint binary logits.

        Success = Pb_logFC > pb_threshold AND Zn_logFC < zn_threshold.
        Uses steep sigmoid approximation for each condition, then combines.
        """
        pb_pred = raw_output[:, 0]
        zn_pred = raw_output[:, 1]

        # P(Pb > threshold) via sigmoid
        pb_logit = self.k * (pb_pred - self.pb_threshold)
        # P(Zn < threshold) via sigmoid (note: lower is better, so negate)
        zn_logit = self.k * (self.zn_threshold - zn_pred)

        # Joint: both conditions must hold. In logit space, add them
        # (approximates AND under the assumption of independent conditions)
        joint_logit = pb_logit + zn_logit

        zero = torch.zeros_like(joint_logit)
        return torch.stack([zero, joint_logit], dim=-1)


# ── Training functions ───────────────────────────────────────────────────────


def train_oracle_mlp(
    all_tokens: torch.LongTensor,
    all_labels: torch.FloatTensor,
    device: torch.device,
    seq_length: int,
    epochs: int = 500,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> PbrROracleOHEMLP:
    """Train OHE MLP oracle on all available data."""
    model = PbrROracleOHEMLP(sequence_length=seq_length)
    model.to(device)

    V = model.vocab_size
    all_ohe = F.one_hot(all_tokens, V).float()

    loader = DataLoader(
        TensorDataset(all_ohe, all_labels),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for ohe_b, label_b in loader:
            ohe_b = ohe_b.to(device)
            pred = model.forward(ohe_b)
            loss = F.mse_loss(pred, label_b.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0 or epoch == 1:
            print(
                f"    Oracle epoch {epoch:3d} | Loss: {avg_loss:.4f} (best: {best_loss:.4f})"
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model


def train_oracle_xgboost(
    all_tokens: torch.LongTensor,
    all_labels: torch.FloatTensor,
) -> PbrROracleXGBoost:
    """Train XGBoost oracle on all available data."""
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

    V = EsmSequenceTokenizer().vocab_size

    model = PbrROracleXGBoost(n_estimators=1000, max_depth=6, learning_rate=0.05)
    all_ohe = F.one_hot(all_tokens, V).float().reshape(len(all_tokens), -1)
    model.fit(all_ohe, all_labels)
    return model


def train_noisy_classifier(
    train_tokens: torch.LongTensor,
    train_labels: torch.FloatTensor,
    device: torch.device,
    seq_length: int,
    mask_token_id: int,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    pb_threshold: float = 2.0,
    zn_threshold: float = -0.6,
) -> PbrRNoisyClassifier:
    """Train noisy classifier for DEG guidance.

    At each training step, randomly masks a fraction of non-special positions
    to make the classifier robust to partial sequences seen during generation.
    """
    model = PbrRNoisyClassifier(
        sequence_length=seq_length,
        pb_threshold=pb_threshold,
        zn_threshold=zn_threshold,
    )
    model.to(device)

    V = model.vocab_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0

        # Shuffle
        perm = torch.randperm(len(train_tokens))
        for start in range(0, len(train_tokens), batch_size):
            end = min(start + batch_size, len(train_tokens))
            idx = perm[start:end]
            tokens = train_tokens[idx].clone()
            labels = train_labels[idx]

            # Random masking (uniform t in [0, 1])
            t = torch.rand(1).item()
            mask = torch.rand(tokens.shape) < t
            mask[:, 0] = False  # BOS
            mask[:, -1] = False  # EOS
            tokens[mask] = mask_token_id

            ohe = F.one_hot(tokens, V).float().to(device)
            pred = model.forward(ohe)
            loss = F.mse_loss(pred, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 30 == 0 or epoch == 1:
            print(
                f"    Classifier epoch {epoch:3d} | Loss: {avg_loss:.6f} (best: {best_loss:.6f})"
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model


# ── Generation ───────────────────────────────────────────────────────────────


def generate_sequences(
    model,
    masked_wt: torch.LongTensor,
    n_sequences: int,
    batch_size: int = 10,
    label: str = "",
) -> torch.LongTensor:
    """Generate sequences by iterative unmasking from masked WT."""
    all_seqs = []

    for start in tqdm(range(0, n_sequences, batch_size), desc=f"Generating {label}"):
        n_batch = min(batch_size, n_sequences - start)
        batch = masked_wt.unsqueeze(0).repeat(n_batch, 1).to(model.device)

        generated = sample_any_order_ancestral(
            model,
            batch,
            n_parallel=1,
            return_string=False,
        )
        all_seqs.append(generated.cpu())

    return torch.cat(all_seqs, dim=0)


# ── Oracle scoring ───────────────────────────────────────────────────────────


def score_with_oracle(
    oracle,
    sequences: torch.LongTensor,
    batch_size: int = 64,
) -> torch.FloatTensor:
    """Score sequences with oracle model, returning predicted log-FC."""
    V = oracle.tokenizer.vocab_size if hasattr(oracle, "tokenizer") else 33
    all_preds = []

    for start in range(0, len(sequences), batch_size):
        end = min(start + batch_size, len(sequences))
        batch = sequences[start:end]
        ohe = F.one_hot(batch, V).float()

        if hasattr(oracle, "vocab_size"):
            # OHE MLP
            pred = oracle.forward(ohe.to(next(oracle.parameters()).device))
        else:
            pred = oracle.forward(ohe)
        all_preds.append(pred.detach().cpu())

    return torch.cat(all_preds, dim=0)


# ── Plotting ─────────────────────────────────────────────────────────────────


COLOR_DATASET = "#9E9E9E"
COLOR_UNCONDITIONAL = "#008080"
COLOR_SFT = "#7C4DFF"
COLOR_GUIDED = "#FF69B4"
COLOR_GUIDED_SFT = "#D13100"
COLOR_TARGET = "#FFE74D"


def plot_generation_comparison(
    scores: dict[str, torch.FloatTensor],
    dataset_labels: torch.FloatTensor,
    output_dir: Path,
    pb_threshold: float = 2.0,
    zn_threshold: float = -0.6,
):
    """Plot Pb vs Zn oracle scores for all generation methods."""
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "Pretrained Unconditional": COLOR_UNCONDITIONAL,
        "Finetuned Unconditional": COLOR_SFT,
        "Guided Pretrained": COLOR_GUIDED,
        "Guided Finetuned": COLOR_GUIDED_SFT,
    }

    # Individual plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for idx, (name, Y) in enumerate(scores.items()):
        ax = axes[idx // 2, idx % 2]
        color = colors.get(name, "#333333")

        ax.scatter(
            dataset_labels[:, 0].numpy(),
            dataset_labels[:, 1].numpy(),
            alpha=0.15,
            color=COLOR_DATASET,
            s=10,
            label="Dataset",
            linewidths=0,
        )
        ax.scatter(
            Y[:, 0].numpy(),
            Y[:, 1].numpy(),
            alpha=0.6,
            color=color,
            s=20,
            edgecolors="white",
            linewidths=0.3,
            label=f"{name} (n={len(Y)})",
        )

        # Target region
        ax.axvline(
            pb_threshold, color=COLOR_TARGET, linestyle=":", linewidth=1, alpha=0.8
        )
        ax.axhline(
            zn_threshold, color=COLOR_TARGET, linestyle=":", linewidth=1, alpha=0.8
        )

        ax.set_xlabel("Pb log-FC (Higher → Better)")
        ax.set_ylabel("Zn log-FC (Lower → Better)")
        ax.invert_yaxis()
        ax.set_title(name)
        ax.legend(fontsize=8)

    fig.suptitle("Oracle-Scored Generated Sequences: Pb vs Zn Affinity", fontsize=14)
    fig.tight_layout()
    fig.savefig(
        output_dir / "step3_generation_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # Combined overlay plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        dataset_labels[:, 0].numpy(),
        dataset_labels[:, 1].numpy(),
        alpha=0.1,
        color=COLOR_DATASET,
        s=10,
        label="Dataset",
        linewidths=0,
    )
    for name, Y in scores.items():
        color = colors.get(name, "#333333")
        ax.scatter(
            Y[:, 0].numpy(),
            Y[:, 1].numpy(),
            alpha=0.5,
            color=color,
            s=20,
            edgecolors="white",
            linewidths=0.3,
            label=f"{name} (n={len(Y)})",
        )

    ax.axvline(pb_threshold, color=COLOR_TARGET, linestyle=":", linewidth=1, alpha=0.8)
    ax.axhline(zn_threshold, color=COLOR_TARGET, linestyle=":", linewidth=1, alpha=0.8)
    ax.set_xlabel("Pb log-FC (Higher → Better)")
    ax.set_ylabel("Zn log-FC (Lower → Better)")
    ax.invert_yaxis()
    ax.set_title("All Methods — Oracle-Scored Generations")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "step3_combined_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved generation comparison to {output_dir}")

    # Success rate bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(scores.keys())
    success_rates = []
    for name in names:
        Y = scores[name]
        success = (
            (Y[:, 0] >= pb_threshold) & (Y[:, 1] <= zn_threshold)
        ).float().mean().item() * 100
        success_rates.append(success)

    bars = ax.bar(
        names,
        success_rates,
        color=[colors.get(n, "#333") for n in names],
        edgecolor="white",
        linewidth=0.75,
    )
    for bar, rate in zip(bars, success_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("% in Target Region")
    ax.set_title(f"Success Rate (Pb logFC ≥ {pb_threshold}, Zn logFC ≤ {zn_threshold})")
    ax.set_ylim(0, max(success_rates) * 1.3 if max(success_rates) > 0 else 10)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "step3_success_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--oracle-type",
        default="xgboost",
        choices=["mlp", "xgboost"],
        help="Oracle model type (use best from Step 2)",
    )
    parser.add_argument("--oracle-epochs", type=int, default=500)
    parser.add_argument("--classifier-epochs", type=int, default=300)
    parser.add_argument(
        "--n-sequences",
        type=int,
        default=100,
        help="Number of sequences to generate per method",
    )
    parser.add_argument("--gen-batch-size", type=int, default=10)
    parser.add_argument(
        "--pb-threshold",
        type=float,
        default=2.0,
        help="Pb log-FC threshold (in log space, e.g. 2.0 = FC of ~7.4x)",
    )
    parser.add_argument(
        "--zn-threshold",
        type=float,
        default=-0.6,
        help="Zn log-FC threshold (in log space, e.g. -0.6 = FC of ~0.55x)",
    )
    parser.add_argument(
        "--guidance-temp",
        type=float,
        default=1.0,
        help="Predictor temperature for DEG (lower = stronger guidance)",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation, only train oracle and classifier",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    # Thresholds are already in log fold-change space
    # Pb threshold 2.0 → Pb FC ≥ e^2 ≈ 7.4x; Zn threshold -0.6 → Zn FC ≤ e^-0.6 ≈ 0.55x
    pb_threshold = args.pb_threshold
    zn_threshold = args.zn_threshold

    # ── Load data ────────────────────────────────────────────────────────
    print("Loading PbrR data...")
    round1_data = load_round1_data()
    load_splits()
    ssm_positions = find_ssm_positions(round1_data)
    masked_wt = make_masked_wt(round1_data, ssm_positions)
    n_masked = (masked_wt == 32).sum().item()
    print(f"  Round 1: {len(round1_data['sequences'])} sequences")
    print(f"  SSM positions: {len(ssm_positions)} ({n_masked} masked in WT)")

    print("\nLoading all-rounds data for oracle training...")
    all_data = load_all_rounds_data()
    print(f"  All rounds: {len(all_data['sequences'])} sequences")

    # Labels for oracle: log fold change
    all_labels = torch.log(
        torch.tensor(
            np.stack([all_data["pb_fc"], all_data["zn_fc"]], axis=1),
            dtype=torch.float32,
        )
    )
    round1_labels = torch.log(
        torch.tensor(
            np.stack([round1_data["pb_fc"], round1_data["zn_fc"]], axis=1),
            dtype=torch.float32,
        )
    )

    seq_length = round1_data["tokenized"].shape[1]
    mask_token_id = 32  # ESM mask token

    # ── Step 3a: Train oracle on all data ────────────────────────────────
    print(f"\n{'=' * 60}")
    print(
        f"Training {args.oracle_type.upper()} oracle on ALL {len(all_data['sequences'])} sequences..."
    )
    print("=" * 60)

    if args.oracle_type == "mlp":
        oracle = train_oracle_mlp(
            all_data["tokenized"],
            all_labels,
            device=device,
            seq_length=seq_length,
            epochs=args.oracle_epochs,
        )
    else:
        oracle = train_oracle_xgboost(all_data["tokenized"], all_labels)

    # Evaluate oracle on all data (should fit well since trained on all)
    oracle_preds = score_with_oracle(oracle, all_data["tokenized"])
    for i, metal in enumerate(["Pb", "Zn"]):
        rho, _ = spearmanr(all_labels[:, i].numpy(), oracle_preds[:, i].numpy())
        rmse = np.sqrt(
            mean_squared_error(all_labels[:, i].numpy(), oracle_preds[:, i].numpy())
        )
        print(f"  Oracle {metal} (all data): ρ={rho:.4f}, RMSE={rmse:.4f}")

    # Also check on round 1 data specifically
    r1_oracle_preds = score_with_oracle(oracle, round1_data["tokenized"])
    for i, metal in enumerate(["Pb", "Zn"]):
        rho, _ = spearmanr(round1_labels[:, i].numpy(), r1_oracle_preds[:, i].numpy())
        print(f"  Oracle {metal} (round 1): ρ={rho:.4f}")

    # ── Step 3b: Train noisy classifier on round 1 for guidance ──────────
    print(f"\n{'=' * 60}")
    print(
        f"Training noisy classifier on round 1 ({len(round1_data['sequences'])} sequences)..."
    )
    print("=" * 60)

    classifier = train_noisy_classifier(
        round1_data["tokenized"],
        round1_labels,
        device=device,
        seq_length=seq_length,
        mask_token_id=mask_token_id,
        epochs=args.classifier_epochs,
        pb_threshold=pb_threshold,
        zn_threshold=zn_threshold,
    )

    # Quick sanity check: classifier predictions on clean round 1 data
    classifier.eval()
    with torch.no_grad():
        V = classifier.vocab_size
        test_ohe = F.one_hot(round1_data["tokenized"][:10], V).float().to(device)
        test_pred = classifier.forward(test_ohe)
        print("  Sample classifier predictions (first 10):")
        print(f"    Pb: {test_pred[:5, 0].cpu().numpy()}")
        print(f"    Zn: {test_pred[:5, 1].cpu().numpy()}")

    # Save classifier checkpoint
    ckpt_path = CHECKPOINTS_FOLDER / "step3_noisy_classifier.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(classifier.state_dict(), ckpt_path)
    print(f"  Saved classifier to {ckpt_path}")

    if args.skip_generation:
        print("\nSkipping generation (--skip-generation flag set)")
        return

    # ── Step 3c: Generate with 4 methods ─────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Generating {args.n_sequences} sequences per method...")
    print("=" * 60)

    # Set up target for classifier
    classifier.set_target_(True)  # target = success event
    classifier.set_temp_(args.guidance_temp)

    all_scores = {}

    # 1. Pretrained unconditional ESMC
    print("\n[1/4] Pretrained unconditional ESMC...")
    pretrained = ESMC("esmc_300m")
    pretrained.to(device)
    pretrained.eval()

    uncond_seqs = generate_sequences(
        pretrained,
        masked_wt,
        args.n_sequences,
        args.gen_batch_size,
        label="Pretrained Unconditional",
    )
    uncond_scores = score_with_oracle(oracle, uncond_seqs)
    all_scores["Pretrained Unconditional"] = uncond_scores
    print(f"  Generated {len(uncond_seqs)} sequences")

    # 2. Finetuned unconditional ESMC (load LoRA from Step 1)
    print("\n[2/4] Finetuned unconditional ESMC...")
    lora_path = CHECKPOINTS_FOLDER / "step1_lora" / "lora_adapter"
    if lora_path.exists():
        finetuned = ESMC("esmc_300m")
        finetuned.load_lora(lora_path)
        finetuned.to(device)
        finetuned.eval()

        sft_seqs = generate_sequences(
            finetuned,
            masked_wt,
            args.n_sequences,
            args.gen_batch_size,
            label="Finetuned Unconditional",
        )
        sft_scores = score_with_oracle(oracle, sft_seqs)
        all_scores["Finetuned Unconditional"] = sft_scores
        print(f"  Generated {len(sft_seqs)} sequences")

        # Free finetuned model for guided generation
        finetuned.cpu()
        del finetuned
    else:
        print(f"  WARNING: LoRA adapter not found at {lora_path}. Run Step 1 first.")
        print("  Skipping finetuned unconditional.")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 3. DEG-guided pretrained ESMC
    print("\n[3/4] DEG-guided pretrained ESMC...")
    classifier.to(device)

    deg_pretrained = DEG(
        gen_model=pretrained,
        pred_model=classifier,
        argmax_masked_positions=True,  # fill masks with argmax for classifier
    )

    guided_seqs = generate_sequences(
        deg_pretrained,
        masked_wt,
        args.n_sequences,
        args.gen_batch_size,
        label="Guided Pretrained",
    )
    guided_scores = score_with_oracle(oracle, guided_seqs)
    all_scores["Guided Pretrained"] = guided_scores
    print(f"  Generated {len(guided_seqs)} sequences")

    # Free pretrained
    pretrained.cpu()
    del pretrained, deg_pretrained
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 4. DEG-guided finetuned ESMC
    print("\n[4/4] DEG-guided finetuned ESMC...")
    if lora_path.exists():
        finetuned = ESMC("esmc_300m")
        finetuned.load_lora(lora_path)
        finetuned.to(device)
        finetuned.eval()

        deg_finetuned = DEG(
            gen_model=finetuned,
            pred_model=classifier,
            argmax_masked_positions=True,
        )

        guided_sft_seqs = generate_sequences(
            deg_finetuned,
            masked_wt,
            args.n_sequences,
            args.gen_batch_size,
            label="Guided Finetuned",
        )
        guided_sft_scores = score_with_oracle(oracle, guided_sft_seqs)
        all_scores["Guided Finetuned"] = guided_sft_scores
        print(f"  Generated {len(guided_sft_seqs)} sequences")
    else:
        print("  Skipping (no LoRA adapter from Step 1)")

    # ── Step 3d: Compare and plot ────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("GENERATION COMPARISON")
    print("=" * 60)

    for name, Y in all_scores.items():
        n_total = len(Y)
        n_success = ((Y[:, 0] >= pb_threshold) & (Y[:, 1] <= zn_threshold)).sum().item()
        pct = 100 * n_success / n_total if n_total > 0 else 0
        print(f"  {name:30s}: {n_success:3d}/{n_total} ({pct:5.1f}%) in target region")
        print(f"    Pb logFC: mean={Y[:, 0].mean():.3f}, max={Y[:, 0].max():.3f}")
        print(f"    Zn logFC: mean={Y[:, 1].mean():.3f}, min={Y[:, 1].min():.3f}")

    plot_generation_comparison(
        all_scores,
        dataset_labels=round1_labels,
        output_dir=OUTPUTS_FOLDER,
        pb_threshold=pb_threshold,
        zn_threshold=zn_threshold,
    )

    # Save all generated sequences
    save_path = CHECKPOINTS_FOLDER / "step3_generations.pt"
    torch.save(
        {
            name: {"sequences": seqs, "oracle_scores": scores}
            for name, seqs, scores in [
                ("pretrained_unconditional", uncond_seqs, uncond_scores),
                ("guided_pretrained", guided_seqs, guided_scores),
            ]
            + (
                [("finetuned_unconditional", sft_seqs, sft_scores)]
                if "Finetuned Unconditional" in all_scores
                else []
            )
            + (
                [("guided_finetuned", guided_sft_seqs, guided_sft_scores)]
                if "Guided Finetuned" in all_scores
                else []
            )
        },
        save_path,
    )
    print(f"\nSaved all generations to {save_path}")


if __name__ == "__main__":
    main()
