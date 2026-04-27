"""Train an MLP probe on ESMC embeddings to predict TrpB fitness, then run guided sampling.

Uses frozen ESMC embeddings at variable positions (concatenated per-position)
with an MLP head to predict continuous fitness values from the SaProtHub TrpB
dataset (160k protein sequences, length 397, 4 variable positions).

Pre-computes embeddings once (cached to disk), trains the MLP head,
then demonstrates TAG-guided sampling vs unguided sampling.

Usage:
    uv run python examples/trpb_linear_probe.py --device cuda
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset

from proteingen.modeling import TAG
from proteingen.modeling import ESMC
from proteingen.modeling import LinearProbe, point_estimate_binary_logits
from proteingen.sampling import sample


DATASET_ID = "SaProtHub/Dataset-TrpB_fitness_landsacpe"


# ── TrpB fitness predictor ──────────────────────────────────────────────────


class TrpBFitnessPredictor(LinearProbe):
    """Predicts TrpB fitness using ESMC embeddings at variable positions + MLP head.

    Uses per-position embeddings at the 4 variable sites (concatenated, not averaged),
    preserving position-specific information. The MLP head learns nonlinear
    interactions between positions.
    """

    def __init__(
        self,
        esmc_checkpoint: str = "esmc_300m",
        variable_positions: list[int] | None = None,
        hidden_dim: int = 256,
        k: float = 100.0,
    ):
        esmc = ESMC(esmc_checkpoint)

        # Variable-position concat pooling: extract and concatenate embeddings
        # at the variable sites. +1 offset for CLS token.
        var_pos_tokens = [p + 1 for p in variable_positions] if variable_positions else None

        def varpos_concat_pool(emb_SPD, seq_SP):
            if var_pos_tokens is not None:
                return emb_SPD[:, var_pos_tokens, :].reshape(emb_SPD.shape[0], -1)
            # Fallback: mean pool excluding special tokens
            mask = (seq_SP != esmc.tokenizer.pad_token_id).unsqueeze(-1).float()
            return (emb_SPD * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        super().__init__(embed_model=esmc, output_dim=1, pooling_fn=varpos_concat_pool)

        # Replace linear head with MLP
        n_pos = len(variable_positions) if variable_positions else 1
        head_input_dim = n_pos * esmc.EMB_DIM
        self.w = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.k = k

    def format_raw_to_logits(
        self, raw_output: torch.FloatTensor, ohe_seq_SPT: torch.FloatTensor, **kwargs
    ) -> torch.FloatTensor:
        return point_estimate_binary_logits(raw_output, self.target, self.k)


# ── Data loading ─────────────────────────────────────────────────────────────


def load_trpb_data() -> pd.DataFrame:
    path = hf_hub_download(DATASET_ID, "dataset.csv", repo_type="dataset")
    return pd.read_csv(path)


def find_variable_positions(df: pd.DataFrame) -> list[int]:
    arr = np.array([list(s) for s in df["protein"].tolist()])
    n_unique = np.array([len(set(arr[:, i])) for i in range(arr.shape[1])])
    return np.where(n_unique > 1)[0].tolist()


def get_or_compute_embeddings(
    split_name: str,
    sequences: list[str],
    predictor: TrpBFitnessPredictor,
    batch_size: int,
    device: torch.device,
    cache_dir: Path,
) -> torch.Tensor:
    cache_path = cache_dir / f"{split_name}.pt"
    if cache_path.exists():
        emb = torch.load(cache_path, weights_only=True)
        print(f"  Loaded cached {split_name} embeddings: {emb.shape}")
        return emb

    embeddings = predictor.precompute_embeddings(sequences, batch_size, device)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, cache_path)
    print(f"  Cached to {cache_path}: {embeddings.shape}")
    return embeddings


# ── Scoring helper ───────────────────────────────────────────────────────────


def score_sequences(
    sequences: list[str],
    predictor: TrpBFitnessPredictor,
    variable_pos: list[int],
    seq_to_label: dict,
) -> list[tuple[str, float, float | None]]:
    tokenizer = predictor.tokenizer
    results = []
    with torch.no_grad():
        for seq in sequences:
            var_res = "".join(seq[p] for p in variable_pos)
            tokens = tokenizer([seq], return_tensors="pt")["input_ids"]
            emb = predictor.embed_model.embed(tokens)
            pooled = predictor.pooling_fn(emb, tokens)
            pred = predictor.w(pooled).item()
            actual = seq_to_label.get(seq)
            results.append((var_res, pred, actual))
    return results


def print_results(results: list, label: str):
    print(f"\n{label}")
    print(f"{'Var res':>10} | {'Predicted':>10} | {'Actual':>10}")
    print("-" * 36)
    actual_vals = []
    for var_res, pred, actual in results:
        actual_str = f"{actual:.4f}" if actual is not None else "N/A"
        print(f"{var_res:>10} | {pred:>10.4f} | {actual_str:>10}")
        if actual is not None:
            actual_vals.append(actual)
    if actual_vals:
        print(f"  Mean actual fitness: {np.mean(actual_vals):.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--esmc-checkpoint", default="esmc_600m")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--n-runs", type=int, default=5, help="Number of sampling runs for statistics")
    parser.add_argument("--gen-temp", type=float, default=2.0)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    args = parser.parse_args()
    device = torch.device(args.device)

    # ── Load data ────────────────────────────────────────────────────────
    print("Loading TrpB fitness dataset...")
    df = load_trpb_data()
    variable_pos = find_variable_positions(df)
    seq_to_label = dict(zip(df["protein"], df["label"]))
    labels = df["label"].values
    print(f"  {len(df)} sequences, variable positions: {variable_pos}")
    print(f"  Labels: mean={labels.mean():.4f}, std={labels.std():.4f}, max={labels.max():.4f}")

    train_df = df[df["stage"] == "train"].reset_index(drop=True)
    valid_df = df[df["stage"] == "valid"].reset_index(drop=True)
    test_df = df[df["stage"] == "test"].reset_index(drop=True)

    # ── Build predictor & compute embeddings ─────────────────────────────
    print(f"\nBuilding predictor (ESMC={args.esmc_checkpoint}, hidden={args.hidden_dim})...")
    predictor = TrpBFitnessPredictor(
        args.esmc_checkpoint,
        variable_positions=variable_pos,
        hidden_dim=args.hidden_dim,
    )

    cache_dir = Path(f"data/trpb_embeddings/{args.esmc_checkpoint}_varpos")
    print("Computing embeddings...")
    train_emb = get_or_compute_embeddings(
        "train", train_df["protein"].tolist(), predictor, args.embed_batch_size, device, cache_dir
    )
    valid_emb = get_or_compute_embeddings(
        "valid", valid_df["protein"].tolist(), predictor, args.embed_batch_size, device, cache_dir
    )
    test_emb = get_or_compute_embeddings(
        "test", test_df["protein"].tolist(), predictor, args.embed_batch_size, device, cache_dir
    )

    # Free ESMC GPU memory during training
    predictor.embed_model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_labels = torch.tensor(train_df["label"].values, dtype=torch.float32)
    valid_labels = torch.tensor(valid_df["label"].values, dtype=torch.float32)
    test_labels = torch.tensor(test_df["label"].values, dtype=torch.float32)

    # ── Train the MLP head ───────────────────────────────────────────────
    head = predictor.w.to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"MLP head: {n_params} params, input_dim={train_emb.shape[1]}")

    train_loader = DataLoader(
        TensorDataset(train_emb, train_labels), batch_size=args.train_batch_size, shuffle=True
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_rho, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, args.epochs + 1):
        head.train()
        for emb_batch, label_batch in train_loader:
            emb_batch, label_batch = emb_batch.to(device), label_batch.to(device)
            pred = head(emb_batch).squeeze(-1)
            loss = F.mse_loss(pred, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            head.eval()
            with torch.no_grad():
                vpreds = head(valid_emb.to(device)).squeeze(-1).cpu()
                rho, _ = spearmanr(vpreds.numpy(), valid_labels.numpy())
            if rho > best_rho:
                best_rho = rho
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in head.state_dict().items()}
            print(f"  Epoch {epoch:3d} | valid ρ={rho:.4f} (best={best_rho:.4f})")

    assert best_state is not None
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        tpreds = head(test_emb.to(device)).squeeze(-1).cpu()
        test_rho, _ = spearmanr(tpreds.numpy(), test_labels.numpy())
    print(f"\nBest valid ρ={best_rho:.4f} @ epoch {best_epoch}, test ρ={test_rho:.4f}")

    # ── Guided sampling comparison ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("GUIDED SAMPLING COMPARISON")
    print("=" * 60)

    threshold = float(np.percentile(labels, 99))
    predictor.set_target_(threshold)
    predictor.cpu()
    # Guidance strength is controlled by predictor temperature:
    # lower temp → steeper log_softmax → larger gradient magnitude.
    # guidance_scale of N corresponds to pred temp of 1/N.
    predictor.set_temp_(1.0 / args.guidance_scale)
    print(f"Target threshold: {threshold:.4f} (99th percentile)")
    print(f"Gen temp: {args.gen_temp}, Pred temp: {predictor.temp:.4f} (guidance_scale={args.guidance_scale})")

    gen_model = ESMC(args.esmc_checkpoint)
    tokenizer = gen_model.tokenizer
    template_seq = df["protein"].iloc[0]

    mask_id = tokenizer.vocab["<mask>"]
    template_tokens = tokenizer([template_seq], return_tensors="pt")["input_ids"]
    mask_positions = [p + 1 for p in variable_pos]

    def make_masked(n):
        masked = template_tokens.repeat(n, 1)
        for mp in mask_positions:
            masked[:, mp] = mask_id
        return masked

    n = args.n_samples

    # Collect results over multiple runs
    results_by_condition = {
        "random": [],
        "unguided": [],
        "guided": [],
        "guided_hot": [],
    }
    aas = "ACDEFGHIKLMNPQRSTVWY"

    for run in range(args.n_runs):
        print(f"\n--- Run {run + 1}/{args.n_runs} ---")

        # 1. Random
        random_seqs = []
        for _ in range(n):
            seq_list = list(template_seq)
            for p in variable_pos:
                seq_list[p] = random.choice(aas)
            random_seqs.append("".join(seq_list))
        for _, _, actual in score_sequences(random_seqs, predictor, variable_pos, seq_to_label):
            if actual is not None:
                results_by_condition["random"].append(actual)

        # 2. Unguided (ESMC, temp=1.0)
        gen_model.set_temp_(1.0)
        unguided = sample(gen_model, make_masked(n))["sequences"]
        unguided = [s.replace(" ", "") for s in unguided]
        for _, _, actual in score_sequences(unguided, predictor, variable_pos, seq_to_label):
            if actual is not None:
                results_by_condition["unguided"].append(actual)

        # 3. TAG guided, gen temp=1.0
        tag = TAG(gen_model, predictor)
        guided = sample(tag, make_masked(n))["sequences"]
        guided = [s.replace(" ", "") for s in guided]
        for _, _, actual in score_sequences(guided, predictor, variable_pos, seq_to_label):
            if actual is not None:
                results_by_condition["guided"].append(actual)

        # 4. TAG guided, higher gen temp
        gen_model.set_temp_(args.gen_temp)
        tag_hot = TAG(gen_model, predictor)
        guided_hot = sample(tag_hot, make_masked(n))["sequences"]
        guided_hot = [s.replace(" ", "") for s in guided_hot]
        gen_model.set_temp_(1.0)
        for _, _, actual in score_sequences(guided_hot, predictor, variable_pos, seq_to_label):
            if actual is not None:
                results_by_condition["guided_hot"].append(actual)

    # ── Summary statistics ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY (actual fitness, aggregated over all runs)")
    print("=" * 60)
    print(f"{'Condition':>30s} | {'N':>5s} | {'Mean':>8s} | {'Std':>8s} | {'Median':>8s} | {'% > 0.3':>8s}")
    print("-" * 80)
    for name, label in [
        ("random", "Random"),
        ("unguided", "Unguided (temp=1.0)"),
        ("guided", f"TAG guided (scale={args.guidance_scale})"),
        ("guided_hot", f"TAG (temp={args.gen_temp}, scale={args.guidance_scale})"),
    ]:
        vals = results_by_condition[name]
        vals_arr = np.array(vals)
        pct_above = (vals_arr > 0.3).mean() * 100
        print(
            f"{label:>30s} | {len(vals):>5d} | "
            f"{vals_arr.mean():>8.4f} | {vals_arr.std():>8.4f} | "
            f"{np.median(vals_arr):>8.4f} | {pct_above:>7.1f}%"
        )


if __name__ == "__main__":
    main()
