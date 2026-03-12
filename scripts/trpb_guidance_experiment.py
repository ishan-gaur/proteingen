"""TrpB guided sampling experiment: compare unguided vs DEG-guided sampling.

Runs on GPU for speed. Uses the proper DEG class from guide.py with
argmax_masked_positions=True (since the predictor was trained on clean sequences).
Guidance strength controlled via pred model temperature (lower = stronger).
Gen model temperature flattens the prior to give guidance more room.
"""
import sys; sys.path.insert(0, ".")

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset

from dfm.guide import DEG
from dfm.models.esm import ESMC
from dfm.sampling import sample_any_order_ancestral
from examples.trpb_linear_probe import (
    TrpBFitnessPredictor,
    find_variable_positions,
    load_trpb_data,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--esmc-checkpoint", default="esmc_300m")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--train-epochs", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--n-runs", type=int, default=5)
    args = parser.parse_args()
    device = torch.device(args.device)

    # ── Data ─────────────────────────────────────────────────────────────
    print("Loading data...", flush=True)
    df = load_trpb_data()
    variable_pos = find_variable_positions(df)
    seq_to_label = dict(zip(df["protein"], df["label"]))
    labels = df["label"].values
    threshold = float(np.percentile(labels, 99))
    print(f"  {len(df)} seqs, var_pos={variable_pos}, threshold={threshold:.4f}", flush=True)

    # ── Train predictor ──────────────────────────────────────────────────
    print(f"\nTraining predictor ({args.esmc_checkpoint})...", flush=True)
    predictor = TrpBFitnessPredictor(
        args.esmc_checkpoint, variable_positions=variable_pos, hidden_dim=args.hidden_dim
    )

    cache_dir = Path(f"data/trpb_embeddings/{args.esmc_checkpoint}_varpos")
    train_emb = torch.load(cache_dir / "train.pt", weights_only=True)
    valid_emb = torch.load(cache_dir / "valid.pt", weights_only=True)
    test_emb = torch.load(cache_dir / "test.pt", weights_only=True)
    train_labels = torch.tensor(df[df["stage"] == "train"]["label"].values, dtype=torch.float32)
    valid_labels = torch.tensor(df[df["stage"] == "valid"]["label"].values, dtype=torch.float32)
    test_labels = torch.tensor(df[df["stage"] == "test"]["label"].values, dtype=torch.float32)

    head = predictor.w.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.train_epochs)
    loader = DataLoader(TensorDataset(train_emb, train_labels), batch_size=512, shuffle=True)

    best_rho, best_state = -1.0, None
    for epoch in range(1, args.train_epochs + 1):
        head.train()
        for e, l in loader:
            e, l = e.to(device), l.to(device)
            loss = F.mse_loss(head(e).squeeze(-1), l)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if epoch % 20 == 0:
            head.eval()
            with torch.no_grad():
                pv = head(valid_emb.to(device)).squeeze(-1).cpu()
                rho, _ = spearmanr(pv.numpy(), valid_labels.numpy())
            if rho > best_rho:
                best_rho = rho
                best_state = {k: v.clone() for k, v in head.state_dict().items()}
            print(f"  Epoch {epoch}: valid ρ={rho:.4f} (best={best_rho:.4f})", flush=True)

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        pt = head(test_emb.to(device)).squeeze(-1).cpu()
        test_rho, _ = spearmanr(pt.numpy(), test_labels.numpy())
    print(f"\n  Test ρ = {test_rho:.4f}", flush=True)

    predictor.to(device)
    predictor.set_target_(threshold)

    # ── Verify predictor rankings ────────────────────────────────────────
    gen_model = ESMC(args.esmc_checkpoint).to(device)
    tokenizer = gen_model.tokenizer
    template_seq = df["protein"].iloc[0]
    mask_id = tokenizer.vocab["<mask>"]
    template_tokens = tokenizer([template_seq], return_tensors="pt")["input_ids"].to(device)
    mask_positions = [p + 1 for p in variable_pos]
    aas = "ACDEFGHIKLMNPQRSTVWY"
    aa_indices = torch.tensor([tokenizer.vocab[aa] for aa in aas]).to(device)

    masked = template_tokens.clone()
    for mp in mask_positions:
        masked[:, mp] = mask_id
    gen_lp = gen_model.get_log_probs(masked)
    filled = masked.clone()
    for mp in mask_positions:
        filled[0, mp] = gen_lp[0, mp, :33].argmax()

    print("\nPredictor rankings at variable positions (context=gen argmax fill):")
    for mp in mask_positions:
        cands = filled.expand(20, -1).clone()
        cands[:, mp] = aa_indices
        with torch.no_grad():
            emb = predictor.embed_model.embed(cands)
            pooled = predictor.pooling_fn(emb, cands)
            raw = predictor.w(pooled).squeeze(-1)
        scores = [(aas[i], raw[i].item()) for i in range(20)]
        scores.sort(key=lambda x: -x[1])
        print(f"  Pos {mp}: {scores[:5]}", flush=True)

    # ── Sampling experiment ──────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("SAMPLING EXPERIMENT", flush=True)
    print("=" * 70, flush=True)

    def make_masked(n):
        m = template_tokens.repeat(n, 1)
        for mp in mask_positions:
            m[:, mp] = mask_id
        return m

    # Guidance strength via pred model temp: lower temp → more peaked log p(y|x).
    # Gen model temp flattens the prior to give guidance room.
    conditions = [
        ("Random", None, 1.0, 1.0),
        ("Unguided t=1", "gen", 1.0, 1.0),
        ("Unguided t=2", "gen", 2.0, 1.0),
        ("Unguided t=3", "gen", 3.0, 1.0),
        ("DEG gt=2 pt=1.0", "deg", 2.0, 1.0),
        ("DEG gt=2 pt=0.1", "deg", 2.0, 0.1),
        ("DEG gt=2 pt=0.05", "deg", 2.0, 0.05),
        ("DEG gt=3 pt=0.1", "deg", 3.0, 0.1),
        ("DEG gt=3 pt=0.05", "deg", 3.0, 0.05),
    ]

    all_results = {name: [] for name, _, _, _ in conditions}
    n = args.n_samples

    for run in range(args.n_runs):
        print(f"\n--- Run {run + 1}/{args.n_runs} ---", flush=True)
        for name, mode, gen_temp, pred_temp in conditions:
            t0 = time.time()
            gen_model.set_temp_(gen_temp)
            if mode is None:
                seqs = []
                for _ in range(n):
                    s = list(template_seq)
                    for p in variable_pos:
                        s[p] = random.choice(aas)
                    seqs.append("".join(s))
            elif mode == "gen":
                seqs = sample_any_order_ancestral(gen_model, make_masked(n), return_string=True)
                seqs = [s.replace(" ", "") for s in seqs]
            elif mode == "deg":
                predictor.set_temp_(pred_temp)
                deg = DEG(gen_model, predictor, argmax_masked_positions=True)
                seqs = sample_any_order_ancestral(deg, make_masked(n), return_string=True)
                seqs = [s.replace(" ", "") for s in seqs]
                predictor.set_temp_(1.0)
            gen_model.set_temp_(1.0)

            actuals = [seq_to_label.get(s) for s in seqs]
            actuals = [a for a in actuals if a is not None]
            all_results[name].extend(actuals)
            elapsed = time.time() - t0
            arr = np.array(actuals)
            combos = sorted(set("".join(s[p] for p in variable_pos) for s in seqs))
            print(
                f"  {name:>20s}: mean={arr.mean():.4f}, median={np.median(arr):.4f}, "
                f">0.5={100*(arr>0.5).mean():.0f}%, >0.7={100*(arr>0.7).mean():.0f}%, "
                f"{len(combos)} unique, {elapsed:.1f}s",
                flush=True,
            )

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"{'Condition':>20s} | {'N':>5s} | {'Mean':>8s} | {'Std':>8s} | "
        f"{'Median':>8s} | {'>0.3':>6s} | {'>0.5':>6s} | {'>0.7':>6s}"
    )
    print("-" * 85)
    for name, _, _, _ in conditions:
        vals = np.array(all_results[name])
        print(
            f"{name:>20s} | {len(vals):>5d} | {vals.mean():>8.4f} | {vals.std():>8.4f} | "
            f"{np.median(vals):>8.4f} | {100*(vals>0.3).mean():>5.1f}% | "
            f"{100*(vals>0.5).mean():>5.1f}% | {100*(vals>0.7).mean():>5.1f}%"
        )


if __name__ == "__main__":
    main()
