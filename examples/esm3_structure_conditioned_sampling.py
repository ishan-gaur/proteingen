"""ESM3 structure-conditioned sampling and embedding extraction.

Demonstrates using ESM3 as both a generative model (masked language model)
and an embedding model, with optional structure conditioning via atom37
coordinates from a PDB file.

Usage:
    # Unconditional (sequence-only)
    uv run python examples/esm3_structure_conditioned_sampling.py

    # Structure-conditioned (provide a PDB file)
    uv run python examples/esm3_structure_conditioned_sampling.py --pdb path/to/structure.pdb
"""

import argparse

import torch

from proteingen.models.esm import ESM3
from proteingen.sampling import sample_any_order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", type=str, default=None, help="PDB file for structure conditioning")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=50, help="Sequence length (ignored if --pdb)")
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = ESM3().to(device)
    tokenizer = model.tokenizer
    mask_id = tokenizer.vocab["<mask>"]

    # ── Structure conditioning ───────────────────────────────────────────
    if args.pdb:
        from proteingen.models.utils import pdb_to_atom37_and_seq

        coords_RAX, wt_seq = pdb_to_atom37_and_seq(args.pdb, backbone_only=True)
        seq_len = len(wt_seq)
        print(f"Loaded structure: {args.pdb}")
        print(f"  WT sequence ({seq_len} residues): {wt_seq[:60]}{'...' if seq_len > 60 else ''}")
        model.set_condition_({"coords_RAX": coords_RAX})
    else:
        seq_len = args.seq_len
        wt_seq = None
        print(f"No structure provided — unconditional sampling (L={seq_len})")

    # ── Sample sequences ─────────────────────────────────────────────────
    # Fully masked starting point: CLS + masks + EOS
    template = torch.tensor(
        [[tokenizer.cls_token_id] + [mask_id] * seq_len + [tokenizer.eos_token_id]],
        dtype=torch.long,
        device=device,
    )
    masked_batch = template.repeat(args.n_samples, 1)

    print(f"\nSampling {args.n_samples} sequences (temp={args.temp})...")
    model.set_temp_(args.temp)
    sequences = sample_any_order(model, masked_batch, return_string=True)

    print("\nGenerated sequences:")
    for i, seq in enumerate(sequences):
        print(f"  [{i}] {seq[:80]}{'...' if len(seq) > 80 else ''}")

    # ── Extract embeddings ───────────────────────────────────────────────
    print("\nExtracting embeddings for generated sequences...")
    tokens = tokenizer(sequences, padding=True, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        embeddings = model.embed(tokens)  # (N, L+2, D) — also structure-conditioned
    print(f"  Embedding shape: {embeddings.shape} (N, L+special, D={model.EMB_DIM})")

    # Mean-pool excluding special tokens
    special_ids = set(tokenizer.all_special_ids)
    mask = torch.tensor(
        [[t.item() not in special_ids for t in row] for row in tokens],
        device=device,
    ).unsqueeze(-1)
    pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    print(f"  Pooled shape: {pooled.shape}")

    # Cosine similarity between samples
    if args.n_samples > 1:
        normed = pooled / pooled.norm(dim=-1, keepdim=True)
        sim = normed @ normed.T
        print(f"  Pairwise cosine similarity:\n{sim.cpu()}")


if __name__ == "__main__":
    main()
