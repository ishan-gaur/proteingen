"""Step 1: Sample sequences from SwissProt and generate decoding orders.

Fetches reviewed UniProt (Swiss-Prot) entries via the REST API, filters by
length, randomly samples N_SWISSPROT_SEQS, and pre-generates N_ORDERS random
decoding orders for each. Also creates the masked input tensors for all
(sequence, order, mask_fraction) combinations.

Usage:
    uv run python examples/benchmark_model_families/prepare_data.py
    uv run python examples/benchmark_model_families/prepare_data.py --seed 123
"""

import argparse
import json
import random
import sys

import requests
import torch
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

from config import (
    DATA_DIR,
    MASK_FRACTIONS,
    MAX_SEQ_LEN,
    MIN_SEQ_LEN,
    N_ORDERS,
    N_SWISSPROT_SEQS,
    RNG_SEED,
)

sys.path.insert(0, str(DATA_DIR.parent))
from protstar.sampling import generate_unmask_orders, mask_by_order


def fetch_swissprot_sequences(
    n: int,
    min_len: int,
    max_len: int,
    seed: int,
) -> list[dict]:
    """Fetch random reviewed (Swiss-Prot) sequences from UniProt REST API.

    Queries for sequences in the given length range that are reviewed and
    belong to diverse organisms, then randomly samples n entries.
    """
    print(f"Querying UniProt for reviewed sequences ({min_len}-{max_len} aa)...")
    query = f"(reviewed:true) AND (length:[{min_len} TO {max_len}])"
    url = "https://rest.uniprot.org/uniprotkb/search"

    # Fetch a large pool to sample from
    pool_size = min(500, n * 50)
    params = {
        "query": query,
        "format": "json",
        "size": pool_size,
        "fields": "accession,protein_name,sequence,organism_name,length",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    print(f"  Got {len(results)} candidates from UniProt")

    # Parse into clean records
    entries = []
    for r in results:
        seq = r.get("sequence", {}).get("value", "")
        if not seq or len(seq) < min_len or len(seq) > max_len:
            continue
        # Skip sequences with non-standard amino acids
        if any(c not in "ACDEFGHIKLMNPQRSTVWY" for c in seq):
            continue
        entries.append(
            {
                "accession": r["primaryAccession"],
                "name": r.get("proteinDescription", {})
                .get("recommendedName", {})
                .get("fullName", {})
                .get("value", r["primaryAccession"]),
                "organism": r.get("organism", {}).get("scientificName", "Unknown"),
                "sequence": seq,
                "length": len(seq),
            }
        )

    assert len(entries) >= n, f"Only found {len(entries)} valid sequences, need {n}"

    # Randomly sample
    rng = random.Random(seed)
    selected = rng.sample(entries, n)
    selected.sort(key=lambda e: e["length"])
    return selected


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark data")
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--n-sequences", type=int, default=N_SWISSPROT_SEQS)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fetch SwissProt sequences
    entries = fetch_swissprot_sequences(
        n=args.n_sequences,
        min_len=MIN_SEQ_LEN,
        max_len=MAX_SEQ_LEN,
        seed=args.seed,
    )
    print(f"\nSelected {len(entries)} sequences:")
    for e in entries:
        print(
            f"  {e['accession']:10s} | {e['length']:4d} aa | {e['organism'][:30]:30s} | {e['name'][:50]}"
        )

    # Save metadata
    with open(DATA_DIR / "sequences.json", "w") as f:
        json.dump(entries, f, indent=2)

    # 2. Tokenize sequences (using ESM tokenizer as reference — all models use
    # BOS/EOS conventions, orders refer to interior positions)
    tokenizer = EsmSequenceTokenizer()
    sequences = [e["sequence"] for e in entries]
    tokenized = tokenizer(sequences, padding=False)
    token_ids_list = tokenized["input_ids"]  # list of lists

    seq_lengths = [len(toks) for toks in token_ids_list]
    print(f"\nTokenized lengths (incl BOS/EOS): {seq_lengths}")

    # 3. Generate decoding orders
    # Special positions = {0, L-1} (BOS, EOS) for ESM-family tokenizers
    special_positions = [{0, L - 1} for L in seq_lengths]
    orders = generate_unmask_orders(
        seq_lengths=seq_lengths,
        n_orders=N_ORDERS,
        special_positions=special_positions,
        seed=args.seed,
    )

    # Save orders
    orders_serializable = [
        [order.tolist() for order in seq_orders] for seq_orders in orders
    ]
    with open(DATA_DIR / "orders.json", "w") as f:
        json.dump(orders_serializable, f)

    # 4. Create masked inputs for all (sequence, order, mask_fraction) combos
    mask_token_id = tokenizer.mask_token_id
    masked_inputs = {}  # key: f"{seq_idx}_{order_idx}_{mask_frac}" → token list

    for seq_idx in range(len(entries)):
        tokens = torch.LongTensor(token_ids_list[seq_idx])
        for order_idx in range(N_ORDERS):
            order = orders[seq_idx][order_idx]
            for mask_frac in MASK_FRACTIONS:
                masked = mask_by_order(tokens, order, mask_frac, mask_token_id)
                key = f"{seq_idx}_{order_idx}_{mask_frac:.2f}"
                masked_inputs[key] = masked.tolist()

    with open(DATA_DIR / "masked_inputs.json", "w") as f:
        json.dump(masked_inputs, f)

    # Also save token IDs for reference
    torch.save(
        {
            "token_ids": [torch.LongTensor(t) for t in token_ids_list],
            "sequences": sequences,
            "mask_token_id": mask_token_id,
        },
        DATA_DIR / "tokenized.pt",
    )

    n_combos = len(entries) * N_ORDERS * len(MASK_FRACTIONS)
    print(f"\nSaved {n_combos} masked input configurations")
    print(f"  Sequences: {DATA_DIR / 'sequences.json'}")
    print(f"  Orders: {DATA_DIR / 'orders.json'}")
    print(f"  Masked inputs: {DATA_DIR / 'masked_inputs.json'}")
    print(f"  Tokenized: {DATA_DIR / 'tokenized.pt'}")


if __name__ == "__main__":
    main()
