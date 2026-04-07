"""Step 2: Generate sequences with all models at all masking levels.

Loads the prepared data from step 1, then for each model runs ordered
ancestral sampling on every (sequence, order, mask_fraction) combination.
Saves per-step log probabilities and generated sequences.

Can be run for a single model (useful for parallel/SLURM execution):
    uv run python examples/benchmark_model_families/generate.py --model esmc_300m
    uv run python examples/benchmark_model_families/generate.py --model all

Usage:
    uv run python examples/benchmark_model_families/generate.py --model esmc_300m --device cuda
    uv run python examples/benchmark_model_families/generate.py --model all --device cuda
"""

import argparse
import json
import sys
import time

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from config import DATA_DIR, MASK_FRACTIONS, MODEL_CONFIGS, N_ORDERS, OUTPUT_DIR


def load_model(family: str, checkpoint: str, device: str):
    """Instantiate a TransitionModel by family and checkpoint."""
    if family == "esmc":
        from proteingen.models.esm import ESMC

        return ESMC(checkpoint).to(device).eval()
    elif family == "esm3":
        from proteingen.models.esm import ESM3

        return ESM3(checkpoint).to(device).eval()
    elif family == "dplm2":
        from proteingen.models import DPLM2

        return DPLM2(checkpoint).to(device).eval()
    else:
        raise ValueError(f"Unknown model family: {family}")


def get_unmask_tail(
    order: list[int], mask_frac: float, mask_token_id: int, masked_tokens: list[int]
) -> torch.LongTensor:
    """Get the tail of the order that corresponds to masked positions.

    Returns the positions in unmask order (first element = first to unmask).
    """
    import math

    n_maskable = len(order)
    n_to_mask = math.ceil(mask_frac * n_maskable)
    n_keep = n_maskable - n_to_mask
    # The tail of the order = positions that were masked
    return torch.LongTensor(order[n_keep:])


def retokenize_for_model(sequence: str, model) -> torch.LongTensor:
    """Tokenize a sequence using the model's own tokenizer."""
    tokenized = model.tokenizer([sequence], padding=False, return_tensors="pt")
    return tokenized["input_ids"][0]


def translate_masked_input(
    esm_masked_tokens: list[int],
    esm_mask_token_id: int,
    original_sequence: str,
    model,
) -> torch.LongTensor:
    """Translate a masked input from ESM tokenization to the model's tokenizer.

    Finds which CHARACTER positions are masked (by mapping ESM token positions
    to sequence positions), then applies masking to the model's own tokenization.
    """
    model_tokens = retokenize_for_model(original_sequence, model)
    model_mask_id = model.tokenizer.mask_token_id

    # Find which sequence positions are masked in the ESM tokenization
    # ESM tokens: [CLS, aa1, aa2, ..., EOS] → sequence positions 0..L-1 map to token positions 1..L
    masked_seq_positions = set()
    for tok_idx, tok in enumerate(esm_masked_tokens):
        if tok == esm_mask_token_id:
            # token position → sequence position (subtract 1 for BOS)
            seq_pos = tok_idx - 1
            if 0 <= seq_pos < len(original_sequence):
                masked_seq_positions.add(seq_pos)

    # Apply masking to model's tokenization
    # Need to figure out which token positions in the model correspond to which sequence positions
    # For ESM-family and DPLM2: [BOS, aa1, aa2, ..., EOS] → seq pos i maps to token pos i+1
    # This is the same convention, so we can directly mask token positions offset by 1
    model_masked = model_tokens.clone()
    for seq_pos in masked_seq_positions:
        tok_pos = seq_pos + 1  # +1 for BOS token
        if tok_pos < len(model_masked) - 1:  # don't mask EOS
            model_masked[tok_pos] = model_mask_id

    return model_masked


def translate_unmask_order(
    esm_order_positions: list[int],
    original_sequence: str,
    model,
) -> torch.LongTensor:
    """Translate unmask order from ESM token positions to model token positions.

    ESM order positions are in ESM tokenizer space (BOS at 0, aa at 1..L, EOS at L+1).
    We convert to the model's token position space.
    """
    # For ESM and DPLM2, the mapping is the same: BOS at 0, aa at 1..L, EOS at L+1
    # So the order positions are the same
    # But let's be explicit about the mapping via sequence positions
    model_positions = []
    for esm_tok_pos in esm_order_positions:
        seq_pos = esm_tok_pos - 1  # ESM token pos → sequence position
        model_tok_pos = (
            seq_pos + 1
        )  # sequence position → model token pos (assumes BOS at 0)
        model_positions.append(model_tok_pos)
    return torch.LongTensor(model_positions)


def run_generation_for_model(
    family: str,
    display_name: str,
    checkpoint: str,
    device: str,
):
    """Run ordered ancestral sampling for one model across all configs."""
    import math

    from proteingen.sampling import sample_ordered_ancestral

    print(f"\n{'=' * 60}")
    print(f"Generating with {display_name} ({checkpoint})")
    print(f"{'=' * 60}")

    # Load prepared data
    with open(DATA_DIR / "sequences.json") as f:
        entries = json.load(f)
    with open(DATA_DIR / "orders.json") as f:
        all_orders = json.load(f)
    with open(DATA_DIR / "masked_inputs.json") as f:
        masked_inputs = json.load(f)

    tokenized_data = torch.load(DATA_DIR / "tokenized.pt", weights_only=False)
    esm_mask_token_id = tokenized_data["mask_token_id"]
    sequences = tokenized_data["sequences"]

    # Load model
    t0 = time.time()
    model = load_model(family, checkpoint, device)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Output directory for this model — load existing results for resume
    model_dir = OUTPUT_DIR / display_name.replace(" ", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / "generation_results.json"

    results = {}
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} cached results from {out_path}")

    total_keys = len(entries) * N_ORDERS * len(MASK_FRACTIONS)
    n_done = len(results)
    print(f"Need to generate {total_keys - n_done} / {total_keys} samples")

    for seq_idx in range(len(entries)):
        seq = sequences[seq_idx]
        print(
            f"\n  Sequence {seq_idx}: {entries[seq_idx]['accession']} ({len(seq)} aa)"
        )

        for order_idx in range(N_ORDERS):
            esm_order = all_orders[seq_idx][order_idx]

            for mask_frac in MASK_FRACTIONS:
                key = f"{seq_idx}_{order_idx}_{mask_frac:.2f}"

                # Skip if already computed
                if key in results:
                    continue

                esm_masked = masked_inputs[key]

                # Translate to model's tokenizer space
                model_masked = translate_masked_input(
                    esm_masked, esm_mask_token_id, seq, model
                )
                model_unmask_order = translate_unmask_order(esm_order, seq, model)

                # Get the unmask tail (only the masked positions, in order)
                n_maskable = len(esm_order)
                n_to_mask = math.ceil(mask_frac * n_maskable)
                n_keep = n_maskable - n_to_mask
                model_unmask_tail = model_unmask_order[n_keep:]

                # Verify masking is correct
                n_actual_masked = (
                    (model_masked == model.tokenizer.mask_token_id).sum().item()
                )
                assert n_actual_masked == n_to_mask, (
                    f"Expected {n_to_mask} masked positions, got {n_actual_masked}"
                )

                # Run ordered ancestral sampling (batch size 1)
                x_SP = model_masked.unsqueeze(0).to(device)
                t0 = time.time()
                trajectory = sample_ordered_ancestral(
                    model=model,
                    x_SP=x_SP,
                    unmask_orders=[model_unmask_tail],
                )
                dt = time.time() - t0

                gen_seq = trajectory["sequences"][0]
                step_lps = trajectory["step_log_probs"][0]  # (n_steps,)

                print(
                    f"    order={order_idx} mask={mask_frac:.0%}: "
                    f"{n_to_mask} steps, {dt:.1f}s, "
                    f"mean_lp={step_lps[~step_lps.isnan()].mean():.3f}"
                )

                results[key] = {
                    "generated_sequence": gen_seq,
                    "step_log_probs": step_lps.tolist(),
                    "step_positions": trajectory["step_positions"][0].tolist(),
                    "step_tokens": trajectory["step_tokens"][0].tolist(),
                    "time_seconds": dt,
                    "mask_fraction": mask_frac,
                    "order_idx": order_idx,
                    "seq_idx": seq_idx,
                }

        # Save after each sequence (for resume)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [checkpoint] Saved {len(results)} results to {out_path}")

    print(f"\nDone — {len(results)} total results saved to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate sequences with models")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model checkpoint to run (e.g. esmc_300m) or 'all'",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        configs_to_run = MODEL_CONFIGS
    else:
        configs_to_run = [
            c for c in MODEL_CONFIGS if c[2] == args.model or c[1] == args.model
        ]
        assert len(configs_to_run) > 0, (
            f"Model '{args.model}' not found. Available: "
            + ", ".join(c[2] for c in MODEL_CONFIGS)
        )

    for family, display_name, checkpoint, _ in configs_to_run:
        try:
            run_generation_for_model(family, display_name, checkpoint, args.device)
        except Exception as e:
            print(f"\nERROR generating with {display_name}: {e}")
            import traceback

            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
