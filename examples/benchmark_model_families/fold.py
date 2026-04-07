"""Step 3: Fold all generated sequences (and originals) with AlphaFold 3.

Reads generated sequences from step 2, submits them to the AF3 server,
collects pLDDT and ranking scores. Also folds the original SwissProt
sequences as reference.

Prerequisites:
    - AF3 server running (see af3-server/launch.sh)
    - Step 2 (generate.py) completed for at least one model

Usage:
    uv run python examples/benchmark_model_families/fold.py
    uv run python examples/benchmark_model_families/fold.py --server http://localhost:8080
    uv run python examples/benchmark_model_families/fold.py --model ESMC-300M
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import AF3_POLL_INTERVAL, AF3_SERVER_URL, AF3_TIMEOUT, DATA_DIR, OUTPUT_DIR

from af3_server import AF3Client


def collect_sequences_to_fold() -> dict[str, str]:
    """Collect all unique sequences that need folding: originals + generated.

    Returns dict mapping a unique name to the sequence string.
    Deduplicates identical sequences across models/conditions.
    """
    sequences = {}

    # Original sequences
    with open(DATA_DIR / "sequences.json") as f:
        entries = json.load(f)
    for i, entry in enumerate(entries):
        name = f"original_{i}_{entry['accession']}"
        sequences[name] = entry["sequence"]

    # Generated sequences from each model
    for model_dir in OUTPUT_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        results_path = model_dir / "generation_results.json"
        if not results_path.exists():
            continue

        with open(results_path) as f:
            results = json.load(f)

        model_name = model_dir.name
        for key, result in results.items():
            seq = result["generated_sequence"]
            name = f"{model_name}_{key}"
            sequences[name] = seq

    return sequences


def fold_all(
    client: AF3Client,
    sequences: dict[str, str],
    output_dir: Path,
    skip_existing: bool = True,
):
    """Fold all sequences, saving results as they complete."""
    fold_results_path = output_dir / "fold_results.json"

    # Load existing results if resuming
    existing = {}
    if skip_existing and fold_results_path.exists():
        with open(fold_results_path) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing fold results")

    to_fold = {k: v for k, v in sequences.items() if k not in existing}
    print(f"Need to fold {len(to_fold)} sequences ({len(existing)} already done)")

    if not to_fold:
        print("Nothing to fold!")
        return existing

    results = dict(existing)
    total = len(to_fold)

    for idx, (name, seq) in enumerate(to_fold.items()):
        print(f"\n[{idx + 1}/{total}] Folding {name} ({len(seq)} aa)...", flush=True)

        try:
            result = client.fold(sequence=seq, name=name)

            results[name] = {
                "sequence": seq,
                "length": len(seq),
                "best_ranking_score": result.best_ranking_score,
                "summary_confidences": result.summary_confidences,
                "output_dir": result.output_dir,
                "elapsed_seconds": result.elapsed_seconds,
                "job_id": result.job_id,
            }

            ptm = result.summary_confidences.get("ptm", None)
            print(
                f"  Done in {result.elapsed_seconds:.1f}s, "
                f"ranking_score={result.best_ranking_score:.4f}, "
                f"pTM={ptm}"
            )

            # Download CIF
            cif_dir = output_dir / "cif_files"
            cif_dir.mkdir(parents=True, exist_ok=True)
            cif_path = cif_dir / f"{name}.cif"
            client.download_cif(result.job_id, cif_path)

        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = {
                "sequence": seq,
                "length": len(seq),
                "error": str(e),
            }

        # Save after each fold (for resume capability)
        with open(fold_results_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Fold sequences with AF3")
    parser.add_argument("--server", default=AF3_SERVER_URL)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Only fold sequences from this model (e.g. ESMC-300M)",
    )
    parser.add_argument("--no-skip", action="store_true", help="Re-fold everything")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    client = AF3Client(
        base_url=args.server,
        poll_interval=AF3_POLL_INTERVAL,
        timeout=AF3_TIMEOUT,
    )

    # Check server health
    try:
        h = client.health()
        print(f"AF3 server OK, queue size: {h['queue_size']}")
    except Exception as e:
        print(f"Cannot reach AF3 server at {args.server}: {e}")
        print("Start the server first: sbatch af3-server/launch.sh")
        sys.exit(1)

    # Collect sequences
    sequences = collect_sequences_to_fold()
    print(f"\nTotal unique sequences to fold: {len(sequences)}")

    # Filter by model if specified
    if args.model:
        model_key = args.model.replace(" ", "_")
        filtered = {
            k: v
            for k, v in sequences.items()
            if k.startswith(model_key) or k.startswith("original_")
        }
        print(f"Filtered to {len(filtered)} sequences (model={args.model})")
        sequences = filtered

    fold_dir = OUTPUT_DIR / "fold_results"
    fold_dir.mkdir(parents=True, exist_ok=True)

    results = fold_all(
        client=client,
        sequences=sequences,
        output_dir=fold_dir,
        skip_existing=not args.no_skip,
    )

    # Summary
    n_success = sum(1 for r in results.values() if "error" not in r)
    n_fail = sum(1 for r in results.values() if "error" in r)
    print(f"\nFolding complete: {n_success} succeeded, {n_fail} failed")


if __name__ == "__main__":
    main()
