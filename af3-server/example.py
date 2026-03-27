"""Example: fold a batch of designed proteins using the AF3 server.

Usage:
    # 1. Start the server (in another terminal or via SLURM):
    #    sbatch af3-server/launch.sh
    #    # Wait for "Server ready." in the log
    #
    # 2. Run this script:
    #    python af3-server/example.py --server http://localhost:8080

    # With a FASTA file:
    #    python af3-server/example.py --server http://localhost:8080 --fasta designs.fasta
"""

import argparse
import sys
from pathlib import Path

from client import AF3Client


def main():
    parser = argparse.ArgumentParser(description="Fold proteins via AF3 server")
    parser.add_argument(
        "--server", default="http://localhost:8080", help="AF3 server URL"
    )
    parser.add_argument("--fasta", type=str, help="FASTA file with sequences")
    parser.add_argument(
        "--sequence", type=str, action="append", help="Protein sequence(s)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="af3_results", help="Output directory for CIF files"
    )
    args = parser.parse_args()

    client = AF3Client(args.server)

    # Check server health
    try:
        h = client.health()
        print(f"Server OK, queue size: {h['queue_size']}")
    except Exception as e:
        print(f"Cannot reach server at {args.server}: {e}")
        sys.exit(1)

    # Collect sequences
    sequences = []
    names = []

    if args.fasta:
        # Simple FASTA parser
        name = None
        seq_lines = []
        with open(args.fasta) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if name is not None:
                        sequences.append("".join(seq_lines))
                        names.append(name)
                    name = line[1:].split()[0]
                    seq_lines = []
                elif line:
                    seq_lines.append(line)
            if name is not None:
                sequences.append("".join(seq_lines))
                names.append(name)

    if args.sequence:
        for i, seq in enumerate(args.sequence):
            sequences.append(seq)
            names.append(f"seq_{len(names) + i}")

    if not sequences:
        # Demo sequences
        sequences = [
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
            "GPAVFVNKECQPKEGNVLALKKIDGTGHYELAEVNPELIGQYIDKIK",
        ]
        names = ["hemoglobin_alpha", "ubiquitin_fragment"]
        print(f"No sequences provided, using {len(sequences)} demo sequences")

    print(f"\nFolding {len(sequences)} sequences:")
    for name, seq in zip(names, sequences):
        print(f"  {name}: {len(seq)} residues")
    print()

    # Fold batch
    results = client.fold_batch(sequences, names=names)

    # Download CIF files
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nResults:")
    for r in results:
        cif_path = output_dir / f"{r.name}.cif"
        client.download_cif(r.job_id, cif_path)
        print(
            f"  {r.name}: score={r.best_ranking_score:.4f}, "
            f"elapsed={r.elapsed_seconds:.1f}s, "
            f"cif={cif_path}"
        )


if __name__ == "__main__":
    main()
