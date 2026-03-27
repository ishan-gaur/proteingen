"""Fold EphB1 MSA domain sequences with AF3 and pre-compute ESM3 structure tokens.

Submits sequences to the AF3 server, downloads CIF files, converts to atom37,
encodes structure tokens via ESM3's VQ-VAE, and saves everything incrementally
to a .pt checkpoint.

The output file contains:
    {
        "sequences": [str, ...],           # raw sequences
        "structure_tokens": [Tensor, ...],  # (L+2,) per sequence
        "coordinates": [Tensor, ...],       # (L+2, 37, 3) per sequence
        "ranking_scores": [float, ...],     # AF3 confidence scores
    }

Usage:
    # Start AF3 server first: sbatch af3-server/launch.sh
    uv run python examples/finetune_esm3/fold_msa_domains.py --server-url http://localhost:8080
    uv run python examples/finetune_esm3/fold_msa_domains.py --server-url http://localhost:8080 --max-sequences 100
"""

import argparse
import sys
import tempfile
import time
import warnings
from pathlib import Path

import biotite.structure.io.pdb as pdb_io
import biotite.structure.io.pdbx as pdbx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "af3-server"))
from client import AF3Client

from esm.utils.structure.protein_chain import ProteinChain

from proteingen.data import aligned_sequences_to_raw, read_fasta
from proteingen.models.esm import ESM3

DATA_DIR = Path(__file__).parent

warnings.filterwarnings("ignore", category=UserWarning, module="biotite")


def cif_to_atom37(cif_path: str) -> torch.Tensor:
    """Convert an AF3 output CIF to atom37 coordinates (L, 37, 3)."""
    f = pdbx.CIFFile.read(cif_path)
    atoms = pdbx.get_structure(f, model=1, extra_fields=["b_factor", "occupancy"])

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        pdb_file = pdb_io.PDBFile()
        pdb_file.set_structure(atoms)
        pdb_file.write(tmp.name)
        tmp_path = tmp.name

    pc = ProteinChain.from_pdb(tmp_path, chain_id="A")
    Path(tmp_path).unlink()
    return torch.from_numpy(pc.atom37_positions).float()


def main():
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="Fold EphB1 MSA domains with AF3 and pre-compute structure tokens"
    )
    parser.add_argument(
        "--server-url", default="http://localhost:8080", help="AF3 server URL"
    )
    parser.add_argument(
        "--min-length", type=int, default=200, help="Minimum sequence length"
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Cap number of sequences to fold",
    )
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "ephb1_structures.pt"),
        help="Output .pt file path",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Save checkpoint every N sequences",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file",
    )
    args = parser.parse_args()

    # ── Load sequences ────────────────────────────────────────────────────
    entries = read_fasta(str(DATA_DIR / "EphB1_MSA.fasta"))
    aligned = [seq for _, seq in entries]
    raw = aligned_sequences_to_raw(aligned)
    raw = [s for s in raw if len(s) >= args.min_length]
    print(f"Loaded {len(raw)} sequences (>= {args.min_length} residues)")

    if args.max_sequences is not None:
        raw = raw[: args.max_sequences]
        print(f"Capped to {len(raw)} sequences")

    # ── Resume from checkpoint ────────────────────────────────────────────
    done_sequences: set[str] = set()
    data = {
        "sequences": [],
        "structure_tokens": [],
        "coordinates": [],
        "ranking_scores": [],
    }
    if args.resume and Path(args.output).exists():
        data = torch.load(args.output, weights_only=False)
        done_sequences = set(data["sequences"])
        print(f"Resuming: {len(done_sequences)} sequences already folded")

    # ── Load ESM3 for VQ-VAE encoding ─────────────────────────────────────
    print("Loading ESM3 for structure token encoding...")
    model = ESM3("esm3-open")
    # Trigger VQ-VAE load by encoding a dummy structure
    dummy_coords = torch.randn(10, 37, 3)
    model.preprocess_observations({"coords_RAX": dummy_coords})
    print("ESM3 VQ-VAE ready")

    # ── Connect to AF3 server ─────────────────────────────────────────────
    client = AF3Client(args.server_url, poll_interval=5.0, timeout=600.0)
    health = client.health()
    print(f"AF3 server: {health}")

    # ── Fold sequences ────────────────────────────────────────────────────
    to_fold = [s for s in raw if s not in done_sequences]
    print(f"\nFolding {len(to_fold)} sequences...")
    t0 = time.time()

    for i, seq in enumerate(to_fold):
        name = f"ephb1_{len(data['sequences']):05d}"
        try:
            result = client.fold(seq, name=name)

            # Download and convert CIF → atom37
            cif_path = f"{result.output_dir.replace('/app/af_output', '/data/af3_server_output')}/{result.name}_model.cif"
            coords = cif_to_atom37(cif_path)
            assert coords.shape[0] == len(seq), (
                f"Length mismatch: coords {coords.shape[0]} vs seq {len(seq)}"
            )

            # Encode structure tokens
            obs = model.preprocess_observations({"coords_RAX": coords})

            data["sequences"].append(seq)
            data["structure_tokens"].append(obs["structure_tokens"])
            data["coordinates"].append(obs["coordinates"])
            data["ranking_scores"].append(result.best_ranking_score)

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 3600
            remaining = (len(to_fold) - i - 1) / rate * 3600 if rate > 0 else 0
            print(
                f"[{i + 1}/{len(to_fold)}] {name}: len={len(seq)}, "
                f"score={result.best_ranking_score:.3f}, "
                f"fold_time={result.elapsed_seconds:.0f}s, "
                f"rate={rate:.0f}/hr, ETA={remaining / 3600:.1f}h"
            )

        except Exception as e:
            print(f"[{i + 1}/{len(to_fold)}] FAILED {name} (len={len(seq)}): {e}")
            continue

        # Incremental save
        if (i + 1) % args.save_every == 0:
            torch.save(data, args.output)
            print(f"  Saved checkpoint: {len(data['sequences'])} sequences")

    # ── Final save ────────────────────────────────────────────────────────
    torch.save(data, args.output)
    elapsed = time.time() - t0
    print(f"\nDone: {len(data['sequences'])} sequences folded in {elapsed / 3600:.1f}h")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
