# MSA → Sequence + Structure Dataset

Turn a multiple sequence alignment (MSA) into a training-ready dataset of sequences paired with predicted structures.

## Overview

```
MSA FASTA → strip gaps → filter by length → fold with AF3 → encode structures → .pt dataset
```

Many protein engineering projects start with a family of homologous sequences. This workflow converts that raw MSA into a dataset suitable for fine-tuning generative models — either sequence-only or with per-sequence structure conditioning.

---

## Step 1: Load and clean the MSA

Use `read_fasta` and `aligned_sequences_to_raw` to strip gap characters from aligned sequences:

```python
from protstar.data import read_fasta, aligned_sequences_to_raw

entries = read_fasta("EphB1_MSA.fasta")
aligned = [seq for _, seq in entries]
raw = aligned_sequences_to_raw(aligned)  # strips '-' and '.' characters

# Filter by minimum length
raw = [s for s in raw if len(s) >= 200]
print(f"{len(raw)} sequences after filtering")
```

!!! tip "MSA sequences are domain fragments"
    Sequences from MSA tools (jackhmmer, MMseqs2) are aligned to a query region — they contain only the homologous domain, not the full-length protein. After gap removal, each sequence is the domain fragment from that organism. These are foldable as isolated domains.

---

## Step 2: Fold with AF3 (optional)

For structure-conditioned training, each sequence needs its own predicted structure. The AF3 server folds sequences via a persistent REST API.

### Start the server

```bash
# From the af3-server/ directory
sbatch launch.sh
# Wait for "Server ready." in the log
tail -f af3-server.*.out
```

### Fold sequences

The `fold_msa_domains.py` script handles the full pipeline: fold → download CIF → convert to atom37 → encode ESM3 structure tokens → save incrementally.

```bash
uv run python examples/finetune_esm3/fold_msa_domains.py \
    --server-url http://localhost:8080 \
    --output structures.pt \
    --save-every 50
```

This saves a `.pt` checkpoint every 50 sequences so you can start training before all structures are ready. Use `--resume` to continue after interruption.

### What's in the output file

`torch.save` serializes a dict of lists — each entry corresponds to one sequence:

```python
{
    "sequences": [str, ...],              # raw amino acid sequences
    "structure_tokens": [Tensor, ...],     # (L+2,) ESM3 VQ-VAE tokens per sequence
    "coordinates": [Tensor, ...],          # (L+2, 37, 3) atom37 coords per sequence
    "ranking_scores": [float, ...],        # AF3 confidence score per sequence
}
```

Structure tokens and coordinates include BOS/EOS padding (hence L+2). Tensors are variable-length across sequences since domain lengths differ.

### Timing expectations

On an RTX 6000 Ada (49GB), AF3 folds ~120-135 sequences per hour for domains of 200-300 residues. A 10k-sequence MSA takes ~80 hours. The incremental saving means you can start training on partial data.

---

## Step 3: Sequence-only dataset (no folding needed)

If you only need sequences (no structure conditioning), skip the AF3 step and create a `ProteinDataset` directly:

```python
from protstar.data import ProteinDataset, read_fasta, aligned_sequences_to_raw

entries = read_fasta("my_msa.fasta")
raw = aligned_sequences_to_raw([seq for _, seq in entries])
raw = [s for s in raw if len(s) >= 200]

dataset = ProteinDataset(sequences=raw)
```

The `ProteinDataset` stores raw sequences. All model-specific transforms (tokenization, noising, padding) happen in the collator — see [Fine-tuning workflow](finetune-generative.md) for how to set up training.
