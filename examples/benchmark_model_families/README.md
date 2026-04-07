# Benchmarking Model Families: Generation Quality vs. Scale

How do different protein language model families compare in their ability to
generate realistic sequences, and how does model scaling affect generation
quality at different masking levels?

## Experiment Design

**Setup:**
- 10 sequences randomly sampled from Swiss-Prot (reviewed UniProt, 80–300 aa)
- 4 masking levels: 10%, 25%, 50%, 100%
- 5 random decoding orders per protein (shared across all models)
- Orders are generated up-front; positions are masked according to the order
  tail, and unmasked in that same order during generation

**Models tested (3 families, 6 models):**

| Family | Model | Parameters |
|--------|-------|-----------|
| ESM-C  | ESMC-300M | 300M |
| ESM-C  | ESMC-600M | 600M |
| ESM3   | ESM3-Open | 1.4B |
| DPLM-2 | DPLM2-150M | 150M |
| DPLM-2 | DPLM2-650M | 650M |
| DPLM-2 | DPLM2-3B  | 3B   |

**Metrics:**
- **Sequence recovery**: % identity between generated and original sequence
- **Generation log-likelihood**: per-step log p(sampled token) during ancestral sampling
- **AF3 pLDDT**: AlphaFold 3 predicted local distance difference test (structural confidence)
- **AF3 pTM**: predicted template modeling score
- **TM-score**: structural similarity between generated and original AF3 structures
- **Fold class agreement**: whether generated protein maintains the same broad fold class
  (all-α, all-β, α+β, small/other) as the original, classified from DSSP secondary structure

## Running the Benchmark

### Step 1: Prepare data
```bash
uv run python examples/benchmark_model_families/prepare_data.py
```
Samples sequences from Swiss-Prot, generates decoding orders, creates masked inputs.

### Step 2: Generate sequences
```bash
# Run all models (sequential, slow)
uv run python examples/benchmark_model_families/generate.py --model all --device cuda

# Or run models individually (for parallel execution)
uv run python examples/benchmark_model_families/generate.py --model esmc_300m --device cuda
uv run python examples/benchmark_model_families/generate.py --model "airkingbd/dplm2_3b" --device cuda
```

### Step 3: Fold with AF3 (optional, requires AF3 server)
```bash
# Start AF3 server first (see af3-server/launch.sh)
uv run python examples/benchmark_model_families/fold.py --server http://localhost:8080
```

### Step 4: Analyze results
```bash
uv run python examples/benchmark_model_families/analyze.py
```

## Key Questions

1. **Family comparison**: Do ESM-C, ESM3, and DPLM-2 show systematic differences in
   generation quality? Which family produces the most structurally realistic sequences?

2. **Scaling laws**: Within families that have multiple sizes (ESM-C: 300M/600M,
   DPLM-2: 150M/650M/3B), does bigger always mean better? Does the scaling advantage
   grow or shrink at higher masking levels?

3. **Masking sensitivity**: How gracefully does each model degrade as the masking
   level increases from 10% (mostly inpainting) to 100% (fully unconditional)?

4. **Fold preservation**: At what masking level do models start generating proteins
   that no longer fold into the same structural class as the original?

## Output Structure

```
outputs/
├── ESMC-300M/generation_results.json
├── ESMC-600M/generation_results.json
├── ESM3-Open/generation_results.json
├── DPLM2-150M/generation_results.json
├── DPLM2-650M/generation_results.json
├── DPLM2-3B/generation_results.json
├── fold_results/
│   ├── fold_results.json
│   └── cif_files/
├── metrics.json
└── plots/
    ├── likelihood_trajectories.png
    ├── sequence_identity.png
    ├── mean_step_log_prob.png
    ├── plddt.png
    ├── ptm.png
    ├── tm_score.png
    ├── fold_class_agreement.png
    ├── scaling_sequence_identity.png
    ├── scaling_mean_step_log_prob.png
    ├── scaling_plddt.png
    └── scaling_tm_score.png
```

## Design Notes

### Controlled decoding orders
The 5 decoding orders are generated once per protein and shared across all
models. This controls for the effect of random masking patterns — any
performance difference between models at the same masking level and order
is attributable to the model, not the random mask.

### Order → masking → sampling
An order is a permutation of all maskable positions. For X% masking, the
last X% of positions in the order are masked. During sampling, these
positions are unmasked following that same order (first position in the
tail is revealed first). This means lower masking levels reveal a prefix
of the same trajectory as higher masking levels.

### Fold classification
Rather than using complex tools like Foldclass/Merizo-search, we use a
simple heuristic based on DSSP secondary structure content: classify
structures as all-α (≥40% helix, <10% strand), all-β (<10% helix, ≥30%
strand), α+β (≥15% each), or small/other. This is a coarse approximation
of SCOP classes but sufficient for tracking whether generation preserves
the overall fold topology.
