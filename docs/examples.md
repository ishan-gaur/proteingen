# Examples

All examples live in the `examples/` directory. Run them with `uv run python examples/<script>.py`.

## Unconditional Sampling

The simplest example — generate protein sequences from scratch using ESMC as a masked language model.

```python
from proteingen.models import ESMC
from proteingen import sample_any_order_ancestral

model = ESMC().cuda()
initial_x = ["<mask>" * 256 for _ in range(5)]
sequences = sample_any_order_ancestral(model, initial_x)
```

```bash
uv run python examples/unconditional_sampling.py
```

This starts from fully masked sequences and iteratively unmasks positions using ESMC's learned distribution.

## Structure-Conditioned Sampling (ESM3)

Generate sequences conditioned on a known protein backbone structure using ESM3.

```bash
uv run python examples/esm3_structure_conditioned_sampling.py
```

ESM3 accepts atom37-format coordinates as conditioning input. The model's `set_condition_()` method runs the VQ-VAE structure encoder once (expensive), then all subsequent sampling steps use the cached structure tokens.

See [Models → ESM3](models.md) for details on structure conditioning.

## PCA Embedding Initialization

Initialize a small `EmbeddingMLP` predictor using PCA-compressed embeddings from ESMC. This transfers learned amino acid representations from the pretrained model into a lightweight head.

```bash
uv run python examples/pca_embedding_init.py
```

Key idea: ESMC's 960-dim token embeddings encode useful AA similarities, but are too large for a small predictor. `init_embed_from_pretrained_pca` compresses them via PCA and handles vocabulary mapping automatically.

## Training a Probe + Guided Sampling (TrpB)

End-to-end example: train a linear probe on the TrpB fitness landscape using cached ESMC embeddings, then use it for guided sampling with enumeration-based guidance (DEG).

```bash
uv run python examples/trpb_linear_probe.py --device cuda
```

This example demonstrates:

1. Loading a HuggingFace dataset (`SaProtHub/Dataset-TrpB_fitness_landsacpe`)
2. Training a `LinearProbe` on cached embeddings from ESMC
3. Combining the probe with ESMC via DEG for guided generation
4. Evaluating guided vs. unguided samples

## Likelihood Curves (TrpB)

Evaluate how well a model predicts masked amino acids under progressive unmasking, using real TrpB sequences from SaProtHub.

```bash
uv run python examples/trpb_likelihood_curves.py --device cuda --n-sequences 50 --n-time-points 20
```

Produces a plot showing average log p(true token) at masked positions vs fraction unmasked. See the [Likelihood Curves workflow](workflows/likelihood-curves.md) for interpretation and usage.

## Fine-tuning ESM3 on EphB1 (Sequence-only MLM)

Fine-tune ESM3 with LoRA on ~10k EphB1 kinase domain homologs using masked language modeling.

```bash
uv run python examples/finetune_esm3/finetune_esm3_ephb1.py --device cuda --amp --epochs 5
```

This trains the model to predict randomly masked amino acids from surrounding sequence context. After 5 epochs on ~10k sequences: **loss 1.80 → 1.60, perplexity 6.04 → 4.96**.

See the [Fine-tuning workflow](workflows/finetune-generative.md) for details.

## Fine-tuning ESM3 as Inverse Folding Model (EphB1)

The full pipeline: fold MSA sequences with AF3, then fine-tune ESM3 to predict sequence from structure.

### Step 1: Fold MSA sequences

```bash
# Start AF3 server
cd af3-server && sbatch launch.sh

# Fold all sequences (runs ~80h for 10k sequences, saves incrementally)
uv run python examples/finetune_esm3/fold_msa_domains.py \
    --server-url http://localhost:8080
```

### Step 2: Train inverse folding model

```bash
uv run python examples/finetune_esm3/finetune_inverse_folding.py \
    --device cuda --amp --epochs 5
```

Evaluates both structure-conditioned and sequence-only likelihood curves at each epoch. Results on ~9.2k EphB1 structures:

| Epoch | Loss  | PPL  | Struct log p (t=0) | Seq-only log p (t=0) |
|-------|-------|------|--------------------|---------------------|
| 0     | —     | —    | -2.075             | -2.955              |
| 5     | 0.572 | 1.77 | -0.798             | -2.953              |

The structure-conditioned model improves from -2.075 to -0.798 while sequence-only stays flat at -2.95, confirming the model learns to use structure information.

![Inverse folding likelihood curves](assets/images/inverse_folding_likelihood_curves.png)

See the [Fine-tuning workflow](workflows/finetune-generative.md) and [MSA → Dataset workflow](workflows/msa-to-dataset.md) for the full walkthrough.

## Stability-Guided Generation

!!! note "Coming soon"
    The `examples/stability_guidance/` directory contains a work-in-progress reimplementation of stability-guided protein generation using the ProteinMPNN stability predictor and ESM3 as the generative model.

<!-- TODO[pi]: finish stability guidance example and document it here -->
