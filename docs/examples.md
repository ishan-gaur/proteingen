# Examples

All examples live in the `examples/` directory. Run them with `uv run python examples/<script>.py`.

## Unconditional Sampling

The simplest example — generate protein sequences from scratch using ESMC as a masked language model.

```python
from dfm.models import ESMC
from dfm import sample_any_order_ancestral

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

## Stability-Guided Generation

!!! note "Coming soon"
    The `examples/stability_guidance/` directory contains a work-in-progress reimplementation of stability-guided protein generation using the ProteinMPNN stability predictor and ESM3 as the generative model.

<!-- TODO[pi]: finish stability guidance example and document it here -->
