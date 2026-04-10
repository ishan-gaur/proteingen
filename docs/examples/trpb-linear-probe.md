# Training a Probe + Guided Sampling (TrpB)

??? abstract "Architecture Breakdown"
    **Data:** TrpB fitness landscape from SaProtHub (HuggingFace dataset) with continuous fitness labels → [data](../reference/data.md).

    **Models:**

    - **ESMC** — pretrained generative model (no fine-tuning) → [generative_modeling](../reference/generative_modeling.md)
    - **LinearProbe** on cached ESMC embeddings — trained as the predictive model → [predictive_modeling](../reference/predictive_modeling.md). Uses `precompute_embeddings` for fast training → [Training Predictors](../workflows/training-predictors.md)
    - **DEG**(ESMC, probe) — combines them via enumeration-based guidance → [guide](../reference/guide.md)

    **Sampling:** `sample` (discrete-time ancestral) — DEG automatically passes position info → [sampling](../reference/sampling.md)

    **Evaluation:** Guided vs. unguided fitness comparison → [evaluation](../reference/evaluation.md)

End-to-end example: train a linear probe on the TrpB fitness landscape using cached ESMC embeddings, then use it for guided sampling with enumeration-based guidance (DEG).

## Quick Start

```bash
uv run python examples/trpb_linear_probe.py --device cuda
```

## What This Demonstrates

1. **Loading a HuggingFace dataset** (`SaProtHub/Dataset-TrpB_fitness_landsacpe`)
2. **Training a `LinearProbe`** on cached embeddings from ESMC
3. **Combining the probe with ESMC** via DEG for guided generation
4. **Evaluating guided vs. unguided samples**

## How Guidance Works

Discrete Expert-based Guidance (DEG) uses the probe as an "expert" to bias the generative model's sampling toward sequences predicted to have high fitness. At each unmasking step, the model's token probabilities are reweighted by the probe's predictions.

See the [ProteinGuide workflow](../workflows/protein-guide.md) for the full guided sampling framework.

**Source**: [`examples/trpb_linear_probe.py`](https://github.com/ishan-gaur/protstar/blob/main/examples/trpb_linear_probe.py)
