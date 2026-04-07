# Stability-Guided Generation

Generate thermodynamically stable protein sequences using ESM3 inverse folding guided by a learned stability predictor.

## Quick Start

```bash
uv run python examples/stability_guidance/main.py
```

Requires a GPU with enough memory for ESM3 + the stability classifier.

## What It Does

Runs **unguided vs. guided** ESM3 inverse folding on the Rocklin cluster 146 miniprotein (5KPH), then evaluates predicted ΔΔG with an oracle stability model. Produces:

1. A **ddG histogram** comparing guided vs. unguided sequences
2. A **sequence identity vs. ddG scatter** showing the stability–diversity tradeoff

## How Guidance Works

The example uses **TAG (Twisted Annealed Guidance)** to combine:

- **ESM3** as the generative backbone — structure-conditioned inverse folding via `set_condition_({"coords_RAX": ...})`
- **PreTrainedStabilityPredictor** as the classifier — a ProteinMPNN-based stability predictor trained on Rocklin miniprotein data

At each denoising step, TAG reweights ESM3's predicted token distribution using the classifier's stability signal. The `guide_temp` parameter controls guidance strength (lower = stronger).

```python
from proteingen.models.esm import ESM3
from proteingen.models.rocklin_ddg.stability_predictor import (
    PreTrainedStabilityPredictor,
)
from proteingen.guide import TAG
from proteingen.sampling import sample_linear_interpolation

esm_model = ESM3().cuda()
esm_model.set_condition_({"coords_RAX": coords})

classifier = PreTrainedStabilityPredictor(checkpoint_path).cuda()
classifier.set_condition_(stability_cond)
classifier.set_temp_(0.03)  # lower = stronger guidance

guided_model = TAG(esm_model, classifier)
sequences = sample_linear_interpolation(guided_model, init_tokens, n_steps=100)
```

## Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `gen_temp` | 1.0 | ESM3 generation temperature |
| `guide_temp` | 0.03 | Classifier guidance strength (lower = stronger) |
| `n_steps` | 100 | Number of denoising steps |
| `num_samples` | 100 | Sequences to generate per condition |

## Related

- [ProteinGuide workflow](../workflows/protein-guide.md) — the guided sampling framework
- [Models → Rocklin Stability](../models.md) — the stability predictor architecture

**Source**: [`examples/stability_guidance/`](https://github.com/ishan-gaur/proteingen/blob/main/examples/stability_guidance/)
