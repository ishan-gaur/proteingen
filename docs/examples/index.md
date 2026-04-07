# Examples

All examples live in the [`examples/`](https://github.com/ishan-gaur/proteingen/tree/main/examples) directory. Run them with `uv run python examples/<script>.py`.

---

## Generation

Generate protein sequences using masked language models with different conditioning strategies.

- **[Unconditional Sampling](unconditional-sampling.md)** — generate sequences from scratch with ESMC
- **[Structure-Conditioned Sampling](structure-conditioned-sampling.md)** — inverse folding with ESM3 backbone conditioning
- **[Guided Sampling (TrpB)](trpb-linear-probe.md)** — train a fitness probe, then use DEG guidance to steer generation

## Evaluation

Assess model quality and compare generation strategies.

- **[Likelihood Curves (TrpB)](likelihood-curves.md)** — diagnostic: log p(true token) vs. fraction unmasked
- **[Benchmark: Model Families](benchmark-model-families.md)** — compare 6 models across 3 families (ESMC, ESM3, DPLM2) at 4 masking levels, with AF3 structure validation

## Fine-tuning

Adapt pretrained models to specific protein families or tasks.

- **[Sequence-only MLM (EphB1)](finetune-esm3-mlm.md)** — LoRA fine-tune ESM3 on kinase domain homologs
- **[Inverse Folding (EphB1)](finetune-inverse-folding.md)** — fold MSA with AF3, then train ESM3 to predict sequence from structure

## Predictive Modeling

Build lightweight predictors on top of pretrained representations.

- **[PCA Embedding Initialization](pca-embedding-init.md)** — compress ESMC embeddings via PCA for small predictors

## Coming Soon

!!! note "Stability-Guided Generation"
    The `examples/stability_guidance/` directory contains a work-in-progress reimplementation of stability-guided protein generation using the ProteinMPNN stability predictor and ESM3 as the generative model.
