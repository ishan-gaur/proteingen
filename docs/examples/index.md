# Examples

Concrete implementations showing how to compose ProteinGen's [modules](../workflows/index.md) into working pipelines. Each example includes an **Architecture Breakdown** at the top that maps its components to the four library design areas (Data, Models, Sampling, Evaluation) and links to the relevant modules and reference docs.

All examples live in the [`examples/`](https://github.com/ishan-gaur/proteingen/tree/main/examples) directory. Run them with `uv run python examples/path-to-script.py`.

---

## Guided Generation

Full pipelines that combine generative models with predictive models for property-optimized sampling.

- **[Stability-Guided Generation](stability-guided-generation.md)** — ESM3 inverse folding + TAG guidance with a noisy stability predictor (all four modules)
- **[Guided Sampling (TrpB)](trpb-linear-probe.md)** — Train a fitness probe on ESMC embeddings, then DEG-guided sampling

## Sampling

Generate protein sequences using pretrained models.

- **[Unconditional Sampling](unconditional-sampling.md)** — generate sequences from scratch with ESMC (Models + Sampling only)
- **[Autoregressive Generation (ProGen3)](autoregressive-generation.md)** — unconditional protein generation with ProGen3's left-to-right sampling
- **[Structure-Conditioned Sampling](structure-conditioned-sampling.md)** — inverse folding with ESM3 backbone conditioning

## Fine-tuning

Adapt pretrained models to specific protein families.

- **[Sequence-only MLM (EphB1)](finetune-esm3-mlm.md)** — LoRA fine-tune ESM3 on kinase domain homologs (Data + Models)
- **[Inverse Folding (EphB1)](finetune-inverse-folding.md)** — LoRA fine-tune ESM3 with AF3-predicted structures (Data + Models + Evaluation)

## Evaluation

Assess model quality and compare generation strategies.

- **[Likelihood Curves (TrpB)](likelihood-curves.md)** — diagnostic: log p(true token) vs. fraction unmasked (Evaluation only)
- **[Benchmark: Model Families](benchmark-model-families.md)** — compare 6 models across 3 families at 4 masking levels, with AF3 structural validation

## Predictive Modeling

Build lightweight predictors on top of pretrained representations.

- **[PCA Embedding Initialization](pca-embedding-init.md)** — compress ESMC embeddings via PCA for small predictors
