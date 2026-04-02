# Examples — Agent Notes

End-to-end usage examples demonstrating proteingen workflows.

## Examples

| File | Description | Status |
|------|-------------|--------|
| `unconditional_sampling.py` | ESMC unconditional masked diffusion sampling | Working |
| `stability_guidance/main.py` | Stability-guided protein design using TAG + flow-matching | Working |
| `stability_guidance/compare_legacy_sampler.py` | Comparison with original demo | **Stale** — uses `from dfm.*` imports |
| `original_stability_guidance/` | Original working example (local utils, not proteingen abstractions) | Working (legacy) |
| `pca_embedding_init.py` | ESMC PCA → EmbeddingMLP initialization | Working |
| `trpb_linear_probe.py` | TrpB fitness predictor with ESMC LinearProbe | Working |
| `conditional_scoring.py` | Structure-conditioned scoring | Working |
| `esm3_structure_conditioned_sampling.py` | ESM3 structure-conditioned generation | Working |
| `pbrr_walkthrough/` | PbrR 3-step guided design walkthrough (LoRA, predictors, DEG guidance) | Working |

## Stability Demo Reimplementation Notes

`stability_guidance/main.py` mirrors `original_stability_guidance/example_usage.py` settings:
- Flow-matching Euler sampling (`dt=0.01`, `x1_temp=0.1`, `num_samples=100`, `batch_size=50`)
- Denoising logits apply original-style token constraints: fixed CLS/EOS, inner positions restricted to canonical 20 AAs
- Guided run uses TAG with projection-based cross-tokenizer path

## Cached Data

- Varpos embeddings: `data/trpb_embeddings/{esmc_300m,esmc_600m}_varpos/{train,valid,test}.pt`
- TrpB dataset: HuggingFace `SaProtHub/Dataset-TrpB_fitness_landsacpe`
