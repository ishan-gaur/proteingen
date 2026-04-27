# Tests — Agent Notes

Pytest test suite for proteingen. Run with `uv run python -m pytest tests/ -v`. [×1]

## Current Status

320 tests, all passing (1 skipped: SolubleMPNN checkpoint).

## Test Files

| File | Count | Covers |
|------|-------|--------|
| `test_generative_model.py` | — | GenerativeModel, GenerativeModelWithEmbedding |
| `test_logit_formatter.py` | 24 | MaskedModelLogitFormatter with ESM and BERT tokenizers |
| `test_embedding_mlp.py` | 49 | OneHotMLP, EmbeddingMLP, PairwiseLinearModel: construction, forward, gradients, get_log_probs, grad_log_prob |
| `test_pca_embed_init.py` | 21 | pca_embed_init, init_embed_from_pretrained_pca, ESMC integration |
| `test_esm.py` | — | ESMC: construction, embedding, log probs |
| `test_esm3.py` | 21 | ESM3: construction, embed vs forward match, gradients, batching, structure conditioning (8 tests), temperature |
| `test_esmc_lora.py` | 11 | ESMC + LoRA: apply, forward, save/load round-trip |
| `test_lora.py` | 23 | General LoRA + checkpointing tests |
| `test_sampling.py` | — | Sampling algorithms |
| `test_guidance_data.py` | — | GuidanceDataset, noise schedules |
| `test_tag_projection.py` | — | TAG GuidanceProjection: prepare-step token/position mapping, Taylor delta math |

## Testing Patterns

- **Mock tokenizer**: `SimpleNamespace(vocab_size=N, pad_token_id=M)` for predictive model tests
- **ESM tests**: some tests require `esm` package and model downloads — may be slow or skip on CI
- **Test categories for new models** (see `add-generative-model` skill):
  - Construction & basic forward
  - Embedding path (shape, matches forward, gradient flow)
  - Log probabilities (valid, temperature scaling)
  - Batching consistency
  - Conditioning (if applicable)
  - LoRA (if applicable)
  - Checkpointing (if applicable)

## Gotchas

- Run tests with `uv run python -m pytest` (not `uv run pytest` — pytest may not be on PATH directly)
- ESM3 structure conditioning tests need the VQ-VAE encoder — triggers lazy loading, slow first run
- `test_protein_mpnn.py::test_soluble_mpnn` checks Foundry checkpoint path existence before constructing `ProteinMPNN("solublempnn")`; if the file is missing, the test skips even though the wrapper can lazy-download at runtime.
