# DPLM-2 — Design Notes

ByteDance's discrete diffusion protein language model (DPLM-2), wrapped as a `GenerativeModelWithEmbedding`.

**Paper**: [DPLM-2: A Multimodal Diffusion Protein Language Model](https://arxiv.org/abs/2410.13782) (ICLR'25)
**Repo**: [bytedance/dplm](https://github.com/bytedance/dplm) — commit `8a2e15e`
**Weights**: HuggingFace hub under `airkingbd/`

## Dependencies

- [generative_modeling.md](../../generative_modeling.md) — `GenerativeModelWithEmbedding`, `MaskedModelLogitFormatter`
- `transformers` — `AutoModelForMaskedLM`, `AutoConfig`, `EsmTokenizer`

## Used By

- `tests/test_dplm2.py` — 20 tests (tokenizer, forward, embedding, regression)

## Architecture

DPLM-2 is a modified ESM-2 architecture extended for multimodal (sequence + structure) diffusion. In proteingen, we load it via HuggingFace's `EsmForMaskedLM` for sequence-only use.

### Model variants

| Checkpoint | Params | Hidden | Layers | Heads |
|---|---|---|---|---|
| `airkingbd/dplm2_150m` | 150M | 640 | 30 | 20 |
| `airkingbd/dplm2_650m` | 650M | 1280 | 33 | 20 |
| `airkingbd/dplm2_3b` | 3B | 2560 | 36 | 40 |

### Vocabulary (8229 tokens)

- **0–2**: `<cls_aa>`, `<pad>`, `<eos_aa>` (AA special tokens)
- **3**: `<unk_aa>`
- **4–23**: 20 standard amino acids (L, A, G, V, S, E, R, T, I, D, P, K, Q, N, F, Y, M, H, W, C)
- **24–28**: non-standard AAs (X, B, U, Z, O)
- **29–31**: `.`, `-`, `<null_1>`
- **32**: `<mask_aa>` — mask token for AA diffusion
- **33–34**: `<cls_struct>`, `<eos_struct>` — structure BOS/EOS
- **35**: `<unk_struct>`
- **36–8227**: 8192 structure codebook tokens
- **8228**: `<mask_struct>` — mask token for structure diffusion

### Forward path

```
input_ids → EsmEmbeddings (word embed + token_dropout) → EsmEncoder (rotary attention × N layers) → EsmLMHead → logits (8229)
```

### Embedding path (for TAG guidance)

```
ohe_SPT → soft matmul @ word_embeddings.weight → token_dropout scaling → encoder → last_hidden_state
          → lm_head → logits
```

## DPLM2Tokenizer

Custom tokenizer wrapping HF `EsmTokenizer` with DPLM2-specific special token assignments. The HF-loaded tokenizer has incorrect special token IDs (generic `<mask>` at 8232 instead of `<mask_aa>` at 32), so the wrapper overrides `mask_token_id`, `cls_token_id`, `eos_token_id`, and `added_tokens_decoder`.

Also filters `vocab` to exclude generic HF special tokens (IDs ≥ 8229) that aren't in the model's embedding table.

## Gotchas

- **`tie_word_embeddings` must be False** — the HF config says `True`, but the model was trained with untied weights. Loading with `True` loses the separate decoder weights (max logit diff ~0.97). Our wrapper overrides this in the config before loading.

- **Token dropout runs in eval mode** — `EsmEmbeddings` always scales embeddings by `(1 - 0.12) / (1 - mask_ratio_observed)`, even during inference. For inputs without mask tokens, this is a uniform 0.88× scaling. The `differentiable_embedding` path must replicate this exactly, including zeroing out mask token positions before scaling.

- **Standard `EsmForMaskedLM` vs upstream `EsmForDPLM2`** — we load with HuggingFace's standard `EsmForMaskedLM`, not the upstream `EsmForDPLM2` which has modified rotary embeddings for dual-modality position encoding. For sequence-only inputs (all AA tokens), the two are numerically identical (~1e-6 diff). The modified rotary embedding only diverges when both struct AND AA tokens are present in the input.

- **Structure conditioning not yet supported** — DPLM-2's joint sequence+structure generation concatenates `[struct_tokens | aa_tokens]` as input. This requires the structure VQ-VAE tokenizer (`airkingbd/struct_tokenizer`) and the modified rotary embeddings. Deferred to future work.

- **HF tokenizer warning** — loading `EsmTokenizer.from_pretrained("airkingbd/dplm2_*")` prints a warning about mismatched tokenizer class (`DPLM2Tokenizer` vs `EsmTokenizer`). This is harmless — the vocab file loads correctly.

- **Regression tests pinned to upstream commit `8a2e15e`** — reference logits in `test_dplm2.py` were generated from `bytedance/dplm` at that commit using `EsmForDPLM2.from_pretrained` with explicit `type_ids`. If upstream changes model weights or architecture, the references need updating.
