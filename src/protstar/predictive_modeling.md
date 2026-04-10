# Predictive Modeling — Design Notes

PredictiveModel ABC, template models (LinearProbe, OneHotMLP, EmbeddingMLP, PairwiseLinearModel), binary logit functions, and PCA embedding initialization.

## Dependencies

- [probability_model.md](probability_model.md) — `PredictiveModel(ProbabilityModel)`, inherits conditioning/temp/get_log_probs
- [generative_modeling.md](generative_modeling.md) — `LinearProbe` wraps `GenerativeModelWithEmbedding`

## Used By

- [guide.md](guide.md) — TAG/DEG call `get_log_probs` and `grad_log_prob`
- [models/AGENTS.md](models/AGENTS.md) — `PreTrainedStabilityPredictor(PredictiveModel)`
- Examples: `trpb_linear_probe.py`, `pca_embedding_init.py`, `stability_guidance/main.py`
- Tests: `test_embedding_mlp.py` (49), `test_pca_embed_init.py` (21), `test_tag_projection.py`

## PredictiveModel

**ABC** — user subclasses directly (no `model` arg like GenerativeModel).

### Constructor

`__init__(tokenizer)` — sets `self.tokenizer`, `self.target = None`, `self._ohe = None`.

### Abstract methods (MUST implement)

| Method | Signature | Notes |
|--------|-----------|-------|
| `forward` | `(ohe_seq_SPT, **kwargs) -> Any` | Takes OHE features (not token IDs). Must be differentiable for TAG. |
| `format_raw_to_logits` | `(raw_output, seq_SPT, **kwargs) -> FloatTensor(B, 2)` | Must return binary logits `[false_logit, true_logit]`. Use the binary logit helper functions. |

### Concrete methods

| Method | Behavior |
|--------|----------|
| `get_log_probs(seq_SP)` | Token IDs → OHE → forward → format_raw_to_logits → log_softmax → `[:, 1]` → scalar log p(target\|x) |
| `predict(seq_SP)` | Token IDs → OHE → forward → raw output (no binary conversion). For training/scoring. |
| `grad_log_prob(seq_SP)` | Runs `get_log_probs`, backprops, returns `self._ohe.grad` shape `(B, P, K)`. This is what TAG calls. |
| `tokens_to_ohe(seq_SP)` | Maps token IDs to OHE features using `token_ohe_basis()` |
| `token_ohe_basis()` | Returns `(T, K)` matrix. Default: identity. Override for reduced OHE space. |
| Target management | `set_target_()`, `set_target()`, `with_target()` (context manager with revert) |

### `get_log_probs` Pipeline

```
get_log_probs(seq_SP)
    → tokens_to_ohe(seq_SP) → ohe_SPK (requires_grad=True, stashed as self._ohe)
    → super().get_log_probs(ohe):
        → forward(ohe) → raw output
        → format_raw_to_logits(raw, ohe) → (B, 2) binary logits
        → log_softmax(logits / temp) → (B, 2)
    → [:, 1] → (B,) log p(target | x)
```

### Token OHE Basis

- Default `token_ohe_basis()` is `torch.eye(vocab_size)` — each token maps to its own one-hot vector (`K = T`)
- Override when tokens should map to a reduced space (e.g. stability predictor maps `<mask>` to all-zero OHE row)
- `grad_log_prob` returns gradients in feature space `(B, P, K)` — not always `(B, P, vocab_size)`

## Structure Conditioning

Structure-conditioned predictive models (e.g. `PreTrainedStabilityPredictor`) use `set_condition_()` / `conditioned_on()` inherited from `ProbabilityModel`. When adding a new structure-conditioned predictor, see [models/utils.md](models/utils.md) for the two-layer API (`PDBStructure` → `atom_array_to_encoding`) and the pattern for `condition_from_structure()` helpers. The PMPNN implementation in [models/mpnn/mpnn.md](models/mpnn/mpnn.md) is the reference for generative models; the same pattern applies to predictive models.

### Maintenance

If changes are made to `PredictiveModel`'s interface (abstract methods, binary logit pattern, OHE basis), update the `add-predictive-model` skill (`.agents/skills/add-predictive-model/SKILL.md`) to reflect the new contract.

## Gotchas

- **Target must be set** before `get_log_probs` — asserts `self.target is not None`
- **`format_raw_to_logits` must return `(B, 2)`** — asserted in `get_log_probs`
- **Tokenizer mismatch**: predictor's tokenizer (used for OHE) may differ from gen model's tokenizer. TAG's `GuidanceProjection` handles this mapping — see [guide.md](guide.md).
- **`forward` takes OHE, not token IDs** — all template models receive OHE from `get_log_probs`. The matmul embedding in EmbeddingMLP (`ohe @ embed.weight`) is differentiable for TAG.

## Binary Logit Functions

Standalone functions for converting raw predictions to `(B, 2)` binary logits. Call these from your `format_raw_to_logits`.

| Function | Input | Use case |
|----------|-------|----------|
| `categorical_binary_logits(logits_BC, target_class)` | Multi-class logits | Classification (target class vs rest) |
| `binary_logits(logit_B, target)` | Single logit | Binary classification |
| `point_estimate_binary_logits(pred_B, threshold, k)` | Scalar prediction | Thresholding a regression output. **Steep sigmoid (large k) saturates gradient** — use DEG not TAG. |
| `gaussian_binary_logits(mu_B, log_var_B, threshold)` | Gaussian params | P(Y > threshold) via CDF. Differentiable through both mean and variance — works with TAG. |

## LinearProbe

`LinearProbe(PredictiveModel, ABC)` — frozen embed_model + trainable linear head.

### Constructor

`__init__(embed_model: GenerativeModelWithEmbedding, output_dim, pooling_fn=None, freeze_embed_model=True)`

- Default pooling: mean over non-padding positions
- `pooling_fn` takes **two args**: `(embeddings_SPD, seq_SP)` — seq_SP needed for masking special tokens
- Set `freeze_embed_model=False` when using LoRA on the embed_model

### Key methods

| Method | Behavior |
|--------|----------|
| `forward(ohe_SPT)` | `differentiable_embedding(ohe)` → pool → linear |
| `precompute_embeddings(sequences, batch_size, device)` | Calls `embed_model.embed()` → pooled embeddings for training |

### Checkpointing

- `save()` writes `head.pt` + delegates to `embed_model.save(path/embed_model/)`
- `from_checkpoint()` reconstructs, loads LoRA onto embed_model if present, loads head

## OneHotMLP

`OneHotMLP(PredictiveModel, ABC)` — flattened OHE → MLP.

- Takes `tokenizer`, `sequence_length`, `model_dim`, `n_layers`, `output_dim`, `dropout`
- Forward: flatten `(S, P, T)` → `(S, P*T)` → MLP layers → `(S, output_dim)`
- User implements `format_raw_to_logits` using binary logit functions

## EmbeddingMLP

`EmbeddingMLP(PredictiveModel, ABC)` — differentiable embedding lookup → MLP.

- `forward`: `ohe @ self.embed.weight` → flatten → MLP → `(S, output_dim)`
- The `ohe @` matmul makes embedding lookup differentiable for TAG gradient flow
- `padding_idx` defaults to `tokenizer.pad_token_id`

### `init_embed_from_pretrained_pca(source, source_vocab, target_vocab)`

Post-construction method that initializes embeddings from PCA of a pretrained model:

- Token matching by string key — shared tokens = intersection of vocab keys
- PCA computed ONLY over shared tokens (special tokens excluded from centering/SVD)
- Uses `torch.linalg.svd` (not numpy)
- Zeroes padding row after copy
- **Design decision**: post-construction method, NOT constructor param — avoids redundant shape args

### PCA Math Notes

- ESMC `embed`: `Embedding(64, 960)` — 64 tokens (33 real + 31 alignment padding)
- First 20 PCs of 20 AA embeddings capture ~100% variance (rank 19 after centering)
- `pca_embed_init()` is an internal helper — not exported from `protstar.__init__`

## PairwiseLinearModel

`PairwiseLinearModel(PredictiveModel, ABC)` — linear model on single + pairwise OHE features.

- Computes pairwise outer product of OHE features, takes upper triangle
- Single linear layer on the pairwise features
- Quadratic in sequence_length × vocab_size

## Testing Notes

- Mock tokenizer for tests: `SimpleNamespace(vocab_size=N, pad_token_id=M)`
- Exports from `protstar.__init__`: `OneHotMLP`, `EmbeddingMLP`, `PairwiseLinearModel` (not `pca_embed_init`)
