# predictive_modeling

This module contains the `PredictiveModel` ABC, template model classes (`LinearProbe`, `OneHotMLP`, `EmbeddingMLP`, `PairwiseLinearModel`), binary logit helper functions, and PCA embedding initialization.

## PredictiveModel

An ABC extending `ProbabilityModel` for models that answer "what is log p(target | sequence)?". Uses a **binary logit pattern**: `format_raw_to_logits` must return `(B, 2)` logits `[false_logit, true_logit]`, and `get_log_probs` extracts the true_logit after log-softmax.

### Abstract methods

| Method | Signature | Notes |
|--------|-----------|-------|
| `forward` | `(ohe_seq_SPT, **kwargs) → Any` | Takes OHE features (not token IDs). Must be differentiable for TAG. |
| `format_raw_to_logits` | `(raw_output, seq_SPT, **kwargs) → FloatTensor(B, 2)` | Must return binary logits. Use the binary logit helper functions below. |

### The `get_log_probs` pipeline

`PredictiveModel` overrides the base pipeline to add OHE creation and binary extraction:

```
get_log_probs(seq_SP)
    → tokens_to_ohe(seq_SP) → ohe_SPK (requires_grad=True, stashed as self._ohe)
    → forward(ohe) → raw output
    → format_raw_to_logits(raw, ohe) → (B, 2) binary logits
    → log_softmax(logits / temp) → (B, 2)
    → [:, 1] → (B,) log p(target | x)
```

### Target management

A target must be set before calling `get_log_probs`:

```python
model.set_target_(True)               # in-place
model = model.set_target(True)         # chainable
with model.with_target(True):          # context manager
    log_prob = model.get_log_probs(x)
```

### Gradient access (for TAG)

```python
grad = model.grad_log_prob(seq_SP)  # ∂log p(target|x) / ∂OHE, shape (B, P, K)
```

This runs `get_log_probs` with gradient tracking, backprops through the binary logits, and returns `self._ohe.grad`. This is the core computation that TAG uses to compute guidance deltas.

### Token OHE basis

- Default `token_ohe_basis()` returns `torch.eye(vocab_size)` — each token maps to its own one-hot vector (`K = T`)
- Override when tokens should map to a reduced space (e.g. the stability predictor maps `<mask>` to an all-zero OHE row to preserve original PMPNN masking semantics)
- `grad_log_prob` returns gradients in feature space `(B, P, K)`, which is not always `(B, P, vocab_size)`

## Binary logit functions

Standalone functions for converting raw predictions to `(B, 2)` binary logits. Call these from your `format_raw_to_logits`:

| Function | Input | Use case |
|----------|-------|----------|
| `categorical_binary_logits(logits_BC, target_class)` | Multi-class logits | Classification (target class vs rest) |
| `binary_logits(logit_B, target)` | Single logit | Binary classification |
| `point_estimate_binary_logits(pred_B, threshold, k)` | Scalar prediction | Thresholding a regression output |
| `gaussian_binary_logits(mu_B, log_var_B, threshold)` | Gaussian params | P(Y > threshold) via CDF |

!!! warning "Steep sigmoid saturates gradients"
    `point_estimate_binary_logits` uses a sigmoid with steepness parameter `k`. Large values (k=100) make `sigmoid(k*(pred - threshold)) ≈ 1`, driving gradients to zero. Use k=5–10, or prefer DEG over TAG when gradients are unreliable.

!!! tip "Gaussian logits are TAG-friendly"
    `gaussian_binary_logits` computes P(Y > threshold) via the CDF and is differentiable through both the mean and variance. This makes it naturally compatible with TAG gradient-based guidance.

## Template models

All template classes are ABCs — you implement `format_raw_to_logits` using the binary logit functions above.

### LinearProbe

Frozen `GenerativeModelWithEmbedding` + trainable `nn.Linear` head.

```python
LinearProbe(embed_model, output_dim, pooling_fn=None, freeze_embed_model=True)
```

- Default pooling: mean over non-padding positions
- `pooling_fn` takes **two args**: `(embeddings_SPD, seq_SP)` — the token IDs are needed for masking special tokens during pooling
- Set `freeze_embed_model=False` when using LoRA on the embed_model so PEFT's freeze/unfreeze state is preserved
- `precompute_embeddings(sequences, batch_size, device)` caches pooled embeddings for fast training

### EmbeddingMLP

Learnable embeddings + MLP. The embedding lookup uses `ohe @ self.embed.weight`, which is differentiable for TAG gradient flow.

- `padding_idx` defaults to `tokenizer.pad_token_id`
- Supports PCA initialization from pretrained models (see below)

### OneHotMLP

Flattened one-hot encoding → MLP. Takes `sequence_length` as a required constructor argument.

### PairwiseLinearModel

Linear model on single + pairwise OHE features. Computes pairwise outer products and takes the upper triangle. Quadratic in `sequence_length × vocab_size`.

## PCA embedding initialization

`EmbeddingMLP` supports post-construction initialization from PCA of pretrained embeddings:

```python
model.init_embed_from_pretrained_pca(
    source=esmc_model,           # GenerativeModelWithEmbedding
    source_vocab=esmc_tokenizer.vocab,
    target_vocab=model.tokenizer.vocab,
)
```

Key details:

- Token matching by string key — shared tokens are the intersection of vocabulary keys
- PCA is computed only over shared tokens (special tokens excluded from centering/SVD)
- Automatically zeroes the padding row after copy
- This is a post-construction method (not a constructor parameter) to avoid redundant shape arguments

!!! note "Variance capture"
    ESMC's 960-dim embeddings for 20 amino acids have effective rank 19 (after centering). The first 20 PCs capture ~100% of the variance. Small embedding dims (8–32) still capture the most important AA similarity structure.

---

## API Reference

::: proteingen.modeling.predictive_modeling
