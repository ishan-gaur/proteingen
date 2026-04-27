# guide

This module implements TAG (Taylor-Approximate Guidance) and DEG (Discrete Enumeration Guidance) — two algorithms for combining a generative model with a predictive model via Bayes' rule. It also contains the `GuidanceProjection` abstraction for handling cross-tokenizer mapping.

## The guidance equation

Both algorithms implement the same principle:

$$
p_\text{guided}(x_t | x_{<t}) \propto p_\text{gen}(x_t | x_{<t}) \cdot p_\text{pred}(\text{target} | x)^\gamma
$$

Since TAG and DEG are `GenerativeModel` subclasses, they produce guided log-probs that plug directly into any sampler — no special handling needed.

## TAG (Taylor-Approximate Guidance)

Uses first-order Taylor expansion of the predictive model's log-prob to compute guidance deltas efficiently.

```python
TAG(gen_model, pred_model, use_clean_classifier=False, projection=None)
```

### Forward pass

```
logp_gen = gen_model.get_log_probs(seq_SP)           # generative log probs
prepared = projection.prepare(seq_SP, logp_gen, ...)  # build predictor input
grad = pred_model.grad_log_prob(prepared.seq_pred_SP) # predictor gradient (∂log p / ∂OHE)
delta = projection.grad_to_gen_delta(grad, prepared)  # project gradient to gen logit space
return logp_gen + delta / pred_model.temp             # Bayes' rule combination
```

### Temperature as guidance strength

TAG does **not** have a separate `guidance_scale` parameter. Instead:

- **Predictor temperature** controls guidance strength: lower temp → steeper log_softmax → larger gradient magnitude → stronger guidance
- **Generator temperature** controls prior flatness: higher temp → flatter prior → more room for guidance to steer
- Internally, TAG computes the gradient at temp=1, then divides by the predictor's temperature as a linear multiplier

!!! tip "Temperature tuning in practice"
    ESMC's prior at well-determined positions (e.g. conserved glycine with log prob ≈ 0.0) is nearly impossible to override at temp=1. Raising gen temp to 2–3 flattens the prior, giving guidance room to steer. Combined with predictor temp equivalent to guidance scale 10–20, this produces significant improvements.

    **TrpB benchmark:** Unguided mean fitness = 0.48 → DEG (scale=20, temp=3) mean = 0.62, fraction above 0.7 from 0.5% to 32.5% (N=200, 10 runs).

### When to use TAG

TAG is fast (one backward pass per sampling step) but requires reliable gradients through the predictive model. It works well for:

- Small predictive models (OneHotMLP, EmbeddingMLP)
- LoRA-adapted backbones where gradients flow through adapted layers
- Gaussian binary logits (differentiable through mean and variance)

!!! warning "Gradient vanishes on `<mask>` tokens"
    Predictors trained on real AA embeddings produce vanishing gradients (~10⁶× attenuation) when evaluated on `<mask>` inputs through a frozen transformer. Fix: set `use_clean_classifier=True` to fill mask positions with the generative model's argmax before computing predictor gradients.

## DEG (Discrete Enumeration Guidance)

Evaluates the predictor at all vocabulary tokens for a given position and reweights.

```python
DEG(gen_model, pred_model, argmax_masked_positions=False)
```

### Position management

DEG requires position info via a context manager:

```python
with deg.at_position(positions_to_score_S):
    log_probs = deg.get_log_probs(seq_SP)
```

`positions_to_score_S` is a list of length B — one position index per sequence (or `None` to skip). The `sample` function handles this automatically.

### When to use DEG

DEG is more robust than TAG when gradients are unreliable. It only needs correct **rankings** from the predictor, not accurate gradient magnitudes:

- **Frozen-LM probes** — TAG gradients through 30-layer frozen transformers are unreliable. DEG gives better guidance.
- **Point estimate predictors with steep sigmoids** — large `k` in `point_estimate_binary_logits` saturates gradients. DEG sidesteps this.

The tradeoff: DEG requires `vocab_size` forward passes per position per step (vs. one backward pass for TAG).

!!! note "Limitation"
    DEG with `n_parallel > 1` (unmasking multiple positions simultaneously) is not yet implemented.

## GuidanceProjection

When the generative and predictive models use different tokenizers (e.g. ESM with 33 tokens vs. PMPNN with 21 tokens), `GuidanceProjection` handles the mapping.

### LinearGuidanceProjection

The default projection. Builds a fixed linear map between token spaces:

```
M[T_gen, K_pred]  — each gen token's row is its predictor OHE representation
delta(t) = g · (M[t] - M[baseline])   — first-order Taylor delta per gen token
```

Where `g` is the predictor gradient, `M[t]` is the predictor OHE for gen token `t`, and `baseline` is the current/argmax token.

```python
LinearGuidanceProjection(
    tokenizer_gen, tokenizer_pred,
    pred_token_ohe_basis_TK,       # from pred_model.token_ohe_basis()
    fallback_pred_token_id=None,    # for unmapped gen tokens
    strip_prefix=None,              # auto-detect CLS stripping
    strip_suffix=None,              # auto-detect EOS stripping
)
```

Key details:

- Token matching by string key between gen and pred vocabularies
- Unmapped tokens fall back to `fallback_pred_token_id` (defaults to pred's mask or unk token)
- CLS/EOS stripping is auto-detected when gen has CLS/EOS but pred doesn't (e.g. ESM → MPNN)

If `projection` is not provided to TAG, it auto-creates a `LinearGuidanceProjection` from the two models' tokenizers.

---

## API Reference

::: protstar.modeling.guide
