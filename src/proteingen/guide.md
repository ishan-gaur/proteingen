# Guidance — Design Notes

TAG (gradient-based) and DEG (enumeration-based) guidance algorithms, plus the cross-tokenizer projection layer.

## Dependencies

- [generative_modeling.md](generative_modeling.md) — TAG/DEG subclass `TransitionModel`
- [predictive_modeling.md](predictive_modeling.md) — TAG/DEG consume `PredictiveModel` (call `get_log_probs`, `grad_log_prob`)

## Used By

- [sampling.md](sampling.md) — sampling functions call `get_log_probs` on TAG/DEG instances
- Examples: `stability_guidance/main.py`, `trpb_linear_probe.py`
- Tests: `test_tag_projection.py`

## TAG

`TAG(TransitionModel)` — combines gen model + pred model via Bayes' rule using first-order Taylor expansion of predictor gradients.

### Constructor

`__init__(gen_model, pred_model, use_clean_classifier=False, projection=None)`

- If `projection is None`, auto-creates `LinearGuidanceProjection` from the two tokenizers
- `use_clean_classifier=True` fills mask tokens with gen argmax before computing predictor gradients

### Forward

```
forward(seq_SP):
    logp_gen = gen_model.get_log_probs(seq_SP)           # gen log probs
    prepared = projection.prepare(seq_SP, logp_gen, ...)  # build predictor input
    grad = pred_model.grad_log_prob(prepared.seq_pred_SP) # predictor gradient
    delta = projection.grad_to_gen_delta(grad, prepared)  # project to gen space
    return logp_gen + delta / pred_model.temp             # Bayes combination
```

### Temperature semantics

- **No separate `guidance_scale`** parameter
- Predictor's temperature controls guidance strength: lower temp → steeper log_softmax → larger gradient magnitude
- Gen model's temperature controls prior flatness
- TAG internally computes gradient at temp=1, then divides by predictor temp as a linear multiplier

## DEG

`DEG(TransitionModel)` — enumeration-based guidance. Evaluates the predictor at all vocab tokens for a given position.

### Constructor

`__init__(gen_model, pred_model, argmax_masked_positions=False)`

### Position management

- Requires position info via `at_position(positions_to_score_S)` context manager
- `positions_to_score_S` is a list of length B — one position index per sequence (or `None` to skip)
- `sample_any_order_ancestral` passes this automatically

### Forward

For each sequence, enumerates all `vocab_size` tokens at the target position, evaluates predictor on each, combines with gen log probs.

### When to use DEG vs TAG

- **DEG is better for frozen-LM probes** — TAG gradients through 30-layer frozen transformers are unreliable. DEG only needs correct rankings, not accurate gradients.
- **TAG is faster** — one backward pass vs `vocab_size` forward passes per position
- DEG with `n_parallel > 1` is not yet implemented

## GuidanceProjection

`GuidanceProjection(nn.Module, ABC)` — abstract base for mapping between gen and pred token spaces.

### Abstract methods

| Method | Purpose |
|--------|---------|
| `prepare(seq_gen_SP, logp_gen_SPT, *, use_clean_classifier)` | Build predictor input tokens and Taylor baseline |
| `grad_to_gen_delta(grad_pred_SPK, prepared, *, gen_output_dim)` | Project predictor gradients into gen logit space |

## LinearGuidanceProjection

`LinearGuidanceProjection(GuidanceProjection)` — the default. Fixed linear map between token spaces.

### Core math

```
M[T_gen, K_pred]  — each gen token's row is its predictor OHE representation
delta(t) = g · (M[t] - M[baseline])   — first-order Taylor delta per token
```

Where `g` is the predictor gradient, `M[t]` is the predictor OHE for gen token `t`, and `baseline` is the current/argmax token.

### Constructor

```python
LinearGuidanceProjection(
    tokenizer_gen, tokenizer_pred,
    pred_token_ohe_basis_TK,      # from pred_model.token_ohe_basis()
    fallback_pred_token_id=None,   # for unmapped gen tokens
    strip_prefix=None,             # auto-detect CLS stripping
    strip_suffix=None,             # auto-detect EOS stripping
)
```

### Token mapping

- Builds `gen_to_pred_idx_T` by matching token strings between gen and pred vocabs
- Unmapped tokens fall back to `fallback_pred_token_id` (defaults to pred's mask or unk token)
- CLS/EOS stripping: auto-detected when gen has CLS/EOS but pred doesn't (e.g. ESM → MPNN)

### Registered buffers

- `gen_to_pred_idx_T` — gen token → pred token index mapping
- `pred_token_ohe_basis_TK` — predictor's OHE basis matrix
- `gen_to_pred_ohe_TK` — combined: gen token → pred OHE features

## Guidance Gotchas

### Gradient issues

- **Gradient vanishes on `<mask>` tokens**: predictor trained on real AA embeddings → gradients through frozen ESMC transformer vanish ~10^6x on mask inputs. Fix: `use_clean_classifier=True` fills mask positions with gen argmax.
- **Steep sigmoid saturates gradient**: `k=100` in `point_estimate_binary_logits` → `sigmoid(k*(pred-threshold))` ≈ 1 → gradient ≈ 0. Use k=5–10 or use DEG.
- **Enumeration > gradient for frozen-LM probes**: TAG through 30-layer frozen transformer is unreliable. DEG gives better guidance because it only needs correct rankings.

### Temperature tuning

- **Gen temperature is key**: ESMC prior at well-determined positions (e.g. G at pos 227, log prob ≈ -0.00) is unoverridable at temp=1. Higher gen temp (2–3) flattens the prior, giving guidance room to steer. Combined with guidance scale (10–20), this produces significant improvements.
- **TrpB benchmark results**: Unguided mean=0.48, DEG(scale=20, temp=3) mean=0.62, %>0.7 from 0.5% to 32.5% (N=200, 10 runs)
- **ESMC-300m vs 600m**: essentially identical ρ on TrpB (0.38 vs 0.39)

### Cross-tokenizer

- TAG projection keeps tokenizer/OHE mismatch logic in the `GuidanceProjection` layer, NOT inside predictive models
- Stability predictor overrides `token_ohe_basis()` so `<mask>` → all-zero OHE row — see [models/rocklin_ddg/rocklin_ddg.md](models/rocklin_ddg/rocklin_ddg.md)
- Varpos-concat pooling (4 positions × D) is better than full mean pool for TrpB (0.38 vs 0.28 for MLP)
