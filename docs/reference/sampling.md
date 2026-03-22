# sampling

Sampling algorithms that generate sequences from `TransitionModel` instances (including guided TAG/DEG models). Because TAG and DEG are `TransitionModel` subclasses, all samplers work transparently with both guided and unguided models.

## Samplers

### `sample_any_order_ancestral`

The main high-level sampler. Unmasks positions one (or `n_parallel`) at a time in random order.

```python
sample_any_order_ancestral(model, x_SP, n_parallel=1, return_string=True)
```

- Accepts `x_SP` as token IDs or a list of strings (auto-tokenizes)
- If the model is DEG, automatically passes position info via `model.at_position()` before computing log probs
- Shows a progress bar tracking the fraction of positions unmasked
- Asserts no mask tokens remain at completion

Each step:

1. Pick `n_parallel` random masked positions per sequence
2. If DEG, wrap `get_log_probs` in `model.at_position(...)` — positions are picked **before** calling the model
3. Sample from the categorical distribution at selected positions
4. Update `x_SP` in-place

### `sample_linear_interpolation`

Euler integration for flow-matching / linear interpolation:

```python
sample_linear_interpolation(model, x_SP, n_steps, return_string=True)
```

Each step: `X_1 = ((steps_left - 1) / steps_left) * X_0 + (1 / steps_left) * exp(log_probs)`, then sample from the interpolated distribution.

### `sample_flow_matching_legacy`

Legacy flow-matching sampler with `dt` and `x1_temp` parameters. Kept for reproducing results from the original stability guidance demo.

## Helpers

- `tensor_to_string(x_SP, tokenizer)` — batch decode, strips special tokens (`<mask>`, `<cls>`, `<eos>`)
- `build_legacy_predictor_log_prob` — builds a legacy-compatible predictor log-prob callable

## Gotchas

- **DEG + `n_parallel > 1`**: not yet implemented — raises `NotImplementedError`.
- **In-place mutation**: `any_order_ancestral_step` modifies `x_SP` in-place.
- **Device handling**: `sample_any_order_ancestral` and `sample_linear_interpolation` move input to `model.device` and return on the original device (or as strings).

---

## API Reference

::: proteingen.sampling
