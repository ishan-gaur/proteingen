# sampling

Sampling algorithms that generate sequences from `GenerativeModel` instances (including guided TAG/DEG models). Because TAG and DEG are `GenerativeModel` subclasses, all samplers work transparently with both guided and unguided models.

## Samplers

### `sample`

The main unified sampler. Unmasks positions `n_parallel` at a time, either in random order (default) or in an explicit order.

```python
sample(model, x_SP, n_parallel=1, in_order=None) -> SamplingTrajectory
```

- Accepts `x_SP` as token IDs or a list of strings (auto-tokenizes)
- `in_order`: optional `list[LongTensor]` — one per sequence, giving positions to unmask in order. If None, a random permutation of masked positions is sampled.
- Returns a `SamplingTrajectory` dict with `sequences` (strings), `step_log_probs`, `step_positions`, and `step_tokens`.
- If the model is DEG, automatically passes position info via `model.at_position()` before computing log probs

**Sharp edge**: orders are padded to uniform length across sequences with position 0 (BOS/CLS). At padding steps, all sequences are sampled at their designated positions — including padding. If the model's logit formatter is correctly configured, special-token positions predict only themselves, making the padding a no-op. If logits are NOT properly formatted, the token at position 0 may be mutated.

### `sample_ctmc_linear_interpolation`

Euler integration for flow-matching / linear interpolation:

```python
sample_ctmc_linear_interpolation(model, x_SP, n_steps, return_string=True)
```

Each step: `X_1 = ((steps_left - 1) / steps_left) * X_0 + (1 / steps_left) * exp(log_probs)`, then sample from the interpolated distribution.

### `sample_flow_matching_legacy`

Legacy flow-matching sampler with `dt` and `x1_temp` parameters. Kept for reproducing results from the original stability guidance demo.

## Helpers

- `tensor_to_string(x_SP, tokenizer)` — batch decode, strips special tokens (`<mask>`, `<cls>`, `<eos>`)
- `build_legacy_predictor_log_prob` — builds a legacy-compatible predictor log-prob callable

## Gotchas

- **DEG + `n_parallel > 1`**: not yet implemented — raises `NotImplementedError`.
- **In-place mutation**: `any_order_ancestral_step` modifies `x_SP` in-place. `sample` clones first.
- **Device handling**: `sample` and `sample_ctmc_linear_interpolation` move input to `model.device`.

---

## API Reference

::: proteingen.sampling
