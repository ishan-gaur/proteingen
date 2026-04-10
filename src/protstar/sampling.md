# Sampling — Design Notes

Sampling algorithms that generate sequences from GenerativeModel (or guided TAG/DEG) instances.

## Dependencies

- [generative_modeling.md](generative_modeling.md) — calls `get_log_probs` on `GenerativeModel` instances
- [guide.md](guide.md) — TAG/DEG are GenerativeModel subclasses, work transparently with these samplers

## Used By

- Examples: `unconditional_sampling.py`, `stability_guidance/main.py`, `esm3_structure_conditioned_sampling.py`
- Tests: `test_sampling.py`

## Sampling Functions

### `sample(model, x_SP, n_parallel=1, in_order=None)`

Main unified sampler. Unmasks positions `n_parallel` at a time, either in random order (default) or in an explicit order via `in_order`.

- Accepts `x_SP` as token IDs or list of strings (auto-tokenizes)
- `in_order`: optional `list[LongTensor]` — one per sequence. If None, random permutation of masked positions.
- Returns `SamplingTrajectory` with sequences (strings), per-step log probs, positions, and tokens.
- DEG-aware: if model has `at_position()`, passes position info before computing log probs
- **Sharp edge**: orders are padded to uniform length with position 0 (BOS/CLS). If logits are properly formatted, padding is a no-op. If not, position 0 may be mutated.

### `any_order_ancestral_step(model, x_SP, n_parallel, mask_token_id, next_pos_idx_SP=None)`

Single step of ancestral sampling:
1. Pick `n_parallel` random masked positions per sequence (unless `next_pos_idx_SP` provided)
2. If model is DEG, wrap `get_log_probs` in `model.at_position(...)` — **positions picked BEFORE calling model**
3. Sample from categorical distribution at selected positions
4. Update `x_SP` in-place

### `sample_ctmc_linear_interpolation(model, x_SP, n_steps, return_string=True)`

Euler integration for flow-matching / linear interpolation:
- Each step: `X_1 = ((steps_left - 1) / steps_left) * X_0 + (1 / steps_left) * exp(log_probs)`
- Samples from the interpolated distribution via multinomial

### `sample_flow_matching_legacy` / `build_legacy_predictor_log_prob`

Legacy flow-matching sampler used in the stability guidance comparison. Euler sampling with `dt`, `x1_temp` parameters. Kept for reproducing original demo results.

## Type Aliases

- `SamplingStep` — `Callable[[TransitionFunc, LongTensor, ...], [FloatTensor, Optional[List[float]]]]`
- `Integrator` — `Callable[[GenerativeModel, List[str]], List[str]]`

## Helper

- `tensor_to_string(x_SP, tokenizer)` — batch decode, strips special tokens (`<mask>`, `<cls>`, `<eos>`)

## Gotchas

- **DEG + n_parallel > 1**: not implemented yet. `sample` raises `NotImplementedError` for this case.
- **`any_order_ancestral_step` takes model (not callable)** — it needs to detect DEG via `hasattr(model, "at_position")` and pick positions before calling `get_log_probs`.
- **In-place mutation**: `any_order_ancestral_step` modifies `x_SP` in-place. `sample` clones first.
- **Device handling**: `sample` and `sample_ctmc_linear_interpolation` move input to `model.device`.
