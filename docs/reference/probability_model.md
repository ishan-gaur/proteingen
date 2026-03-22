# probability_model

`ProbabilityModel` is the root of the entire class hierarchy — both generative and predictive models inherit from it. It's an `nn.Module` + `ABC` that provides temperature scaling, observation conditioning, log-probability computation, and checkpointing.

## The `get_log_probs` pipeline

Every call to `get_log_probs` follows the same chain:

```
collate_observations(x_B, self.observations)
    → forward(x_B, **obs)
        → format_raw_to_logits(raw, x_B, **obs)
            → log_softmax(logits / temp)
```

When no observations are set, `forward` and `format_raw_to_logits` are called without keyword arguments.

## Abstract methods

Subclasses **must** implement two methods:

| Method | Signature | Notes |
|--------|-----------|-------|
| `forward` | `(x_B, **kwargs) → Any` | Returns raw output — can be a non-tensor type (e.g. ESM returns a dataclass). |
| `format_raw_to_logits` | `(raw_output, x_B, **kwargs) → FloatTensor` | Extracts a logit tensor suitable for `log_softmax`. Receives the full context via `**kwargs`. |

## Overridable defaults

| Method | Default behavior | Override when… |
|--------|-----------------|----------------|
| `preprocess_observations(obs)` | Pass-through | You have expensive one-time ops (e.g. ESM3's VQ-VAE structure encoding). |
| `collate_observations(x_B, obs)` | Tile each tensor to match batch size | You have non-tensor observations or need selective expansion. |

## Conditioning

Conditioning attaches observations (e.g. structure coordinates) that persist across `get_log_probs` calls. The API has three levels:

```python
model.set_condition_(obs)              # in-place, returns None
model = model.set_condition(obs)       # returns self (chainable)
with model.conditioned_on(obs):        # context manager, reverts on exit
    log_probs = model.get_log_probs(x)
```

The `preprocess_observations` → `collate_observations` split is a performance optimization: expensive preprocessing (running a VQ-VAE encoder) happens once when the condition is set, while cheap per-batch collation (tiling tensors to batch size) happens on every forward pass.

!!! warning "Conditioning is mutable state"
    `set_condition_()` modifies `self.observations` in place. The `conditioned_on()` context manager handles revert, but be careful with concurrent usage.

## Checkpointing

Save and restore models with their constructor arguments:

```python
model.save("checkpoints/my_model")
restored = MyModel.from_checkpoint("checkpoints/my_model")
```

Subclasses participate via:

- `_save_args() → dict` — return JSON-serializable constructor kwargs (raises `NotImplementedError` by default)
- Override `save()` to write additional state (weights, LoRA adapters), then call `super().save()`
- Override `from_checkpoint()` to load additional state after construction

For example, `TransitionModel.save` writes a `lora_adapter/` directory if LoRA is present, and `LinearProbe.save` writes `head.pt` plus delegates to `embed_model.save()`.

## Gotchas

- The `device` property uses `next(self.parameters()).device` — it will fail on models with no parameters (all current subclasses have parameters).
- `forward` returns `Any`, not just tensors. This is intentional — ESM models return dataclass outputs. The `format_raw_to_logits` step is where you extract the tensor.
- The default `collate_observations` assumes all observation values are tensors or scalars. If you store non-tensor observations (lists, strings), override this method.

---

## API Reference

::: proteingen.probability_model
