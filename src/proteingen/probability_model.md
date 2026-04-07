# ProbabilityModel — Design Notes

## Purpose

`ProbabilityModel(nn.Module, ABC)` is the shared base for **all** models in the library — both generative (GenerativeModel) and predictive (PredictiveModel). It provides temperature scaling, conditioning, log probability computation, and checkpointing.

## Dependencies

- None within proteingen (this is the root of the hierarchy)
- `torch.nn.Module` — inherits from nn.Module for parameter management, device tracking, serialization

## Used By

- [generative_modeling.md](generative_modeling.md) — `GenerativeModel(ProbabilityModel)`, `GenerativeModelWithEmbedding`
- [predictive_modeling.md](predictive_modeling.md) — `PredictiveModel(ProbabilityModel)`
- [sampling.md](sampling.md) — sampling functions call `get_log_probs`
- [guide.md](guide.md) — guidance modifies log probs from this interface

## API Checklist

### Abstract methods (subclasses MUST implement)

| Method | Signature | Notes |
|--------|-----------|-------|
| `forward` | `(x_B, **kwargs) -> Any` | Returns raw output — can be non-tensor (e.g. ESMCOutput dataclass) |
| `format_raw_to_logits` | `(raw_output, x_B, **kwargs) -> FloatTensor` | Converts raw output to logits suitable for `log_softmax`. Receives full context via `**kwargs`. |

### Concrete methods with overridable defaults

| Method | Default behavior | Override when... |
|--------|-----------------|------------------|
| `preprocess_observations(obs)` | Pass-through | Expensive one-time ops (e.g. VQ-VAE structure encoding in ESM3) |
| `collate_observations(x_B, obs)` | Tile each tensor to batch size | Custom collation needed (non-tensor obs, selective expansion) |

### Concrete methods (usually not overridden)

- `get_log_probs(x_B)` — the main pipeline (see below)
- `set_temp_()` / `set_temp()` / `with_temp()` — temperature management
- `set_condition_()` / `set_condition()` / `conditioned_on()` — observation management
- `save(path)` / `from_checkpoint(path)` / `_save_args()` — checkpointing

## `get_log_probs` Pipeline

This is the core computation path. Understanding the order matters:

```
collate_observations(x_B, self.observations)
    → forward(x_B, **obs)
        → format_raw_to_logits(raw, x_B, **obs)
            → log_softmax(logits / temp)
```

- Asserts `self.temp > 0` before dividing
- When `self.observations is None`, calls `forward(x_B)` and `format_raw_to_logits(raw, x_B)` without kwargs
- `PredictiveModel` overrides this to add OHE creation and `[:, 1]` extraction

## Conditioning

Three-level API with revert support:

- `set_condition_(obs)` — in-place, calls `preprocess_observations` once, caches result in `self.observations`
- `set_condition(obs)` — returns `self` for chaining
- `conditioned_on(obs)` — context manager, reverts `self.observations` on exit

The `preprocess_observations` → cache → `collate_observations` split exists so expensive preprocessing (e.g. running a VQ-VAE encoder) happens once, while cheap per-batch collation (e.g. tiling to batch size) happens every forward pass.

## Checkpointing

Protocol for save/load:

- `_save_args() -> dict` — subclasses return JSON-serializable constructor kwargs. Raises `NotImplementedError` by default.
- `save(path)` — writes `config.json`. Subclasses override to add state (weights, LoRA adapters, etc.) and call `super().save(path)`.
- `from_checkpoint(path)` — reads `config.json`, calls `cls(**args)`. Subclasses override to load additional state.

Concrete examples:
- `GenerativeModel.save` adds `lora_adapter/` if LoRA is present
- `LinearProbe.save` adds `head.pt` + delegates to `embed_model.save(path/embed_model/)`
- ESMC/ESM3 stash checkpoint names (`self._esmc_checkpoint`, `self._esm3_checkpoint`) for `_save_args()`

## Maintenance

If changes are made to `ProbabilityModel`'s interface (abstract methods, conditioning protocol, checkpointing), update the `add-generative-model` skill (`.agents/skills/add-generative-model/SKILL.md`) to reflect the new contract.

## Gotchas

- **`device` property** uses `next(self.parameters()).device` — will fail on models with no parameters. All current subclasses have parameters.
- **`forward` returns `Any`**, not just tensors. This is intentional — ESM models return dataclass outputs. The `format_raw_to_logits` step is where you extract the tensor.
- **`format_raw_to_logits` receives `x_B` and `**kwargs`** — this gives it full context. E.g. `GenerativeModel` needs `seq_SP` for logit formatting; `PredictiveModel` doesn't use the kwargs.
- **Conditioning is mutable state** — `set_condition_()` modifies `self.observations` in place. The `conditioned_on()` context manager handles revert, but be careful with concurrent usage.
- **`collate_observations` default assumes all values are tensors or scalars** — if you store non-tensor observations (e.g. lists, strings), override this method.
