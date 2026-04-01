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

Conditioning attaches observations (e.g. structure coordinates) that persist across `get_log_probs` calls. There are two distinct conditioning modes depending on whether you're doing inference or training.

### Inference: single conditioning, tiled to batch

For sampling and evaluation, you typically have **one** conditioning input (e.g. a single backbone structure) shared across all sequences in the batch. Use `set_condition_()` or `conditioned_on()`:

```python
model.set_condition_(obs)              # in-place, returns None
model = model.set_condition(obs)       # returns self (chainable)
with model.conditioned_on(obs):        # context manager, reverts on exit
    log_probs = model.get_log_probs(x)
```

The flow is:

1. `set_condition_()` calls `preprocess_observations(obs)` **once** and caches the result in `self.observations`
2. Each `get_log_probs()` call runs `collate_observations(x_B, self.observations)` to tile the cached observations to match the batch size
3. The tiled observations are passed as `**kwargs` to `forward()` and `format_raw_to_logits()`

The `preprocess_observations` → `collate_observations` split is a performance optimization: expensive preprocessing (running a VQ-VAE encoder) happens once when the condition is set, while cheap per-batch collation (tiling tensors to batch size) happens on every forward pass.

```python
# Example: ESM3 inverse folding — one structure, many sequence samples
model = ESM3().cuda()
model.set_condition_({"coords_RAX": backbone_coords})  # VQ-VAE runs once

# Every get_log_probs call tiles the cached structure tokens to batch size
for step in range(n_steps):
    log_probs = model.get_log_probs(batch_of_sequences)
```

!!! warning "Conditioning is mutable state"
    `set_condition_()` modifies `self.observations` in place. The `conditioned_on()` context manager handles revert, but be careful with concurrent usage.

### Training: per-sample conditioning via collator

For training, each sequence in the batch typically has its **own** conditioning input (e.g. each protein has a different predicted structure). This pattern bypasses `set_condition_()` / `get_log_probs()` entirely — the collator prepares per-sample observations and the training loop calls `model.forward()` directly.

The flow is:

1. `ProteinDataset` stores per-sequence observations (e.g. structure tokens, coordinates)
2. The collator batches observations from individual samples into batched tensors
3. The training loop passes observations directly to `model(input_ids, **observations)`

```python
# Example: inverse folding training — each sequence has its own structure
for batch in loader:
    input_ids = batch["input_ids"].to(device)
    target_ids = batch["target_ids"].to(device)
    struct_tokens = batch["structure_tokens"].to(device)
    coords = batch["coordinates"].to(device)

    # Call forward directly with per-sample observations
    raw = model(input_ids, structure_tokens=struct_tokens, coordinates=coords)
    logits = model.format_raw_to_logits(
        raw, input_ids, structure_tokens=struct_tokens, coordinates=coords
    )
    loss = F.cross_entropy(logits[masked], target_ids[masked])
```

There are two ways to build the collator:

**`ProteinDataset.collator()`** — for sequence-only training or when `preprocess_observations` handles batched inputs. The built-in collator gathers per-sample observations into list-valued dicts and calls `model.preprocess_observations(batched)`. See [data](data.md) for details.

**Custom collator** — for complex cases like inverse folding where you need to pad both sequences and structures to the batch max length. The [fine-tuning workflow](../workflows/finetune-generative.md) shows a complete example.

!!! note "Why not use `set_condition_()` for training?"
    `set_condition_()` caches a single observation and tiles it to the batch — it assumes every sample in the batch shares the same conditioning. During training, each sample has different conditioning, so you pass observations directly through the collator → `forward()` path instead.

### Summary: which pattern to use

| Scenario | Pattern | Observations flow |
|----------|---------|-------------------|
| Sampling / evaluation | `set_condition_()` or `conditioned_on()` | One obs → `preprocess` (once) → `collate` (tile to batch) → `get_log_probs` |
| Training (sequence-only) | `ProteinDataset.collator()` | No observations needed |
| Training (per-sample conditioning) | Custom collator or `ProteinDataset.collator()` | Per-sample obs → collator batches → `model.forward(**obs)` directly |

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
