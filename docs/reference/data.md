# data

Dataset, collation, and noise utilities for training protein models.

## ProteinDataset

A `torch.utils.data.Dataset` that holds raw protein data: sequences, per-sample observations (conditioning variables), and optional labels.

```python
ProteinDataset(sequences, observations=None, labels=None)
```

- `sequences` — list of amino acid strings
- `observations` — dict mapping names to per-sample lists (e.g. `{"structure_tokens": [...], "coordinates": [...]}`)
- `labels` — optional `(N,)` or `(N, n_targets)` tensor

The dataset stores raw data only — all model-specific transforms (tokenization, noising, padding) happen in the collator.

### Built-in collator

`ProteinDataset.collator()` builds a `collate_fn` that handles tokenization, noising, and observation preprocessing:

```python
collate_fn = dataset.collator(
    model,                                # provides .tokenizer and .preprocess_observations
    noise_fn=uniform_mask_noise(model.tokenizer),  # masking strategy
    time_sampler=uniform_time,            # when/how much to mask
)
loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
```

Each batch produced by the collator contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `input_ids` | `(B, L)` | Tokenized, padded, optionally noised |
| `target_ids` | `(B, L)` | Tokenized, padded (clean — no noise) |
| `observations` | `dict` | Preprocessed observations ready for `model.forward(**obs)` |
| `labels` | `(B, ...)` or `None` | Per-sample targets |

The collator gathers per-sample observations into list-valued dicts and passes them to `model.preprocess_observations(batched)`. This means the model's `preprocess_observations` must accept **batched** inputs (lists of values) when used with the collator, as opposed to single observations when used with `set_condition_()`.

#### Key arguments

**`noise_fn`**: `(input_ids_1D, t) -> noised_input_ids_1D` — the corruption strategy applied independently to each sequence. Built-in options:

- `uniform_mask_noise(tokenizer)` — mask non-special positions with probability `(1 - t)`
- `no_noise` — identity (no corruption)

**`time_sampler`**: `() -> float` in `[0, 1]` — controls how much masking to apply. Built-in options:

- `uniform_time` — sample `t ~ Uniform(0, 1)`
- `fully_unmasked` — always returns `t = 1` (no masking)

**`rename_obs_keys`**: `{model_kwarg: dataset_key}` — for when two models use different names for the same data. One dataset, multiple collators:

```python
# Dataset stores "structure_coords", but the model expects "coordinates"
collate_fn = dataset.collator(
    model,
    noise_fn=no_noise,
    time_sampler=fully_unmasked,
    rename_obs_keys={"coordinates": "structure_coords"},
)
```

### Custom collators

For complex cases — like inverse folding where you need to pad both sequences and structures to the batch max length — write a custom collator instead. The training loop then passes observations directly to `model.forward()`:

```python
def inverse_folding_collator(tokenizer, mask_token_id):
    def collate_fn(batch):
        sequences = [s["sequence"] for s in batch]
        tokenized = tokenizer(sequences, padding=True, return_tensors="pt")
        target_ids = tokenized["input_ids"]
        B, L = target_ids.shape

        # Mask all non-special positions
        input_ids = target_ids.clone()
        input_ids[maskable_positions] = mask_token_id

        # Pad structures to match tokenized length L
        padded_struct = torch.full((B, L), STRUCTURE_PAD, dtype=torch.long)
        padded_coords = torch.zeros(B, L, 37, 3)
        for i, sample in enumerate(batch):
            seq_len = sample["structure_tokens"].shape[0]
            padded_struct[i, :seq_len] = sample["structure_tokens"]
            padded_coords[i, :seq_len] = sample["coordinates"]

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "structure_tokens": padded_struct,
            "coordinates": padded_coords,
        }
    return collate_fn
```

See the [fine-tuning workflow](../workflows/finetune-generative.md) for a complete example and the [conditioning docs](probability_model.md#training-per-sample-conditioning-via-collator) for how this fits into the broader conditioning model.

## Noise design

`noise_fn` and `time_sampler` are intentionally separated:

- **`noise_fn`** owns the corruption strategy (what kind of noise)
- **`time_sampler`** owns the schedule (how much noise)

This lets you reuse the same corruption with different `t` distributions (e.g. uniform for training, fixed for evaluation), or swap corruption strategies while keeping the same schedule.

Both are required arguments to `collator()` — there are no defaults. Use the explicit sentinels `no_noise` + `fully_unmasked` when you want clean (unmasked) training data.

## FASTA utilities

- `read_fasta(path)` — returns `list[tuple[header, sequence]]`
- `aligned_sequences_to_raw(aligned_seqs)` — strips gap characters (`-`, `.`) from MSA-aligned sequences

## GuidanceDataset (legacy)

!!! warning "Deprecated"
    `GuidanceDataset` is the older dataset class. Use `ProteinDataset` with appropriate noise functions for new code.

---

## API Reference

::: proteingen.data
