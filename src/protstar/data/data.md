# Data — Design Notes

ProteinDataset, noise functions, time samplers, and FASTA utilities for training.

## Dependencies

- None within protstar (standalone utilities)

## Used By

- Tests: `test_guidance_data.py`
- Examples: `examples/finetune_esm3/` (ESM3 LoRA fine-tuning)
- Training workflows for both generative (MLM) and predictive models

## Key Types

- `NoiseFn = Callable[[LongTensor, float], LongTensor]` — `(input_ids, t) -> noised_ids`
- `TimeSampler = Callable[[], float]` — `() -> t ∈ [0, 1]`

## Noise Functions

- `uniform_mask_noise(tokenizer)` — factory returning a `NoiseFn` that masks non-special positions with prob `(1-t)`. Uses `tokenizer.all_special_ids` to avoid masking CLS/EOS/PAD.
- `no_noise` — identity function, returns input unchanged.

## Time Samplers

- `uniform_time()` — `t ~ Uniform(0, 1)`
- `fully_unmasked()` — always returns `t=1` (no masking)

## ProteinDataset

`ProteinDataset(Dataset)` — holds raw sequences, observations, and labels. All model-specific transforms happen in the collator.

### Constructor

```python
ProteinDataset(sequences, observations=None, labels=None)
```

- `observations` — `{name: list}` of per-sample conditioning data (structures, etc.)
- `labels` — optional `Tensor` of targets

### `collator(model, noise_fn, time_sampler, rename_obs_keys=None)`

Returns a `collate_fn` for DataLoader that:
1. Tokenizes sequences via `model.tokenizer` (with padding)
2. Applies noise: samples `t` per sequence, calls `noise_fn(input_ids[i], t)`
3. Renames and preprocesses observations via `model.preprocess_observations`
4. Stacks labels

`rename_obs_keys` maps `{model_kwarg: dataset_key}` — use when dataset observation names differ from model forward kwargs.

Returns dicts with: `input_ids`, `target_ids`, `observations`, `labels`

### Design: observations vs conditioning

The dataset stores "observations" (model-ready kwarg names). The collator optionally renames keys via `rename_obs_keys={model_kwarg: dataset_key}` so two models with different forward signatures can use the same dataset.

`preprocess_observations` is called per-batch in the collator (batched — values are lists). No upfront preprocessing step.

## FASTA Utilities

- `read_fasta(path)` — returns `[(header, sequence), ...]`
- `aligned_sequences_to_raw(aligned)` — strips gap chars (`-`, `.`) from MSA sequences
