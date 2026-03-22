# Data — Design Notes

GuidanceDataset base class and noise schedule utilities.

## Dependencies

- None within proteingen (standalone utilities)

## Used By

- Tests: `test_guidance_data.py`
- Intended for training noisy classifiers for guidance workflows

## NoiseSchedule

Type alias: `Callable[[], float]` — returns timestep `t ∈ [0, 1]` where `t=0` is fully masked and `t=1` is fully unmasked.

Built-in schedules:
- `unmasked_only()` — always returns 1.0 (no masking)
- `uniform_schedule()` — samples t uniformly from [0, 1]

## GuidanceDataset

`GuidanceDataset(torch.utils.data.Dataset)` — dataset for training noisy classifiers.

### Constructor

```python
GuidanceDataset(sequences, labels, tokenize, noise_schedule, mask_token, sequence_metadata=None)
```

- `tokenize` — callable converting (sequence, optional metadata) → model input dict
- `noise_schedule` — `NoiseSchedule` callable for sampling timesteps
- `mask_token` — token string used for masking positions
- `sequence_metadata` — optional per-sequence side info (e.g. structures) passed to tokenize

### `__getitem__` returns

```python
{"input": tokenized, "labels": labels[idx], "timestep": t, "mask": mask}
```

Where `mask` is a boolean tensor of positions masked at the sampled timestep.

## Status

This module is partially implemented — TODOs remain for:
- Noising implementation in `__getitem__` (currently masks the string directly)
- Train/dev/test split support with reproducible seeding
