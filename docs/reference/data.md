# data

Dataset utilities and noise schedules for training noisy classifiers — predictive models that work well during the iterative denoising/sampling process.

## NoiseSchedule

A callable that returns a timestep `t ∈ [0, 1]` where `t=0` is fully masked and `t=1` is fully unmasked.

Built-in schedules:

| Schedule | Behavior |
|----------|----------|
| `unmasked_only()` | Always returns 1.0 (no masking) |
| `uniform_schedule()` | Samples t uniformly from [0, 1] |

## GuidanceDataset

A `torch.utils.data.Dataset` for training noisy classifiers.

```python
GuidanceDataset(sequences, labels, tokenize, noise_schedule, mask_token, sequence_metadata=None)
```

- `tokenize` — callable converting (sequence, optional metadata) → model input dict
- `noise_schedule` — `NoiseSchedule` callable for sampling masking timesteps
- `mask_token` — token string used for masking positions (e.g. `"<mask>"`)
- `sequence_metadata` — optional per-sequence side info (e.g. structures) passed to `tokenize`

Each `__getitem__` returns:

```python
{"input": tokenized, "labels": labels[idx], "timestep": t, "mask": mask}
```

Where `mask` is a boolean tensor of positions masked at the sampled timestep.

!!! note "Status"
    This module is partially implemented. TODOs remain for the noising implementation in `__getitem__` (currently masks the string directly) and train/dev/test split support with reproducible seeding.

---

## API Reference

::: proteingen.data
