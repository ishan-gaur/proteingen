# ESM3

ESM3-open (1.4B parameters). Supports optional structure conditioning via atom37 coordinates.

- **Output dim**: 64
- **Embedding dim**: 1536
- **LoRA support**: yes — use `lr ≤ 1e-4` to avoid mode collapse

## Structure conditioning (inverse folding)

ESM3 accepts atom37-format coordinates as conditioning input, enabling structure-conditioned sequence generation (inverse folding). For inference, the `set_condition_()` method runs the VQ-VAE structure encoder once (expensive), then all subsequent calls use cached structure tokens:

```python
from proteingen.models import ESM3
from proteingen.sampling import sample_ctmc_linear_interpolation

model = ESM3("esm3-open").cuda()
coords = ...  # atom37 format, shape (L, 37, 3)

# Set structure conditioning — VQ-VAE encodes coordinates once
model.set_condition_({"coords_RAX": coords})

# Sample sequences conditioned on the structure
init_tokens = tokenizer(["<mask>" * L], return_tensors="pt")["input_ids"].cuda()
sequences = sample_ctmc_linear_interpolation(model, init_tokens, n_steps=100)

# Or use a context manager (reverts conditioning on exit)
with model.conditioned_on({"coords_RAX": coords}):
    log_probs = model.get_log_probs(seq)
```

For training with per-sample structures (e.g. fine-tuning on a family of AF3-predicted structures), pass observations directly through the collator instead of using `set_condition_()`. See the [conditioning docs](../reference/probability_model.md#conditioning) and the [fine-tuning workflow](../workflows/finetune-generative.md) for the full pattern.

!!! warning "Structure conditioning is length-locked"
    `set_condition_()` preprocesses coordinates to fixed-length structure tokens (L+2 with BOS/EOS). All subsequent calls must use sequences of exactly that length, or you get a shape mismatch. For training with variable-length sequences, use per-sample conditioning via the collator.

!!! warning "Lazy VQ-VAE loading and parameter freezing"
    ESM3's structure encoder (`_structure_encoder`) is loaded lazily on first `set_condition_()` call. If you froze parameters before conditioning (e.g. via `apply_lora()`), the encoder's ~30M parameters won't be frozen. **Always set up conditioning before freezing**, or re-freeze after.

## Memory considerations

ESM3's geometric attention computes pairwise (L×L) tensors. For a sequence of length 297:

- `batch_size ≥ 32` OOMs on 48GB GPU
- `batch_size = 16` with bfloat16 AMP is optimal (~32s/epoch for LoRA training)

## LoRA specifics

| Rank (r) | Trainable params | % of 1.4B |
|----------|-----------------|-----------|
| 8 | ~9.8M | 0.69% |
| 4 | ~4.9M | 0.35% |
| 2 | ~2.5M | 0.18% |
