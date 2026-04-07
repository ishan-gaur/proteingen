# Structure-Conditioned Sampling (ESM3)

Generate sequences conditioned on a known protein backbone structure using ESM3.

## Quick Start

```bash
uv run python examples/esm3_structure_conditioned_sampling.py
```

## How It Works

ESM3 accepts atom37-format coordinates as conditioning input. The model's `set_condition_()` method runs the VQ-VAE structure encoder once (expensive), then all subsequent sampling steps use the cached structure tokens.

This is useful for **inverse folding** — given a desired 3D structure, generate sequences likely to fold into that shape.

!!! warning "Fixed-length sequences"
    ESM3 structure conditioning requires all sequences in a batch to match the structure length. Structure tokens are `(L+2,)` with BOS/EOS tokens.

See [Models → ESM3](../models/esm3.md) for details on structure conditioning.

**Source**: [`examples/esm3_structure_conditioned_sampling.py`](https://github.com/ishan-gaur/proteingen/blob/main/examples/esm3_structure_conditioned_sampling.py)
