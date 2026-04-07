# Unconditional Sampling

Generate protein sequences from scratch using ESMC as a masked language model.

## Quick Start

```python
from proteingen.models import ESMC
from proteingen import sample_any_order_ancestral

model = ESMC().cuda()
initial_x = ["<mask>" * 256 for _ in range(5)]
sequences = sample_any_order_ancestral(model, initial_x)
```

```bash
uv run python examples/unconditional_sampling.py
```

## How It Works

This starts from fully masked sequences and iteratively unmasks positions using ESMC's learned distribution. At each step, the model predicts a probability distribution over amino acids for every masked position, one position is sampled, and the process repeats until no masks remain.

The decoding order is random by default — positions are unmasked in a uniformly random permutation. This means each run produces different sequences even from the same starting point.

**Source**: [`examples/unconditional_sampling.py`](https://github.com/ishan-gaur/proteingen/blob/main/examples/unconditional_sampling.py)
