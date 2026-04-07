# Likelihood Curves (TrpB)

Evaluate how well a model predicts masked amino acids under progressive unmasking, using real TrpB sequences from SaProtHub.

## Quick Start

```bash
uv run python examples/trpb_likelihood_curves.py --device cuda --n-sequences 50 --n-time-points 20
```

## What It Produces

A plot showing average log p(true token) at masked positions vs. fraction unmasked. This characterizes how a model's predictive confidence changes as it sees more context — a key diagnostic for masked language models.

See the [Likelihood Curves workflow](../workflows/likelihood-curves.md) for interpretation and usage.

**Source**: [`examples/trpb_likelihood_curves.py`](https://github.com/ishan-gaur/proteingen/blob/main/examples/trpb_likelihood_curves.py)
