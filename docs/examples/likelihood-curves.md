# Likelihood Curves (TrpB)

??? abstract "Architecture Breakdown"
    **Data:** TrpB sequences from SaProtHub (used as evaluation targets, not training data).

    **Models:** ESMC (pretrained, evaluated — not trained) → [generative_modeling](../reference/generative_modeling.md).

    **Sampling:** None (evaluation only).

    **Evaluation:** `compute_log_prob_trajectory` + `plot_log_prob_trajectories` — measures log p(true token) at masked positions across noise levels → [Likelihood Curves](../workflows/likelihood-curves.md), [evaluation](../reference/evaluation.md).

Evaluate how well a model predicts masked amino acids under progressive unmasking, using real TrpB sequences from SaProtHub.

## Quick Start

```bash
uv run python examples/trpb_likelihood_curves.py --device cuda --n-sequences 50 --n-time-points 20
```

## What It Produces

A plot showing average log p(true token) at masked positions vs. fraction unmasked. This characterizes how a model's predictive confidence changes as it sees more context — a key diagnostic for masked language models.

See the [Likelihood Curves workflow](../workflows/likelihood-curves.md) for interpretation and usage.

**Source**: [`examples/trpb_likelihood_curves.py`](https://github.com/ishan-gaur/proteingen/blob/main/examples/trpb_likelihood_curves.py)
