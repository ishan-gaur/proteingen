---
name: likelihood-curves
description: Evaluate and plot log-likelihood trajectories for transition models under progressive unmasking. Use when comparing how well one or more generative models predict masked amino acids as context is revealed, or when evaluating fine-tuned vs base models on a protein dataset.
---

# Likelihood Curves

Evaluate transition models by measuring average log p(x_true) at masked positions across a sweep of noise levels. At each level t ∈ [0, 1), positions are kept with probability t and the rest are masked. Produces mean ± std plots showing how model confidence changes as context is revealed.

## API

Everything lives in `proteingen.eval`:

```python
from proteingen.eval import compute_log_prob_trajectory, plot_log_prob_trajectories, LogProbTrajectory
```

**`compute_log_prob_trajectory(sequences, model, n_time_points, batch_size=32) -> LogProbTrajectory`**
- `model` must be a `TransitionModel` whose tokenizer has a `mask_token_id`
- Returns `time_points: (n_time_points,)` and `avg_log_probs: (n_sequences, n_time_points)` — NaN where no positions were masked

**`plot_log_prob_trajectories(trajectories, labels, output_path, show_individual=True, max_individual=200, title=...)`**
- Overlays multiple trajectories (mean ± std bands) on one figure — up to 10 colors
- `show_individual=True` draws faint per-sequence lines; turn off for large sequence counts

## Gotchas

- At t close to 1, some sequences may have zero masked positions → those entries are NaN. Aggregation is NaN-safe.
- `batch_size` is about GPU memory, not statistical — results are identical regardless of batch size.
- A flat curve means the model isn't using context (e.g. uniform predictor). Higher curves = better model.

## Reference

- Implementation: `src/proteingen/eval/likelihood_curves.py`
- End-to-end example: `examples/trpb_likelihood_curves.py`
- Tests: `tests/test_likelihood_curves.py`
