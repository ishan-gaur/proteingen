# evaluation

Tools for sanity-checking your pipeline at each stage: data quality, model fidelity, generation diversity, and structural plausibility.

## Likelihood Curves

The primary evaluation tool for generative models. Measures how well a model predicts masked amino acids as context is progressively revealed.

```python
from protstar.eval import compute_log_prob_trajectory, plot_log_prob_trajectories

traj = compute_log_prob_trajectory(sequences, model, n_time_points=20)
plot_log_prob_trajectories([traj], ["ESMC-300M"], "likelihood.png")
```

At each noise level $t \in [0, 1)$, positions are randomly unmasked with probability $t$, and the model's average log $p(x_\text{true})$ at the remaining masked positions is recorded. This produces a curve from "no context" (left) to "full context" (right).

**What to look for:**

- **Higher is better** — a fine-tuned model should have higher log-probs than the pretrained baseline on in-distribution sequences
- **Structure conditioning boost** — structure-conditioned models should show uniformly higher curves than sequence-only models
- **Overfitting** — if the fine-tuned model's curve on held-out sequences drops *below* the pretrained model, you've overfit

See the [likelihood curves workflow](../workflows/likelihood-curves.md) for detailed usage and interpretation.

## Oracle Scoring

Score generated sequences with a separately-trained oracle model to estimate how well guidance worked. The oracle should be trained on all available data (including later rounds if available) and is never used during sampling.

```python
# Score generated library with a separately-trained oracle
oracle_preds = oracle.predict(generated_sequences)
```

!!! note "Coming soon"
    Convenience functions for oracle scoring, threshold analysis, and round-over-round improvement tracking.

<!-- TODO: oracle_score(oracle, sequences) convenience, threshold sweep plots -->

## Predictor–Oracle Agreement

Before trusting a noisy predictor during guided sampling, check that it agrees with the oracle on clean (fully unmasked) sequences. Low agreement means the predictor is unreliable during generation.

```python
from scipy.stats import spearmanr

oracle_scores = oracle.predict(val_sequences)
predictor_scores = noisy_predictor.predict(val_sequences)
rho, _ = spearmanr(oracle_scores, predictor_scores)
print(f"Agreement: ρ = {rho:.3f}")  # want ρ > 0.7 ideally
```

!!! note "Coming soon"
    `predictor_oracle_agreement(oracle, predictor, sequences)` — returns correlation metrics and generates scatter plots.

<!-- TODO: predictor_oracle_agreement function + scatter plot -->

## Diversity Metrics

Assess whether generated libraries have sufficient sequence diversity to be useful for experimental screening.

**Key metrics:**

- **Sequence identity to wildtype** — are variants too similar to the starting point?
- **Pairwise sequence identity** — are generated sequences diverse from each other?
- **Mutational distance distribution** — how many mutations from wildtype?
- **Positional entropy** — is diversity spread across positions or concentrated?

!!! note "Coming soon"
    `sequence_diversity(sequences, wildtype)` — computes all diversity metrics in one call.

<!-- TODO: sequence_diversity function with all metrics above -->

## Structural Validation

For critical applications, validate that generated sequences fold into the intended structure.

**Approaches:**

- **AlphaFold 3** — fold generated sequences and compare predicted structures to the target backbone (using pTM, pLDDT, or TM-score)
- **ESM3 structure tokens** — quick structural plausibility check without full folding

!!! note "Coming soon"
    Integration with `af3-server` for batch structure prediction of generated libraries.

<!-- TODO: af3_validate(sequences, target_structure) convenience -->

---

## API Reference

::: protstar.eval
