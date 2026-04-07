# ProteinGuide Workflow

End-to-end recipe for guided protein sequence generation. Each step links to relevant API docs and includes practical tips from our own benchmarking.

## Overview

```
Data → PredictiveModel → Train → Combine with GenModel → Sample → Evaluate
```

---

## Step 1: Organize your data

Prepare your fitness/property dataset for training a predictive model.

!!! note "Coming soon"
    Detailed workflow for data organization, train/val/test splits, and the `GuidanceDataset` interface.

<!-- TODO[pi]: write data organization workflow — cover GuidanceDataset, noise schedules, collation -->

---

## Step 2: Set up a PredictiveModel

Implement a `PredictiveModel` subclass that answers "does this sequence have property X?".

Choose your architecture:

=== "LinearProbe"
    Frozen pretrained embeddings (ESMC/ESM3) + a linear head. Fastest to train, good baseline.

    ```python
    class MyProbe(LinearProbe):
        def format_raw_to_logits(self, raw, ohe, **kwargs):
            return point_estimate_binary_logits(raw.squeeze(-1), threshold=0.7, k=10)
    ```

    - Default pooling: mean over non-padding positions
    - Custom pooling: pass `pooling_fn(embeddings_SPD, seq_SP)` — seq_SP needed for masking special tokens
    - `precompute_embeddings()` caches pooled embeddings for faster training
    - Set `freeze_embed_model=False` when using LoRA on the embed model

=== "EmbeddingMLP"
    Learnable amino acid embeddings + MLP. Can initialize from pretrained via PCA.

    ```python
    class MyMLP(EmbeddingMLP):
        def format_raw_to_logits(self, raw, ohe, **kwargs):
            return gaussian_binary_logits(raw[:, 0], raw[:, 1], threshold=0.7)
    ```

    - Embedding lookup uses `ohe @ embed.weight` — differentiable for TAG
    - Initialize from pretrained: `model.init_embed_from_pretrained_pca(esmc, esmc.tokenizer.vocab, model.tokenizer.vocab)`
    - ESMC's 960-dim embeddings have effective rank 19 for 20 AAs — even 8–32 PCs capture the most important AA similarity structure

=== "OneHotMLP"
    One-hot encoded sequences + MLP. No pretrained knowledge, but fully flexible.

    ```python
    class MyOHE(OneHotMLP):
        def format_raw_to_logits(self, raw, ohe, **kwargs):
            return binary_logits(raw.squeeze(-1), target=self.target)
    ```

    - Requires `sequence_length` at construction (flattens to fixed-size input)

=== "Custom"
    Subclass `PredictiveModel` directly and implement `forward` + `format_raw_to_logits`.

    Key requirement: `forward` takes OHE features (not token IDs) and must be differentiable for TAG.

All template models are ABCs — you implement `format_raw_to_logits` using the [binary logit functions](../reference/predictive_modeling.md#binary-logit-functions).

!!! tip "Choosing a binary logit function"
    - **`point_estimate_binary_logits`** — simplest, but steep sigmoid (large `k`) saturates gradients. Use DEG, not TAG, with high `k` values.
    - **`gaussian_binary_logits`** — differentiable through both mean and variance. Naturally TAG-friendly.
    - **`categorical_binary_logits`** — for multi-class predictors (target class vs rest).

<!-- TODO[pi]: write predictive model setup workflow with code templates -->

---

## Step 3: Train a noisy classifier

Train your predictive model on noisy (partially masked) inputs so it works well during the iterative denoising/sampling process.

!!! note "Coming soon"
    Detailed workflow for noisy classifier training — noise schedules, masking strategies, training loops, and diagnostics.

<!-- TODO[pi]: write training workflow — cover noise schedules, GuidanceDataset, training loop, wandb logging -->

---

## Step 4: Combine with TAG or DEG

Combine your trained predictive model with a generative model using guidance.

=== "TAG (Taylor-Approximate Guidance)"

    ```python
    from proteingen.guide import TAG
    from proteingen.models import ESMC

    gen = ESMC("esmc_300m").cuda()
    guided = TAG(gen, predictor).cuda()
    ```

    TAG uses first-order Taylor expansion of the predictor's log-prob. One backward pass per sampling step.

    **Best for:**

    - Small predictive models (OneHotMLP, EmbeddingMLP)
    - LoRA-adapted backbones where gradients flow through adapted layers
    - Gaussian binary logits (differentiable through mean and variance)

    **Watch out for:**

    - Gradients vanish on `<mask>` tokens through frozen transformers (~10⁶× attenuation). Fix: `TAG(gen, pred, use_clean_classifier=True)` fills mask positions with gen argmax first.
    - Steep sigmoid in `point_estimate_binary_logits` (large `k`) saturates gradients → use k=5–10 or switch to DEG.

=== "DEG (Discrete Enumeration Guidance)"

    ```python
    from proteingen.guide import DEG
    from proteingen.models import ESMC

    gen = ESMC("esmc_300m").cuda()
    guided = DEG(gen, predictor).cuda()
    ```

    DEG enumerates all vocabulary tokens at each position and reweights. More robust because it only needs correct **rankings** from the predictor.

    **Best for:**

    - Frozen-LM probes (LinearProbe on ESMC/ESM3) — TAG gradients through 30-layer frozen transformers are unreliable
    - Point estimate predictors with steep sigmoids
    - Any case where you don't trust the gradient magnitudes

    **Tradeoff:** `vocab_size` forward passes per position per step (vs. one backward pass for TAG).

### Temperature tuning

There is **no separate `guidance_scale` parameter**. Temperature controls guidance strength:

- **Predictor temperature** (lower = stronger guidance): TAG computes gradients at temp=1, then divides by predictor temp as a linear multiplier
- **Generator temperature** (higher = weaker prior): ESMC's prior at well-determined positions (e.g. conserved glycine, log prob ≈ 0.0) is nearly impossible to override at temp=1. Raising to 2–3 flattens the prior.

!!! example "TrpB benchmark results"
    | Configuration | Mean fitness | % above 0.7 |
    |--------------|-------------|-------------|
    | Unguided ESMC | 0.48 | 0.5% |
    | DEG (scale=20, temp=3) | 0.62 | 32.5% |

    (N=200 samples, 10 runs. LinearProbe on ESMC-300m + varpos-concat pooling.)

### Cross-tokenizer setup

When gen and pred models use different tokenizers (e.g. ESM with 33 tokens vs. MPNN with 21), TAG auto-creates a `LinearGuidanceProjection` that maps between token spaces. You can also provide your own:

```python
from proteingen.guide import TAG, LinearGuidanceProjection

projection = LinearGuidanceProjection(
    gen.tokenizer, pred.tokenizer,
    pred.token_ohe_basis(),
)
guided = TAG(gen, pred, projection=projection)
```

<!-- TODO[pi]: write guidance workflow — cover TAG vs DEG tradeoffs, temperature tuning, GuidanceProjection setup -->

---

## Step 5: Sample

Generate candidate sequences using one of the available samplers.

=== "Ancestral (any-order)"

    ```python
    from proteingen import sample

    masked = ["<mask>" * 100] * 8
    sequences = sample(guided, masked)["sequences"]
    ```

    Unmasks positions one at a time in random order. Simple and effective. DEG-aware — automatically passes position info.

=== "Linear interpolation"

    ```python
    from proteingen import sample_ctmc_linear_interpolation

    sequences = sample_ctmc_linear_interpolation(guided, masked, n_steps=100)
    ```

    Euler integration in token-probability space. Interpolates between noise and signal.

=== "Euler (legacy)"

    ```python
    from proteingen.sampling import sample_flow_matching_legacy

    sequences = sample_flow_matching_legacy(guided, masked, dt=0.01)
    ```

    Legacy flow-matching sampler with `dt` and `x1_temp` parameters. Kept for reproducing original stability guidance demo results.

!!! tip "Sampling tips"
    - `n_parallel=1` is the default and most robust setting for ancestral sampling. DEG does not yet support `n_parallel > 1`.
    - All samplers move input to `model.device` automatically and return strings by default.
    - Ancestral sampling modifies the input tensor in-place — pass a copy if you need the original.

<!-- TODO[pi]: write sampling workflow — cover sampler choice, batch_size, temperature, evaluation metrics -->

---

## Step 6: Evaluate

Assess whether generated sequences are trustworthy before committing to wet-lab validation.

!!! note "Coming soon"
    Detailed workflow for evaluation — sequence diversity, predicted fitness distributions, structural plausibility checks, comparison to training data.

<!-- TODO[pi]: write evaluation workflow — cover metrics, plots, sanity checks, red flags -->
