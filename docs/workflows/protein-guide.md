# ProteinGuide Workflow

End-to-end recipe for guided protein sequence generation. Each step links to a detailed sub-workflow.

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

=== "EmbeddingMLP"
    Learnable amino acid embeddings + MLP. Can initialize from pretrained via PCA. Good for small-vocabulary tasks.

=== "OneHotMLP"
    One-hot encoded sequences + MLP. No pretrained knowledge, but fully flexible.

=== "Custom"
    Subclass `PredictiveModel` directly and implement `forward` + `format_raw_to_logits`.

!!! note "Coming soon"
    Detailed workflow for each architecture choice, including code templates and training tips.

<!-- TODO[pi]: write predictive model setup workflow with code templates -->

---

## Step 3: Train a noisy classifier

Train your predictive model on noisy (partially masked) inputs so it works well during the iterative denoising/sampling process.

!!! note "Coming soon"
    Detailed workflow for noisy classifier training — noise schedules, masking strategies, training loops, and diagnostics.

<!-- TODO[pi]: write training workflow — cover noise schedules, GuidanceDataset, training loop, wandb logging -->

---

## Step 4: Combine with TAG or DEG

Combine your trained predictive model with a generative model using guidance:

=== "TAG (Taylor-Approximate Guidance)"
    Uses gradients of the predictive model to steer generation. Works best when gradients are reliable (e.g., small models, LoRA-adapted backbones).

=== "DEG (Discrete Enumeration Guidance)"
    Evaluates all 20 amino acids at each position and reweights. More robust than TAG for frozen-LM probes where gradients through the transformer are unreliable.

!!! note "Coming soon"
    Detailed workflow for setting up TAG/DEG, choosing guidance scale and temperature, and diagnosing guidance quality.

<!-- TODO[pi]: write guidance workflow — cover TAG vs DEG tradeoffs, temperature tuning, GuidanceProjection setup -->

---

## Step 5: Sample

Generate candidate sequences using one of the available samplers:

=== "Ancestral (any-order)"
    `sample_any_order_ancestral` — unmask positions one at a time in random order. Simple and effective.

=== "Euler"
    Flow-matching Euler integration with configurable `dt`. Used in the stability guidance demo.

=== "Linear interpolation"
    Interpolate between noise and signal in token space.

!!! note "Coming soon"
    Detailed workflow for sampling — choosing a sampler, batch size, and evaluating output quality.

<!-- TODO[pi]: write sampling workflow — cover sampler choice, batch_size, temperature, evaluation metrics -->

---

## Step 6: Evaluate

Assess whether generated sequences are trustworthy before committing to wet-lab validation.

!!! note "Coming soon"
    Detailed workflow for evaluation — sequence diversity, predicted fitness distributions, structural plausibility checks, comparison to training data.

<!-- TODO[pi]: write evaluation workflow — cover metrics, plots, sanity checks, red flags -->
