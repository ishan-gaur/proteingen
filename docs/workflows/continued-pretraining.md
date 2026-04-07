# Continued Pretraining

Specialize a pretrained generative model to a protein family using homologous sequences, optionally with predicted structures for inverse folding.

## Overview

```
Acquire homologs → Build dataset → Fine-tune with LoRA → Evaluate with likelihood curves
```

This workflow is for when you want a better base model for your protein family *before* doing any property-guided generation. It's a common first step in [ProteinGuide](protein-guide.md) and useful on its own for unconditional generation of family-like sequences.

---

## Step 1: Acquire homologs

Obtain an MSA of sequences homologous to your protein of interest.

→ See [MSA Acquisition](msa-acquisition.md)

## Step 2: Build the dataset

Strip gaps, filter by length, and optionally fold with AF3 for structure conditioning.

→ See [MSA → Dataset](msa-to-dataset.md)

## Step 3: Fine-tune with LoRA

LoRA fine-tune ESM3 or ESMC on the homolog sequences.

→ See [Fine-tuning Generative Models](finetune-generative.md)

**Key decisions:**

- **Sequence-only vs. inverse folding** — use sequence-only if you don't have structures or want faster training. Use inverse folding if you have AF3-predicted structures and want the model to learn the structure→sequence mapping for your family.
- **LoRA rank** — `r=4` is a good default for specialization. Higher ranks give more capacity but risk overfitting on small MSAs.
- **Train to convergence** — for continued pretraining, you want the model to fully learn the family distribution. Watch the likelihood curves plateau.

## Step 4: Evaluate

Compare pretrained vs. fine-tuned models using likelihood curves on held-out homologs.

→ See [Likelihood Curves](likelihood-curves.md)

**What to look for:**

- Fine-tuned model should have higher log-probs on held-out homologs at all noise levels
- If using structure conditioning, the gap between struct-conditioned and seq-only should widen after fine-tuning
- Sequence-only log-probs should stay roughly flat (no memorization)
