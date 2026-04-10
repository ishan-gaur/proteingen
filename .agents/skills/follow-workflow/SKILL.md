---
name: follow-workflow
description: Guide the user through planning and implementing a protein design pipeline by following ProtStar workflows. Use when the user says they want to follow a workflow (e.g. ProteinGuide, Continued Pretraining) or wants help planning their library design pipeline. Walks through documentation in-order and recursively, helping the user make decisions at each step before writing code.
---

# Follow Workflow

Guide the user through a ProtStar workflow step-by-step, recursively following module documentation to plan their pipeline before coding it up.

## How This Works

ProtStar documentation is organized around **workflows** (paper-level compositions) and **modules** (reusable building blocks in four areas: Data, Models, Sampling, Evaluation). This skill walks the user through a workflow by:

1. Reading the workflow's top-level page to understand the full pipeline
2. At each step, reading the referenced module page for details
3. Helping the user make decisions specific to their protein/data/goals
4. Only writing code after the plan is clear

**Key principle**: plan first, code second. Each workflow step involves decisions that depend on the user's specific situation. Don't jump to code until the user has committed to their choices.

## Phase 1: Identify the Workflow

Ask the user what they want to do. Map their goal to one of the available workflows.

**Read the workflows overview first:**
```
docs/workflows/index.md
```

Available workflows:
- **ProteinGuide** (`docs/workflows/protein-guide.md`) — guided generation combining a generative model with a property predictor via Bayes' rule. Use when: the user has assay-labeled data and wants to generate variants optimized for a property.
- **Continued Pretraining** (`docs/workflows/continued-pretraining.md`) — specialize a pretrained model to a protein family. Use when: the user wants better unconditional generation for their protein family, or as a first step before ProteinGuide.

If the user's goal doesn't match either workflow, help them compose modules manually — read the module pages under `docs/workflows/` and the reference docs under `docs/reference/`.

## Phase 2: Walk Through the Workflow

Read the workflow's documentation page. Present the user with an overview of all steps, then go through each one in order.

**For each step:**

1. **Read the module page** referenced by that step (e.g., `docs/workflows/training-predictors.md`). If the module references sub-modules, read those too (recursive).
2. **Summarize the key decisions** the user needs to make at this step
3. **Ask the user** about their specific situation (what data they have, what models they want, etc.)
4. **Record their decision** before moving to the next step

### Decision Points by Module Area

**Data decisions:**
- What data do you have? (assay labels, MSA, structures, nothing)
- How should data be split? (random, by mutational distance, by position)
- Do you need to acquire homologs? → read `docs/workflows/msa-acquisition.md`
- Do you need AF3-predicted structures? → read `docs/workflows/msa-to-dataset.md`

**Models decisions:**
- Which generative model? (ESMC, ESM3, DPLM2, PMPNN) → read `docs/models.md`
- Fine-tune or use pretrained? → read `docs/workflows/finetune-generative.md`
- Which predictive model architecture? (LinearProbe, OneHotMLP, EmbeddingMLP) → read `docs/reference/predictive_modeling.md`
- TAG or DEG guidance? → read `docs/reference/guide.md`
- What binary logit function? → read `docs/reference/predictive_modeling.md#binary-logit-functions`

**Sampling decisions:**
- Which sampler? (discrete-time ancestral vs linear interpolation) → read `docs/reference/sampling.md`
- What temperatures? (generator temp, predictor temp) → read `docs/workflows/protein-guide.md` temperature tuning section
- How many positions to mask? (full mask vs partial for mutagenesis)
- How many sequences to generate?

**Evaluation decisions:**
- What metrics matter? (predicted activity, diversity, structural quality)
- Do you have an oracle or ground truth? → read `docs/reference/evaluation.md`
- Do you need AF3 structural validation?
- What are the success criteria for proceeding to wet-lab?

## Phase 3: Summarize the Plan

After all decisions are made, present a summary mapping each decision to the four modules:

```
## Pipeline Plan

### Data
- Source: [user's data description]
- Splits: [chosen split strategy]
- Homologs: [yes/no, source]

### Models
- Generative: [model name], [pretrained/fine-tuned]
- Predictive: [architecture], [binary logit function]
- Guidance: [TAG/DEG]

### Sampling
- Sampler: [chosen sampler]
- Temperatures: gen=[X], pred=[Y]
- Library size: [N sequences]

### Evaluation
- Metrics: [chosen metrics]
- Oracle: [yes/no, architecture]
- Structural validation: [yes/no]
```

Ask the user to confirm or modify the plan.

## Phase 4: Implement

Once the plan is confirmed, implement step-by-step. For each step:

1. Read the relevant example for reference (find the closest match in `docs/examples/`)
2. Write the code for that step
3. Run it and verify outputs before proceeding

**Key examples to reference:**
- Unconditional sampling: `examples/unconditional_sampling.py`
- Structure-conditioned sampling: `examples/esm3_structure_conditioned_sampling.py`
- Fine-tuning: `examples/finetune_esm3/`
- Guided generation: `examples/stability_guidance/main.py`
- Likelihood curves: `examples/trpb_likelihood_curves.py`
- Linear probe + DEG: `examples/trpb_linear_probe.py`
- Full ProteinGuide pipeline: see `examples/pbrr_walkthrough/` in the `spawn/pbrr-walkthrough` branch

## Gotchas

- **Don't skip evaluation steps** — it's tempting to go straight from training to generation. Always validate predictor–oracle agreement before trusting guided samples.
- **Temperature tuning is critical** — the default temperatures are rarely optimal. Guide the user through trying 2-3 settings and comparing results.
- **Start simple** — if the user has never used ProtStar before, start with unconditional sampling to verify their setup works, then add complexity.
- **ESM3 + AMP** — always use `--amp` (bfloat16) for ESM3 training. fp32 logits overflow.
- **DEG + n_parallel** — DEG doesn't support unmasking multiple positions at once. Use `n_parallel=1`.
- **LoRA rank** — r=4 is a good default. Higher ranks risk overfitting on small datasets.
