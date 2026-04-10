# Workflows

ProtStar organizes its guides into **workflows** and **modules**.

**Workflows** are end-to-end recipes at the level of a paper's method — they compose multiple modules into a complete pipeline. **Modules** are the reusable building blocks, each implementing a specific mathematical or engineering idea. Modules are organized by the four library design areas: Data, Models, Sampling, and Evaluation.



High-level design strategies that combine modules into complete pipelines.



| Workflow | Description | Key Modules |
|----------|-------------|-------------|
| [ProteinGuide](protein-guide.md) | Guided generation using Bayes' rule to combine a generative model with a property predictor | Fine-tuning, Noisy Predictor Training, Guidance (TAG/DEG), Likelihood Curves |
| [Continued Pretraining](continued-pretraining.md) | Specialize a pretrained model to a protein family using sequence or structural homologs | MSA Acquisition, Fine-tuning, Likelihood Curves |

<!-- Future workflows (not yet documented):
| CbAS | Conditioning by Adaptive Sampling — importance-weighted retraining of a generative model (Brookes et al.) |
| ProteinDPO | Direct Preference Optimization — train a generative model on preference pairs |
| MLDE | Machine Learning-guided Directed Evolution — MCMC in sequence space with a predictor |
| DADO | Data-Augmented Directed Optimization (Bowden et al.) |
-->

## Modules

Reusable building blocks organized by the four library design areas. Workflows reference these modules as sub-steps.

### Data

| Module | Description |
|--------|-------------|
| [MSA → Dataset](msa-to-dataset.md) | Turn an MSA into a training-ready dataset (gap stripping, AF3 folding, structure encoding) |
| [Data Splits](data-splits.md) | Split assay data by mutational distance, activity range, or position for train/eval |
| [MSA Acquisition](msa-acquisition.md) | Tools for obtaining sequence and structure-based MSAs |

### Models

| Module | Description |
|--------|-------------|
| [Fine-tuning Generative Models](finetune-generative.md) | LoRA fine-tuning of ESM3/ESMC for sequence-only or structure-conditioned prediction |
| [Training Predictors](training-predictors.md) | Train oracle and noisy predictive models from assay data |

### Sampling

Sampler selection and configuration is covered in the [sampling reference](../reference/sampling.md). Key decision: use `sample` (discrete-time ancestral) for DEG guidance, `sample_ctmc_linear_interpolation` for TAG guidance.

### Evaluation

| Module | Description |
|--------|-------------|
| [Likelihood Curves](likelihood-curves.md) | Evaluate model quality by tracking log-probability under progressive unmasking |

---

## Skills

Skills are structured instructions that AI coding agents can load on-demand to perform specific tasks. They live in [`.agents/skills/`](https://github.com/ishan-gaur/protstar/tree/main/.agents/skills) and follow the [Agent Skills standard](https://agentskills.io/specification). Feel free to write your own when making your own workflows and sharing them with the community by [contributing](../contributing.md) to ProtStar.

!!! tip "Setup"
    To setup skills, make sure to follow the [setup instructions](../setup.md#add-agent-skills), making sure to adapt them to the particular of what your model provider, e.g. Anthropic, OpenAI, Z.ai, etc. expect.

| Skill | Description |
|-------|-------------|
| [`follow-workflow`](https://github.com/ishan-gaur/protstar/blob/main/.agents/skills/follow-workflow/SKILL.md) | Plan and implement a library design pipeline by walking through workflows step-by-step |
| [`add-generative-model`](https://github.com/ishan-gaur/protstar/blob/main/.agents/skills/add-generative-model/SKILL.md) | Integrate a new generative (transition) model into the library |
| [`add-predictive-model`](https://github.com/ishan-gaur/protstar/blob/main/.agents/skills/add-predictive-model/SKILL.md) | Integrate a new predictive model into the library |
| [`likelihood-curves`](https://github.com/ishan-gaur/protstar/blob/main/.agents/skills/likelihood-curves/SKILL.md) | Evaluate and plot log-likelihood trajectories for generative models |
