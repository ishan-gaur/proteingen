# Workflows

Workflows are step-by-step recipes for common protein design tasks. You can follow them yourself or with an agent. If using an agent, we recommend using Skills and having the agent reproduce the plots we have in the workflows so that you can check that everything looks right.

## Skills

Skills are structured instructions that AI coding agents can load on-demand to perform specific tasks. They live in [`.agents/skills/`](https://github.com/ishan-gaur/proteingen/tree/main/.agents/skills) and follow the [Agent Skills standard](https://agentskills.io/specification). Feel free to write your own when making your own workflows and sharing them with the community by [contributing](contributing/index.md) to ProteinGen.

!!! tip "Setup"
    To setup skills, make sure to follow the [setup instructions](../setup.md#add-agent-skills), making sure to adapt them to the particular of what your model provider, e.g. Anthropic, OpenAI, Z.ai, etc. expect.

| Skill | Description |
|-------|-------------|
| [`add-generative-model`](https://github.com/ishan-gaur/proteingen/blob/main/.agents/skills/add-generative-model/SKILL.md) | Integrate a new generative (transition) model into the library |
| [`add-predictive-model`](https://github.com/ishan-gaur/proteingen/blob/main/.agents/skills/add-predictive-model/SKILL.md) | Integrate a new predictive model into the library |
| [`likelihood-curves`](https://github.com/ishan-gaur/proteingen/blob/main/.agents/skills/likelihood-curves/SKILL.md) | Evaluate and plot log-likelihood trajectories for generative models |

## Available Workflows

### [ProteinGuide](protein-guide.md)

The main workflow: go from a fitness function or dataset to guided protein sequence generation. Covers data organization, training a predictive model, combining it with a generative model via TAG or DEG, and sampling candidates.

### [MSA → Sequence + Structure Dataset](msa-to-dataset.md)

Turn a multiple sequence alignment into a training-ready dataset. Covers loading FASTA files, stripping gaps, folding with AF3, and encoding ESM3 structure tokens.

### [Fine-tuning a Generative Model](finetune-generative.md)

Fine-tune ESM3 or ESMC with LoRA for sequence-only MLM or structure-conditioned inverse folding. Includes training loop patterns, example results, and checkpoint management.

### [Evaluating with Likelihood Curves](likelihood-curves.md)

Measure model quality by tracking log-probability trajectories under progressive unmasking. Compare structure-conditioned vs sequence-only, pretrained vs fine-tuned, and track improvement over training.

### [Stability-Guided Inverse Folding](stability-guidance.md)

Redesign a protein backbone for thermodynamic stability using ESM3 + a noisy stability classifier via TAG. Walks through the full pipeline from structure conditioning through evaluation, with results on the Rocklin cluster 146 topology.

---
