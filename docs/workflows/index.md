# Workflows

Workflows are step-by-step recipes for common protein design tasks. Each workflow is designed to be followed with an AI coding agent — the instructions include prompts you can give directly to Claude Code.

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
