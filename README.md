# Guidance

A minimal library for protein sequence guidance using conditional generative models.

> **CAUTION**: This repo is under **very** active development. The main interfaces (`ProbabilityModel`, `TransitionModel`) are stable and you can try the unconditional sampling example. Guided sampling for stability using a predictor trained on the Rocklin dataset will be released shortly. The `PredictiveModel` API is *unstable* as we are still integrating more models and preparing the training API.

## Install

```bash
uv add .
```

```bash
uv pip install -e .
```

```bash
pip install -e .
```

## References

- **ProteinGuide Paper**: [arXiv:2505.04823](https://arxiv.org/abs/2505.04823)
- **ProteinGuide Code**: [github.com/junhaobearxiong/protein_discrete_guidance](https://github.com/junhaobearxiong/protein_discrete_guidance)
- **ProteinMPNN (Foundry)**: [github.com/RosettaCommons/foundry](https://github.com/RosettaCommons/foundry/tree/production)

## Overview

ProteinGuide materializes a conditional generative model on the fly to sample proteins or score sequences according to a desired property. It combines outputs of a predictive model and a generative model using Bayes' rule.

The library implements:
- **Taylor-Approximate Guidance (TAG)**: Approximate guidance method
- **Discrete-time Exact Guidance (DEG)**: Exact guidance for discrete sequences

## Installation

```bash
uv sync
uv run foundry install proteinmpnn
```

## Project Structure

```
src/dfm/
├── probability_model.py    # ProbabilityModel — shared ABC for all models
├── generative_modeling.py  # TransitionModel, LogitFormatter, MaskedModelLogitFormatter, MPNNTokenizer
├── predictive_modeling.py  # PredictiveModel, ClassValuedPredictiveModel, RealValuedPredictiveModel, OneHotMLP, LinearProbe
├── guide.py                # TAG, DEG, TokenizerTranslator
├── sampling.py             # sample_any_order_ancestral
├── data.py                 # GuidanceDataset, NoiseSchedule, schedule functions
└── models/
    ├── esm.py              # ESMC (wraps ESM via TransitionModel)
    ├── rocklin_ddg/         # Stability predictor (StabilityPMPNN, PreTrainedStabilityPredictor)
    └── utils.py            # Shared utilities (pdb_to_atom37_and_seq)

examples/
├── unconditional_sampling.py   # End-to-end ESMC unconditional sampling
└── stability_guidance/         # Stability-guided generation (WIP)

tests/
├── test_logit_formatter.py
├── test_transition_model.py
├── test_guidance_data.py
└── test_esm.py
```

## Core Abstractions

### ProbabilityModel

The shared base class (`nn.Module`, ABC) for all models. Provides:

- **Conditioning** — `set_condition_()`, `set_condition()`, `conditioned_on()` context manager
- **Temperature** — `set_temp_()`, `with_temp()` context manager
- **Log-prob pipeline** — `get_log_probs(x)` chains `collate_observations → forward → format_raw_to_logits → log_softmax(logits / temp)`
- Four abstract methods that subclasses must implement: `forward`, `format_raw_to_logits`, `preprocess_observations`, `collate_observations`

### TransitionModel

A **concrete** `ProbabilityModel` subclass that wraps any `nn.Module` generative model via composition. Takes a `model`, `tokenizer`, and `logit_formatter` at init. Sensible defaults for preprocessing/collation make simple models work out of the box; override for structure-conditioned models.

### PredictiveModel

ABC extending `ProbabilityModel` for predictive models (classifiers, regressors). `ClassValuedPredictiveModel` and `RealValuedPredictiveModel` provide concrete variants.

### Guidance (TAG / DEG)

`guide.py` implements Taylor-Approximate Guidance and Discrete-time Exact Guidance algorithms that combine a `TransitionModel` with a `PredictiveModel` using Bayes' rule. `TokenizerTranslator` bridges different tokenizer vocabularies between the two models.

### Sampling

`sampling.py` provides `sample_any_order_ancestral` for masked discrete sampling using `model.get_log_probs`.

## Usage

### Unconditional Sampling with ESMC

```bash
uv run python examples/unconditional_sampling.py
```

### Running Tests

```bash
uv run python -m pytest tests/ -v
```
