# ProteinGen

<!-- TODO[pi]: write detailed workflow sub-pages (data org, predictive model setup, training, guidance, sampling, evaluation) -->
<!-- TODO[pi]: add evaluation workflow for wetlab scientists to assess generated sequence quality -->

A library for guided protein sequence generation using discrete generative models.

Combines predictive models and generative models via Bayes' rule to sample sequences conditioned on desired properties. Supports ESM-family masked language models, ProteinMPNN, and custom predictive heads.

> **CAUTION**: Under active development. Core abstractions (`ProbabilityModel`, `GenerativeModel`, `PredictiveModel`) are stable. Guided sampling works end-to-end (see TrpB example). Some modules (`guide.py`, `sampling.py`) have stale imports — fixing in progress.

## Install

```bash
uv sync
uv pip install -e .
```

### Optional dependencies

Some features require packages only available via GitHub:

```bash
# ProteinMPNN (for ProteinMPNN / StabilityPMPNN models)
uv pip install "rc-foundry[all]"
uv run foundry install proteinmpnn

# Docs development (live-reloading mkdocs plugin)
uv pip install "mkdocs-liveedit @ git+https://github.com/ishan-gaur/mkdocs-liveedit.git"

# AlphaFold 3 client (for structure prediction workflows)
uv pip install "af3-server @ git+https://github.com/ishan-gaur/af3-server.git"
```

## References

- **ProteinGuide Paper**: [arXiv:2505.04823](https://arxiv.org/abs/2505.04823)
- **ProteinGuide Code**: [github.com/junhaobearxiong/protein_discrete_guidance](https://github.com/junhaobearxiong/protein_discrete_guidance)
- **ProteinMPNN (Foundry)**: [github.com/RosettaCommons/foundry](https://github.com/RosettaCommons/foundry/tree/production)

## Project Structure

```
src/proteingen/
├── probability_model.py    # ProbabilityModel — shared ABC for all models
├── generative_modeling.py  # GenerativeModel, GenerativeModelWithEmbedding, LogitFormatter,
│                           #   MaskedModelLogitFormatter, PassThroughLogitFormatter, MPNNTokenizer
├── predictive_modeling.py  # PredictiveModel, binary logit functions, LinearProbe, OneHotMLP,
│                           #   EmbeddingMLP, PairwiseLinearModel
├── guide.py                # TAG, DEG, TokenizerTranslator
├── sampling.py             # sample
├── data.py                 # GuidanceDataset, NoiseSchedule, schedule functions
└── models/
    ├── esm.py              # ESMC, ESM3 (GenerativeModelWithEmbedding subclasses)
    ├── rocklin_ddg/         # Stability predictor (StabilityPMPNN, PreTrainedStabilityPredictor)
    └── utils.py            # pdb_to_atom37_and_seq (WIP)

examples/
├── unconditional_sampling.py              # ESMC unconditional masked sampling
├── esm3_structure_conditioned_sampling.py # ESM3 with structure conditioning
├── pca_embedding_init.py                  # PCA-init EmbeddingMLP from ESMC
├── trpb_linear_probe.py                   # Train MLP probe + guided sampling on TrpB fitness
├── conditional_scoring.py                 # ESM3 structure-conditioned scoring (uses older API, needs update)
└── stability_guidance/                    # Stability-guided generation (WIP)

tests/
├── test_logit_formatter.py   # 24 tests
├── test_generative_model.py
├── test_embedding_mlp.py     # 49 tests
├── test_pca_embed_init.py    # 21 tests
├── test_esm3.py              # 21 tests
├── test_esmc_lora.py         # 11 tests
├── test_lora.py              # 23 tests
├── test_guidance_data.py     # 3 tests (broken — missing required args)
├── test_esm.py               # needs update for new ESMC API
└── test_sampling.py          # 31 errors (pre-existing)
```

## Core Abstractions

### ProbabilityModel

Shared base class (`nn.Module`, ABC) for all models. Provides:

- **Conditioning** — `set_condition_()`, `set_condition()`, `conditioned_on()` context manager
- **Temperature** — `set_temp_()`, `set_temp()`, `with_temp()` context manager
- **Log-prob pipeline** — `get_log_probs(x)` chains `collate_observations → forward → format_raw_to_logits → log_softmax(logits / temp)`
- **Checkpointing** — `save(path)` / `from_checkpoint(path)` with `_save_args()` for constructor kwargs

Two abstract methods: `forward` and `format_raw_to_logits`. `preprocess_observations` and `collate_observations` have sensible defaults (pass-through and tile-to-batch).

### GenerativeModel

**Concrete** `ProbabilityModel` subclass that wraps any `nn.Module` generative model via composition. Takes `model`, `tokenizer`, and `logit_formatter`.

- **LoRA support** — `apply_lora()`, `save_lora()`, `load_lora()`, `lora_target_modules()`
- Override `format_raw_to_logits` when the wrapped model returns non-tensor output (e.g. ESM dataclasses)

### GenerativeModelWithEmbedding

ABC extending `GenerativeModel` with differentiable embedding support. Subclasses implement two methods:

- `differentiable_embedding(ohe_SPT) → emb_SPD` — OHE through embedding + transformer
- `embedding_to_outputs(emb_SPD) → Any` — embeddings through output head

Provides `embed(seq_SP) → emb_SPD` for extracting embeddings (used by `LinearProbe`).

### PredictiveModel

ABC extending `ProbabilityModel` for models that answer "what is log p(target | sequence)?". Uses a binary logit pattern — `format_raw_to_logits` returns `(B, 2)` logits `[false_logit, true_logit]`.

- **Target management** — `set_target_()`, `set_target()`, `with_target()` context manager
- **Gradient access** — `grad_log_prob(seq_SP)` returns `∂log p(target|x) / ∂OHE` for TAG
- **Binary logit functions** — `categorical_binary_logits`, `binary_logits`, `point_estimate_binary_logits`, `gaussian_binary_logits`

Template subclasses (all ABC — user implements `format_raw_to_logits`):
- **`LinearProbe`** — frozen `GenerativeModelWithEmbedding` + `nn.Linear` head
- **`OneHotMLP`** — flattened one-hot encoding through an MLP
- **`EmbeddingMLP`** — differentiable embedding lookup + MLP, with `init_embed_from_pretrained_pca()` for PCA initialization from pretrained models
- **`PairwiseLinearModel`** — pairwise position interactions

### Guidance (TAG / DEG)

`guide.py` implements Taylor-Approximate Guidance and Discrete-time Exact Guidance. Both are `GenerativeModel` subclasses that combine a generative model with a predictive model using Bayes' rule. `TokenizerTranslator` bridges different tokenizer vocabularies.

### Models

- **`ESMC`** — ESM-C (300m/600m) as masked LM + embedding extractor
- **`ESM3`** — ESM3-open with optional structure conditioning (inverse folding via `set_condition_()` / `conditioned_on()`)

### Sampling

`sample` — ancestral sampling (random or explicit order) using `model.get_log_probs`. Returns a `SamplingTrajectory` with sequences and per-step data.

## Usage

### Unconditional Sampling

```bash
uv run python examples/unconditional_sampling.py
```

### Structure-Conditioned Sampling (ESM3)

```bash
uv run python examples/esm3_structure_conditioned_sampling.py
```

### PCA Embedding Initialization

```bash
uv run python examples/pca_embedding_init.py
```

### Training a Probe + Guided Sampling (TrpB)

```bash
uv run python examples/trpb_linear_probe.py --device cuda
```

### Running Tests

```bash
uv run python -m pytest tests/ -v
```
