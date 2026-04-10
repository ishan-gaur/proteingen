# Core Library — Agent Notes

ProtStar's core abstractions for probabilistic protein modeling, sampling, and guidance.

## Architecture Overview

Inheritance chain: `ProbabilityModel` → `GenerativeModel` → `GenerativeModelWithEmbedding` (generative side) and `ProbabilityModel` → `PredictiveModel` (predictive side). Guidance (`TAG`/`DEG`) combines both to steer sampling.

```
ProbabilityModel (ABC)          — temp, conditioning, get_log_probs
├── GenerativeModel (concrete)  — wraps nn.Module + tokenizer + logit_formatter
│   └── GenerativeModelWithEmbedding (ABC) — adds differentiable embedding path
│       ├── ESMC, ESM3          — concrete model implementations (models/)
│       └── LinearProbe         — frozen embed_model + trainable head
├── PredictiveModel (ABC)       — binary logit pattern, OHE, grad_log_prob
│   ├── OneHotMLP, EmbeddingMLP, PairwiseLinearModel (ABC templates)
│   └── PreTrainedStabilityPredictor (models/rocklin_ddg/ — Listgarten lab / ProteinGuide)
```

Guidance: `TAG` (gradient-based) and `DEG` (enumeration-based) consume a `PredictiveModel` and modify a `GenerativeModel`'s log probs during sampling.

Sampling: `sample`, `sample_flow_matching_legacy`, etc. orchestrate the generation loop, calling `get_log_probs` on the (optionally guided) generative model.

## Component Design Docs

Per-component design, dependencies, gotchas, and checklists:

- [probability_model.md](probability_model.md) — `ProbabilityModel` ABC, conditioning, checkpointing
- [generative_modeling.md](generative_modeling.md) — `GenerativeModel`, `GenerativeModelWithEmbedding`, `LogitFormatter`, `MPNNTokenizer`, LoRA
- [predictive_modeling.md](predictive_modeling.md) — `PredictiveModel`, `LinearProbe`, `OneHotMLP`/`EmbeddingMLP`, binary logit functions, PCA init
- [guide.md](guide.md) — `TAG`, `DEG`, `GuidanceProjection`, cross-tokenizer mapping
- [sampling.md](sampling.md) — sampling algorithms, noise schedules
- [data.md](data.md) — `GuidanceDataset`, schedule functions
- [models/AGENTS.md](models/AGENTS.md) — model registry, per-model links ([esm](models/esm/esm.md), [mpnn](models/mpnn/mpnn.md), [rocklin_ddg](models/rocklin_ddg/rocklin_ddg.md))
- [models/utils.md](models/utils.md) — structure loading (`PDBStructure`, `load_pdb`) and atom encoding API for structure-conditioned models

## Cross-Cutting Concerns

### Tokenization

Three tokenizer ecosystems in the library — mismatches between them are a recurring source of bugs:

- **ESM** (`EsmSequenceTokenizer`): vocab_size=33, indices 0–3 special (`<cls>`, `<pad>`, `<eos>`, `<unk>`), AAs at 4–23, `<mask>`=32. `.vocab` → `dict[str, int]`.
- **PMPNN** (`MPNNTokenizer`): 20 standard AAs + UNK(X) at idx 20, optional `<mask>` at 21. `.vocab` → `dict[str, int]`.
- **Simple 20-AA**: `{aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}` + padding at 20, vocab_size=21.

Cross-tokenizer gotchas are documented in [guide.md](guide.md) (TAG projection) and [predictive_modeling.md](predictive_modeling.md) (OHE basis).

### External Dependencies

- ProteinMPNN via Foundry: `rc-foundry[all]` — provides `mpnn` and `atomworks` packages
- Importing from `atomworks` prints env var warnings (CCD_MIRROR_PATH, PDB_MIRROR_PATH) — harmless
- `peft>=0.13.0` for LoRA adapter support

### Stale Code / Known Tech Debt

- `models/seki_tyrosine_kinase.py` — manually sets `self.input_dim`, shadowing any future property
- `examples/stability_guidance/compare_legacy_sampler.py` — still uses `from dfm.*` imports
- `~/kortemme_tyrosine_kinase_design/train_ohe_mlp.py` — uses old `pca_embed_init` + `initial_embed_weights` API
- `~/PALM/esm-cath/src/esm_cath/model.py` — older ESM consumer, may need updating to composition pattern
