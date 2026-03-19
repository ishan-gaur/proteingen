# Models

## Available Models

| Model | Class | Source | Weights | Conditioning | Output |
|-------|-------|--------|---------|-------------|--------|
| ESMC (300m/600m) | `proteingen.models.ESMC` | [EvolutionaryScale/esm](https://github.com/evolutionaryscale/esm) | HuggingFace: `esmc_300m`, `esmc_600m` | None (masked LM) | `(B, L, 64)` logits |
| ESM3 | `proteingen.models.ESM3` | [EvolutionaryScale/esm](https://github.com/evolutionaryscale/esm) | HuggingFace: `esm3-open` | Structure (atom37 coords) | `(B, L, 64)` logits |
| ESM3IF | `proteingen.models.ESM3IF` | [EvolutionaryScale/esm](https://github.com/evolutionaryscale/esm) | HuggingFace: `esm3-open` | Structure (atom37 coords, **required**) | `(B, L, 64)` logits |
| StabilityPMPNN | `proteingen.models.rocklin_ddg.PreTrainedStabilityPredictor` | In-repo (ProteinMPNN-based) | Manual download | Structure (PDB → featurize) | Scalar stability logit |

### ESMC

ESM-C masked language model, available in 300m and 600m parameter variants. Wraps the ESM model as a `TransitionModelWithEmbedding`, providing both generative sampling and differentiable embedding extraction (used by `LinearProbe`).

- **Output dim**: 64 (33 real vocab + 31 alignment padding)
- **Embedding dim**: 960 (300m) — set dynamically from model weights
- **LoRA support**: yes, via `apply_lora()`

### ESM3

ESM3-open (1.4B parameters). Supports optional structure conditioning via atom37 coordinates.

- **Output dim**: 64
- **Embedding dim**: 1536
- **Structure conditioning**: pass `coords_RAX` (atom37 format, shape `(L, 37, 3)`) via `set_condition_()` or `conditioned_on()`. Runs VQ-VAE encoder once, caches structure tokens.
- **LoRA support**: yes, via `apply_lora()` — use `lr ≤ 1e-4` to avoid mode collapse; `batch_size ≤ 16` with bfloat16 AMP to avoid OOM from geometric attention.

### ESM3IF

ESM3 configured for inverse folding (structure-conditioned sequence generation). Structure conditioning is **required** — the model predicts sequences given a backbone.

### StabilityPMPNN

ProteinMPNN-based stability predictor from the Rocklin lab. Predicts thermodynamic stability (ΔΔG) from structure + sequence.

- **Encode/decode split**: `encode_structure()` is expensive (run once per structure), `decode()` is cheap (run per sequence sample). This maps to ProbabilityModel's `preprocess_observations` / `forward` pattern.
- **Tokenizer**: `MPNNTokenizer` (21 tokens: 20 standard AAs + UNK)

<!-- TODO[pi]: figure out model storage strategy — should we convert all models to HuggingFace format? Use torch hub cache as default (like evodiff loads from zenodo)? Or extend the save/from_checkpoint interface to support both HF and custom loading? This affects how contributed models are distributed. -->

<!-- TODO[pi]: add models from TODO.md — Progen, Dayhoff, Evodiff, METL, SaProt -->
