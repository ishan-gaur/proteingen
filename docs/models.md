# Models

## Generative Models

| Model | Class | Source | Conditioning | Output |
|-------|-------|--------|-------------|--------|
| ESMC (300m/600m) | `proteingen.models.ESMC` | [EvolutionaryScale/esm](https://github.com/evolutionaryscale/esm) | None (masked LM) | `(B, L, 64)` logits |
| ESM3 | `proteingen.models.ESM3` | [EvolutionaryScale/esm](https://github.com/evolutionaryscale/esm) | Structure (atom37 coords) | `(B, L, 64)` logits |
| DPLM-2 | `proteingen.models.DPLM2` | [bytedance/dplm](https://github.com/bytedance/dplm) | None (masked diffusion) | `(B, L, 8229)` logits |
| ESM Forge API | `proteingen.models.ESMForgeAPI` | [EvolutionaryScale Forge](https://forge.evolutionaryscale.ai) | Structure (ESM3 only) | `(B, L, 64)` logits |
| ProteinMPNN | `proteingen.models.ProteinMPNN` | [Foundry](https://github.com/dauparas/ProteinMPNN) (via `rc-foundry[all]`) | Structure (required) | `(B, L, 22)` logits |
| LigandMPNN | *coming soon* | [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN) | Structure + ligands | Sequence logits |
| SaProt | *coming soon* | [westlake-repl/SaProt](https://github.com/westlake-repl/SaProt) | Structure (Foldseek tokens) | Sequence logits |
| EvoDiff | *coming soon* | [microsoft/evodiff](https://github.com/microsoft/evodiff) | None (discrete diffusion) | Sequence logits |
| AMPLIFY | *coming soon* | [chandar-lab/AMPLIFY](https://github.com/chandar-lab/AMPLIFY) | None (masked LM) | Sequence logits |
| ProGen2 | *coming soon* | [salesforce/progen](https://github.com/salesforce/progen) | None (autoregressive) | Sequence logits |
| Dayhoff | *coming soon* | [microsoft/Dayhoff](https://huggingface.co/microsoft/Dayhoff-170m-UR50) | None (masked LM) | Sequence logits |
| ZymCTRL | *coming soon* | [AI4PD/ZymCTRL](https://huggingface.co/AI4PD/ZymCTRL) | EC number (autoregressive) | Sequence logits |

### ESMC

ESM-C masked language model, available in 300m and 600m parameter variants. Wraps the ESM model as a `TransitionModelWithEmbedding`, providing both generative sampling and differentiable embedding extraction (used by `LinearProbe` and TAG).

- **Output dim**: 64 (33 real vocab + 31 alignment padding — handled automatically by `MaskedModelLogitFormatter`)
- **Embedding dim**: 960 (300m) — set dynamically from model weights
- **LoRA support**: yes, via `apply_lora()`

```python
from proteingen.models import ESMC

model = ESMC("esmc_300m").cuda()  # or "esmc_600m"
```

### ESM3

ESM3-open (1.4B parameters). Supports optional structure conditioning via atom37 coordinates.

- **Output dim**: 64
- **Embedding dim**: 1536
- **LoRA support**: yes — use `lr ≤ 1e-4` to avoid mode collapse

#### Structure conditioning (inverse folding)

ESM3 accepts atom37-format coordinates as conditioning input, enabling structure-conditioned sequence generation (inverse folding). For inference, the `set_condition_()` method runs the VQ-VAE structure encoder once (expensive), then all subsequent calls use cached structure tokens:

```python
from proteingen.models import ESM3
from proteingen.sampling import sample_linear_interpolation

model = ESM3("esm3-open").cuda()
coords = ...  # atom37 format, shape (L, 37, 3)

# Set structure conditioning — VQ-VAE encodes coordinates once
model.set_condition_({"coords_RAX": coords})

# Sample sequences conditioned on the structure
init_tokens = tokenizer(["<mask>" * L], return_tensors="pt")["input_ids"].cuda()
sequences = sample_linear_interpolation(model, init_tokens, n_steps=100)

# Or use a context manager (reverts conditioning on exit)
with model.conditioned_on({"coords_RAX": coords}):
    log_probs = model.get_log_probs(seq)
```

For training with per-sample structures (e.g. fine-tuning on a family of AF3-predicted structures), pass observations directly through the collator instead of using `set_condition_()`. See the [conditioning docs](reference/probability_model.md#conditioning) and the [fine-tuning workflow](workflows/finetune-generative.md) for the full pattern.

!!! warning "Structure conditioning is length-locked"
    `set_condition_()` preprocesses coordinates to fixed-length structure tokens (L+2 with BOS/EOS). All subsequent calls must use sequences of exactly that length, or you get a shape mismatch. For training with variable-length sequences, use per-sample conditioning via the collator.

!!! warning "Lazy VQ-VAE loading and parameter freezing"
    ESM3's structure encoder (`_structure_encoder`) is loaded lazily on first `set_condition_()` call. If you froze parameters before conditioning (e.g. via `apply_lora()`), the encoder's ~30M parameters won't be frozen. **Always set up conditioning before freezing**, or re-freeze after.

#### Memory considerations

ESM3's geometric attention computes pairwise (L×L) tensors. For a sequence of length 297:

- `batch_size ≥ 32` OOMs on 48GB GPU
- `batch_size = 16` with bfloat16 AMP is optimal (~32s/epoch for LoRA training)

#### LoRA specifics

| Rank (r) | Trainable params | % of 1.4B |
|----------|-----------------|-----------|
| 8 | ~9.8M | 0.69% |
| 4 | ~4.9M | 0.35% |
| 2 | ~2.5M | 0.18% |

### DPLM-2

ByteDance's discrete diffusion protein language model ([DPLM-2](https://arxiv.org/abs/2410.13782), ICLR'25). Uses masked diffusion over an extended vocabulary that includes both amino acid and structure codebook tokens. Currently supports sequence-only mode.

- **Output dim**: 8229 (33 AA tokens + 8196 structure tokens — formatted by `MaskedModelLogitFormatter` to expose only the 20 standard AAs + mask)
- **Embedding dim**: 640 (150m), 1280 (650m), 2560 (3b) — set dynamically from model weights
- **LoRA support**: yes, via `apply_lora()`
- **Structure conditioning**: not yet supported (joint sequence+structure generation requires upstream's structure VQ-VAE tokenizer)

Available checkpoints (HuggingFace hub, `airkingbd/`):

| Checkpoint | Params | Hidden | Layers |
|---|---|---|---|
| `airkingbd/dplm2_150m` | 150M | 640 | 30 |
| `airkingbd/dplm2_650m` | 650M | 1280 | 33 |
| `airkingbd/dplm2_3b` | 3B | 2560 | 36 |

```python
from proteingen.models import DPLM2

model = DPLM2("airkingbd/dplm2_650m").cuda()  # default checkpoint
log_probs = model.get_log_probs_from_string(["ACDEFGHIK"])
```

DPLM-2 works with the same sampling, guidance, and probe infrastructure as the ESM models:

```python
from proteingen.sampling import sample_linear_interpolation

init_tokens = model.tokenizer(["<mask>" * 100], return_tensors="pt")["input_ids"].cuda()
sequences = sample_linear_interpolation(model, init_tokens, n_steps=100)
```

!!! warning "Untied embedding weights"
    The HuggingFace config for DPLM-2 incorrectly sets `tie_word_embeddings=True`. The `DPLM2` wrapper overrides this to `False` before loading — if you load the model manually via `AutoModelForMaskedLM`, you'll get wrong logits.

### ProteinMPNN

Structure-conditioned autoregressive sequence design model ([Dauparas et al., 2022](https://www.science.org/doi/10.1126/science.add2187)). Wraps the Foundry implementation (`rc-foundry[all]`) as a `TransitionModelWithEmbedding`.

- **Output dim**: 22 (20 standard AAs + UNK + mask token column — mask column padded with -inf)
- **Embedding dim**: 128
- **Parameters**: 1.7M (small, runs fast on CPU)
- **Structure conditioning**: **required** — the model is a structure-conditioned inverse folding model
- **LoRA support**: yes, via `apply_lora()`

Available checkpoints (from Foundry registry):

| Checkpoint | Description |
|---|---|
| `proteinmpnn` | Standard ProteinMPNN (default) |
| `solublempnn` | Trained on soluble proteins only |

```python
from proteingen.models import ProteinMPNN

model = ProteinMPNN("proteinmpnn")  # or "solublempnn"
```

#### Structure conditioning

ProteinMPNN **requires** backbone structure as input. The conditioning dict provides atom coordinates in the standard 37-atom representation:

```python
import torch

# Structure with L residues
L = 100
condition = {
    "X": coords,         # (L, 37, 3) — atom coordinates (backbone at positions 0-3)
    "X_m": coord_mask,   # (L, 37) — bool mask for which atoms exist
    "R_idx": res_idx,    # (L,) — residue indices within each chain
    "chain_labels": chains,  # (L,) — integer chain IDs (0, 1, 2, ...)
    "residue_mask": mask,    # (L,) — bool mask for valid residues
}

# Set conditioning — runs graph featurization + encoder once
model.set_condition_(condition)

# Get log probabilities for a sequence
tokens = model.tokenizer("A" * L)["input_ids"]
log_probs = model.get_log_probs(tokens)  # (1, L, 22)

# Or use context manager
with model.conditioned_on(condition):
    log_probs = model.get_log_probs(tokens)
```

The wrapper precomputes the MPNN encoder features (graph featurization + message passing on the structure) once during `set_condition_()`. All subsequent forward passes reuse these cached features, making repeated calls cheap.

#### Decoder mode

The wrapper uses **conditional_minus_self** teacher forcing: each position's logit is conditioned on all other positions' sequence identities and the structure, but not on its own identity. This gives a pseudo-likelihood at every position, compatible with masked-diffusion sampling.

For positions with mask tokens in the input, the model outputs a distribution over the 20 standard amino acids. For positions with regular AA tokens, the `MaskedModelLogitFormatter` forces the output to a delta on the input token.

#### Sampling

ProteinMPNN integrates with proteingen's sampling infrastructure:

```python
from proteingen.sampling import sample_linear_interpolation

# Start from all-masked sequence
init_tokens = model.tokenizer(["<mask>" * L])["input_ids"]
with model.conditioned_on(condition):
    sequences = sample_linear_interpolation(model, init_tokens, n_steps=50)
```

#### Differentiable embedding (TAG guidance)

The embedding path supports gradients through the sequence → decoder → logits path. Structure encoder features are treated as constants (precomputed). Mask token positions receive a zero sequence embedding, equivalent to MPNN's "unknown sequence" initialization:

```python
# TAG guidance with ProteinMPNN
emb = model.embed(tokens)       # (B, L, 128) — differentiable
out = model.embedding_to_outputs(emb)  # (B, L, 22) — gradients flow back
```

#### Wrapper design

ProteinMPNN is an autoregressive model — its native forward pass decodes one residue at a time in a random order — but proteingen wraps it to behave like a masked language model so it fits the same `get_log_probs` / sampling / TAG interface as ESM and DPLM-2. Here's how:

**Encoder/decoder caching.** The MPNN architecture has two halves: an encoder that processes the backbone graph (no sequence information) and a decoder that predicts sequence conditioned on the encoder output. Since the structure doesn't change between calls, `set_condition_()` runs the graph featurization and encoder once, caching the node features `h_V` (L, 128), edge features `h_E` (L, K, 128), and neighbor indices `E_idx` (L, K). Every subsequent `forward` / `get_log_probs` / `embed` call only runs the lightweight decoder.

**Conditional-minus-self causality.** MPNN's decoder supports several causal masking modes. The wrapper uses `conditional_minus_self`: each position sees every other position's ground-truth sequence embedding but *not* its own. This gives a pseudo-likelihood estimate — P(residue_i | structure, all other residues) — at every position in a single teacher-forcing pass. This is the same mode the original ProteinMPNN uses for scoring sequences.

**Logit padding to 22-dim.** MPNN natively outputs 21-dim logits (20 standard AAs + UNK). The wrapper appends a -inf column for the mask token (index 21), producing 22-dim output. This lets `MaskedModelLogitFormatter` enforce the right masking semantics: mask-token inputs predict a distribution over the 20 standard AAs; regular AA inputs predict a delta on themselves. The padding is invisible to downstream code — `get_log_probs` returns a valid 22-dim log-probability distribution at every position.

**Soft embeddings for TAG.** The differentiable embedding path replaces the discrete `W_s(token_ids)` lookup with a continuous `ohe @ W_s.weight` matmul, then runs the same decoder layers with the cached encoder features. For mask-token positions, the OHE slice over MPNN's 21 tokens is all zeros, yielding a zero sequence embedding — matching MPNN's native "unknown sequence" initialization. Gradients flow from the output logits back through the decoder and matmul to the input OHE, enabling TAG guidance.

!!! info "Validated against Foundry"
    The wrapper is tested against Foundry's own MPNN pipeline on PDB 1YCR (p53/MDM2, 2 chains, 98 residues) and produces **bitwise-identical logits** — 0.0 max absolute difference, 100% argmax agreement across all positions.

### ESM Forge API

Remote inference via the [EvolutionaryScale Forge](https://forge.evolutionaryscale.ai) API. Wraps Forge clients to provide the same `get_log_probs` interface as local ESM models — no local weights needed. Automatically selects the right client (ESM3 vs ESMC) based on model name.

- **Output dim**: 64 (same tokenizer as local ESM models)
- **Available models**: `esmc-6b-2024-12`, `esm3-open-2024-03`, and others on the Forge platform
- **Structure conditioning**: ESM3 models only (remote VQ-VAE encoding via `preprocess_observations`)
- **LoRA / fine-tuning**: not supported (remote inference)
- **Gradients**: not available (no `embed`, no TAG guidance)

```python
import os
from proteingen.models import ESMForgeAPI

# ESMC — no conditioning, just masked LM
model = ESMForgeAPI("esmc-6b-2024-12", token=os.environ["FORGE_TOKEN"])
log_probs = model.get_log_probs_from_string(["ACDEFGHIK"])
```

#### Structure conditioning (ESM3 only)

For ESM3 models, structure conditioning works through the same `preprocess_observations` / `set_condition_` interface as the local ESM3 wrapper. The VQ-VAE encoding happens remotely:

```python
model = ESMForgeAPI("esm3-open-2024-03", token=os.environ["FORGE_TOKEN"])
coords = ...  # atom37 format, shape (L, 37, 3)

model.set_condition_({"coords_RAX": coords})
log_probs = model.get_log_probs_from_string(["ACDEFGHIK" + "A" * (L - 9)])
```

!!! note "When to use Forge vs local models"
    Forge is useful for accessing larger models (e.g. ESMC-6B) that don't fit in local GPU memory, or for quick experiments without downloading weights. For training, LoRA, TAG guidance, or any workflow requiring gradients, use the local model wrappers instead.

---

## Predictive Models

| Model | Class | Source | Conditioning | Output |
|-------|-------|--------|-------------|--------|
| StabilityPMPNN | `proteingen.models.rocklin_ddg.PreTrainedStabilityPredictor` | [ProteinGuide](https://arxiv.org/abs/2505.04823) (ProteinMPNN-based) | Structure (PDB → featurize) | Scalar stability logit |
| METL | *coming soon* | [gelman-lab/METL](https://github.com/gelman-lab/METL) | Structure (biophysics-pretrained) | Fitness scalar |
| Tranception | *coming soon* | [OATML-Markslab/Tranception](https://github.com/OATML-Markslab/Tranception) | MSA (retrieval-augmented) | Fitness scalar |

### StabilityPMPNN

ProteinMPNN-based stability predictor from the [ProteinGuide paper](https://arxiv.org/abs/2505.04823). Trained on the [Rocklin Megascale stability dataset](https://www.nature.com/articles/s41586-022-04604-z) to predict thermodynamic stability (ΔΔG) from structure + sequence.

- **Encode/decode split**: `encode_structure()` runs once per structure (expensive), `decode()` runs per sequence sample (cheap). This maps naturally to ProbabilityModel's `preprocess_observations` / `forward` pattern.
- **Tokenizer**: `MPNNTokenizer` with 21 tokens (20 standard AAs + UNK). When used with TAG, `include_mask_token=True` adds `<mask>` at idx 21.

#### Cross-tokenizer behavior

The stability predictor overrides `token_ohe_basis()` so that the `<mask>` token maps to an all-zero OHE row — preserving original PMPNN masking semantics while making mask behavior explicit in the interface. This is a key integration point with TAG's `GuidanceProjection`.

<!-- TODO[pi]: figure out model storage strategy — should we convert all models to HuggingFace format? Use torch hub cache as default (like evodiff loads from zenodo)? Or extend the save/from_checkpoint interface to support both HF and custom loading? This affects how contributed models are distributed. -->


