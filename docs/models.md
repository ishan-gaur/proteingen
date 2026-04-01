# Models

## Generative Models

| Model | Class | Source | Conditioning | Output |
|-------|-------|--------|-------------|--------|
| ESMC (300m/600m) | `proteingen.models.ESMC` | [EvolutionaryScale/esm](https://github.com/evolutionaryscale/esm) | None (masked LM) | `(B, L, 64)` logits |
| ESM3 | `proteingen.models.ESM3` | [EvolutionaryScale/esm](https://github.com/evolutionaryscale/esm) | Structure (atom37 coords) | `(B, L, 64)` logits |
| DPLM-2 | `proteingen.models.DPLM2` | [bytedance/dplm](https://github.com/bytedance/dplm) | None (masked diffusion) | `(B, L, 8229)` logits |
| ESM Forge API | `proteingen.models.ESMForgeAPI` | [EvolutionaryScale Forge](https://forge.evolutionaryscale.ai) | Structure (ESM3 only) | `(B, L, 64)` logits |
| ProteinMPNN | *coming soon* | [dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN) | Structure (required) | Sequence logits |
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


