# ProGen3 — Design Notes

Autoregressive protein language model from Profluent Bio. Sparse mixture-of-experts (MoE) architecture trained on 3.4B protein sequences.

## Dependencies

- **Core**: [generative_modeling.md](../../generative_modeling.md) — `ProGen3(GenerativeModel)` with `PassThroughLogitFormatter`
- **External**: `progen3` package (git install from `Profluent-AI/progen3`), which pulls in `megablocks`, `flash_attn`, `transformers`

## Used By

- `docs/examples/autoregressive-generation.md` — unconditional generation example
- `docs/models.md` — model docs
- `tests/test_progen3.py` — test suite

## Architecture

ProGen3 is a **causal** (autoregressive) language model, fundamentally different from the masked models (ESMC, DPLM2) in proteingen. It generates proteins left-to-right.

### Forward pass

`ProGen3ForCausalLM.forward(input_ids, position_ids, sequence_ids)` → `MoeCausalOutputWithPast`:
- `input_ids`: token IDs including `<bos>`, direction tokens (`1`/`2`), amino acids, `<eos>`
- `position_ids`: absolute position indices (required, not inferred from input_ids)
- `sequence_ids`: which sequence each token belongs to (always 0 for single-sequence inference)
- Returns `.logits` of shape `(S, P, 134)` — next-token predictions at each position

### Tokenizer

Uses a custom `tokenizers.Tokenizer` (not HF `PreTrainedTokenizer`). Vocab size 134:
- 0: `<pad>`, 1: `<bos>`, 2: `<eos>`, 3: `<bos_glm>`, 4: `<eos_span>`, 5: `<mask>`
- 6: `1` (N→C direction), 7: `2` (C→N direction)
- 8–33: amino acids A–Z (including non-standard B, J, O, U, X, Z)
- 34–133: `<span_0>` through `<span_99>` (GLM infilling tokens)

The `_ProGen3TokenizerAdapter` wraps this to provide the HF-compatible interface expected by `GenerativeModel`.

### Encoding convention

A protein sequence `"ACDE"` is encoded as: `<bos> 1 A C D E 2 <eos>`. The `1` and `2` tokens are direction markers — `1` = N→C (forward), `2` = C→N (reverse). The model is trained to generate in both directions.

### Generation

Uses HuggingFace's `model.generate()` with nucleus sampling (top-p). The prefix is `<bos> 1 [prompt_AAs]` and the model generates until it outputs `<eos>`. The `2` direction token typically precedes `<eos>` in a valid completion.

### Scoring

Sequences are scored bidirectionally (N→C + C→N average) following the ProGen3 paper's approach. This gives a more robust log-likelihood estimate than unidirectional scoring.

## Available Checkpoints

| Checkpoint | Params | Hidden | Layers | HF Hub |
|---|---|---|---|---|
| `Profluent-Bio/progen3-112m` | 112M | 768 | 12 | [link](https://huggingface.co/Profluent-Bio/progen3-112m) |
| `Profluent-Bio/progen3-3b` | 3B | 2048 | 40 | [link](https://huggingface.co/Profluent-Bio/progen3-3b) |

## Gotchas

- **Requires Flash Attention + megablocks**: the `progen3` package imports `flash_attn.ops.triton.layer_norm` and `megablocks` at model load time. These require compatible CUDA hardware (tested on A100/H100 with 40GB+ VRAM).
- **bfloat16 only**: the model is trained in bfloat16 and the MoE router math assumes it. Don't load in float32.
- **No mask-based workflows**: ProGen3 is autoregressive, not masked. The `sample()`, `sample_ctmc_linear_interpolation()`, TAG guidance, and `LinearProbe` workflows don't apply.
- **No LoRA**: the sparse MoE architecture with megablocks kernels isn't compatible with PEFT LoRA injection.
- **`position_ids` required**: unlike HF GPT-2/LLaMA where position_ids are optional, ProGen3's forward requires explicit position_ids. The wrapper handles this automatically.
- **`sequence_ids` required**: used for MSA-style multi-sequence attention. Always 0 for single-sequence inference.
