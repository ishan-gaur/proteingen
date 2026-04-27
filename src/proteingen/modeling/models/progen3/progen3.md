# ProGen3 — Design Notes

Autoregressive protein language model from Profluent Bio. Sparse mixture-of-experts (MoE) architecture trained on 3.4B protein sequences.

## Dependencies

- **Core**: [generative_modeling.md](../../generative_modeling.md) — `ProGen3(GenerativeModelWithEmbedding)` with `AutoregressiveLogitFormatter`
- **External**: `progen3` package (git install from `Profluent-AI/progen3`), which pulls in `megablocks`, `transformers`

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

### Logit shift

AR models produce `logits[i]` = prediction for position `i+1`. The proteingen framework expects `logits[i]` = prediction for position `i`. `format_raw_to_logits` shifts the logits left by one before applying the formatter.

### AutoregressiveLogitFormatter

Custom logit formatter that makes ProGen3 compatible with `sample()`:
- **Non-mask positions**: one-hot (predict themselves — already decoded or fixed framing)
- **First mask position** (leftmost per sequence): pass through shifted logits, restricted to standard amino acids
- **Remaining mask positions**: all `-inf` (not yet reachable by the autoregressive model)

### Tokenizer

Uses a custom `tokenizers.Tokenizer` (not HF `PreTrainedTokenizer`). Vocab size 134:
- 0: `<pad>`, 1: `<bos>`, 2: `<eos>`, 3: `<bos_glm>`, 4: `<eos_span>`, 5: `<mask>`
- 6: `1` (N→C direction), 7: `2` (C→N direction)
- 8–33: amino acids A–Z (including non-standard B, J, O, U, X, Z)
- 34–133: `<span_0>` through `<span_99>` (GLM infilling tokens)

The `_ProGen3TokenizerAdapter` wraps this to provide the HF-compatible interface expected by `GenerativeModelWithEmbedding`. The `<mask>` token (ID 5) is used as the "to fill" marker for `sample()` integration.

### Encoding convention

A protein sequence `"ACDE"` is encoded as: `<bos> 1 A C D E 2 <eos>`. The `1` and `2` tokens are direction markers — `1` = N→C (forward), `2` = C→N (reverse).

### Embedding path

`ProGen3` extends `GenerativeModelWithEmbedding`:
- `differentiable_embedding(ohe)`: OHE → `embed_tokens` weight matmul → add `embed_seq_id` → transformer layers → final norm → hidden states `(S, P, D)`
- `embedding_to_outputs(hidden)`: `lm_head(hidden)` → logits `(S, P, 134)`
- Embedding at the last real (non-mask) token position serves as a sequence representation for `LinearProbe`

### sample() integration

```python
from proteingen import sample
init_x = ["<mask>" * 100 for _ in range(5)]
result = sample(model, init_x, in_order="left_to_right")
```

Uses `<mask>` tokens as placeholders. The `AutoregressiveLogitFormatter` ensures only the first mask position gets non-trivial logits at each step.

### Generation

`model.generate()` uses HuggingFace's `model.generate()` for open-ended (variable-length) generation with nucleus sampling.

### Scoring

`model.score()` computes bidirectional (N→C + C→N average) log-likelihood following the ProGen3 paper.

## Available Checkpoints

| Checkpoint | Params | Hidden | Layers | HF Hub |
|---|---|---|---|---|
| `Profluent-Bio/progen3-112m` | 112M | 384 | 10 | [link](https://huggingface.co/Profluent-Bio/progen3-112m) |
| `Profluent-Bio/progen3-3b` | 3B | 2048 | 40 | [link](https://huggingface.co/Profluent-Bio/progen3-3b) |

## flash_attn Shim

progen3.modeling imports `flash_attn.ops.triton.layer_norm.rms_norm_fn` at module level. The `flash_attn` pip package's CUDA extension requires GLIBC ≥ 2.32. Since progen3 only uses this one function (a standard RMSNorm), `_ensure_flash_attn_mock()` installs a pure-PyTorch replacement when `flash_attn` can't be imported. The shim also stubs `flash_attn.bert_padding` and `flash_attn_func` that `transformers` tries to import at module level. The attention itself uses PyTorch's built-in SDPA (no flash_attn CUDA kernels needed).

## Gotchas

- **Requires megablocks + grouped_gemm**: the MoE layers use megablocks compiled CUDA kernels. `grouped_gemm` must be installed separately.
- **bfloat16**: the model is trained in bfloat16. The `differentiable_embedding` casts OHE to model dtype for the matmul.
- **No LoRA**: the sparse MoE architecture with megablocks kernels isn't compatible with PEFT LoRA injection.
- **`position_ids` required**: unlike HF GPT-2/LLaMA where position_ids are optional, ProGen3's forward requires explicit position_ids. The wrapper handles this automatically.
- **`sequence_ids` required**: used for MSA-style multi-sequence attention. Always 0 for single-sequence inference.
- **Logit shift**: AR logits are shifted in `format_raw_to_logits` so the framework's position-aligned convention is satisfied. `score()` handles the shift internally using standard AR loss computation.
