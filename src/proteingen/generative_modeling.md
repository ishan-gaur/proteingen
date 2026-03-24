# Generative Modeling — Design Notes

TransitionModel hierarchy, LogitFormatter protocol, MPNNTokenizer, and LoRA support.

## Dependencies

- [probability_model.md](probability_model.md) — `TransitionModel(ProbabilityModel)`, inherits conditioning/checkpointing/temp

## Used By

- [predictive_modeling.md](predictive_modeling.md) — `LinearProbe` wraps `TransitionModelWithEmbedding`
- [guide.md](guide.md) — TAG/DEG subclass `TransitionModel`
- [sampling.md](sampling.md) — sampling functions call `get_log_probs` on TransitionModel instances
- [models/AGENTS.md](models/AGENTS.md) — ESMC, ESM3 subclass `TransitionModelWithEmbedding`

## File Notes

- `from __future__ import annotations` at top — needed because `TransitionModel` references `LogitFormatter` which is defined later in the file

## TransitionModel

**Concrete class** (not ABC), uses composition pattern.

### Constructor

`__init__(model: nn.Module, tokenizer, logit_formatter: LogitFormatter)`

Wraps any model — no abstract methods to implement for basic use.

### Key methods

| Method | Behavior |
|--------|----------|
| `forward(seq_SP, **kwargs)` | Calls `self.model(seq_SP, **kwargs)`, returns raw output (may be non-tensor) |
| `format_raw_to_logits(raw, seq_SP, **kwargs)` | Applies `self.logit_formatter(raw, seq_SP)` — override when raw output isn't a tensor |
| `get_log_probs_from_string(sequences)` | Tokenizes then calls `get_log_probs` |

### Override pattern

When wrapping a model whose forward returns a dataclass (not a raw tensor), override `format_raw_to_logits`:
```python
def format_raw_to_logits(self, raw_output, seq_SP, **kwargs):
    logits = raw_output.sequence_logits.float()
    return self.logit_formatter(logits, seq_SP)
```

## TransitionModelWithEmbedding

**ABC** extending TransitionModel — adds a differentiable embedding path for TAG gradients and LinearProbe.

### Abstract methods (MUST implement)

| Method | Signature | Purpose |
|--------|-----------|---------|
| `differentiable_embedding` | `(ohe_seq_SPT: FloatTensor) -> FloatTensor` | OHE → deep embeddings (through transformer body) |
| `embedding_to_outputs` | `(embedding_SPD: FloatTensor) -> Any` | Deep embeddings → raw model output (same type as forward) |

### Concrete methods (inherited, usually not overridden)

| Method | Behavior |
|--------|----------|
| `embed(seq_SP)` | Token IDs → OHE → `differentiable_embedding()` → deep embeddings |
| `forward(seq_SP)` | `embed()` → `embedding_to_outputs()` |
| `format_raw_to_logits` | Applies logit_formatter (inherited from TransitionModel) |

### Subclass requirements

- Set `EMB_DIM: int` — embedding dimensionality (e.g. 960 for ESMC, 1536 for ESM3)
- The `embed()` output must be `(S, P, EMB_DIM)`

### OHE Padding

`embed()` creates OHE at `tokenizer.vocab_size` (e.g. 33 for ESM). If the model's actual embedding table is larger (e.g. 64 for ESM's alignment padding), `differentiable_embedding` handles this internally by doing `ohe @ embed.weight` where `embed` has the wider dim. Gradients flow back through the matmul to the vocab-sized OHE — TAG only sees vocab-sized gradients.

## LogitFormatter

`@runtime_checkable Protocol` — defines `__call__(logits, input_ids) -> FloatTensor`.

### MaskedModelLogitFormatter

Inherits `(nn.Module, LogitFormatter)` — **nn.Module must come first** in MRO.

Key design decisions:
- **Direct indexing** (`mask_matrix[token_ids]`) NOT one-hot matmul — `0.0 * (-inf) = NaN`
- **Additive masking** (0.0 pass-through, -inf block) NOT multiplicative — multiplying logits by -inf gives wrong signs for negative logits
- Uses `register_buffer` for the mask matrix (device tracking, no gradients)
- Uses `nn.Module.__init__(self)` instead of `super().__init__()` because Protocol in MRO breaks cooperative super chain

### PassThroughLogitFormatter

Returns `logits.float()` unchanged. For models that don't need output masking.

### Gotchas

- `output_dim` can exceed `vocab_size` for alignment (e.g. ESM's 33→64). Extra columns are valid mask outputs.
- ESM tokenizer v3.0.3: `mask_token_id` is `None` — use `tokenizer.vocab["<mask>"]`
- Current `MaskedModelLogitFormatter` constructor is HF-tokenizer-specific (uses `.vocab`, `.added_tokens_decoder`) — TODO for general constructor
- **Special token IDs for noising**: when masking positions for training or guidance, use `tokenizer.all_special_ids` to determine which positions are non-maskable (CLS, EOS, PAD, etc.) rather than hardcoding offsets like `+1` for CLS. This keeps noising logic tokenizer-agnostic — e.g. `~torch.isin(input_ids, torch.tensor(tokenizer.all_special_ids))` gives a boolean mask of maskable positions.

## MPNNTokenizer

Wraps ProteinMPNN's amino acid vocabulary. HF-compatible interface.

- Default: 20 standard AAs + UNK(X), indexed 0–20
- `include_mask_token=True` appends `<mask>` at idx 21 (for guidance setups needing explicit mask)
- `.vocab` returns `dict[str, int]` — same interface as HF tokenizers, used by `pca_embed_init` for cross-tokenizer mapping
- No `cls_token_id`, `eos_token_id`, or `pad_token_id` (all `None`)

## LoRA

LoRA adapter support lives on `TransitionModel`:

- `apply_lora(target_modules, r, lora_alpha, lora_dropout, bias, **kwargs)` — constructs `peft.LoraConfig` internally
- `target_modules=None` auto-discovers all `nn.Linear` modules
- `lora_target_modules()` — discovery helper, collapses block indices to `*` via regex
- After `apply_lora`, `self.model` is a `PeftModel` — attribute access delegates through PEFT
- `save_lora(path)` / `load_lora(path)` for adapter persistence
- `has_lora` property checks `isinstance(self.model, PeftModel)`

### LoRA + Checkpointing

- `TransitionModel.save()` writes `lora_adapter/` if LoRA present
- `TransitionModel.from_checkpoint()` loads LoRA adapter if `lora_adapter/` exists

## Maintenance

If changes are made to `TransitionModel` or `TransitionModelWithEmbedding` interfaces (abstract methods, LoRA API, LogitFormatter protocol), update the `add-generative-model` skill (`.agents/skills/add-generative-model/SKILL.md`) to reflect the new contract.

### LoRA Gotchas

- **lr=1e-3 causes mode collapse** with large models (ESM3 1.4B). Use lr=1e-4 or separate param groups for LoRA vs head.
- **Lazy module loading** — if model loads submodules lazily (e.g. ESM3 VQ-VAE), params loaded after `apply_lora()` won't be frozen. Re-freeze all base params, then re-enable `lora_` params.
- **`LinearProbe.__init__` has `freeze_embed_model`** — set to `False` when using LoRA so PEFT's freeze/unfreeze state is preserved.
- **`differentiable_embedding` works with PEFT** — `self.model.embed.weight` delegates to base model, transformer layers are LoRA-adapted automatically.
