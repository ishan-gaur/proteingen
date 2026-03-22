# generative_modeling

This module contains the `TransitionModel` class hierarchy, `LogitFormatter` protocol, `MPNNTokenizer`, and LoRA support. These are the building blocks for wrapping any generative model (masked LMs, flow-matching models, ProteinMPNN) into ProteinGen's interface.

## TransitionModel

A **concrete** `ProbabilityModel` subclass that wraps any `nn.Module` via composition:

```python
TransitionModel(model: nn.Module, tokenizer, logit_formatter: LogitFormatter)
```

- `forward(seq_SP, **kwargs)` delegates to `self.model(seq_SP, **kwargs)`
- `format_raw_to_logits` applies `self.logit_formatter(raw, seq_SP)`
- `get_log_probs_from_string(sequences)` tokenizes input strings, then calls `get_log_probs`

### Overriding for non-tensor outputs

When the wrapped model returns a dataclass (not a raw tensor), override `format_raw_to_logits`:

```python
def format_raw_to_logits(self, raw_output, seq_SP, **kwargs):
    logits = raw_output.sequence_logits.float()
    return self.logit_formatter(logits, seq_SP)
```

## TransitionModelWithEmbedding

An **ABC** extending `TransitionModel` for models that expose a differentiable embedding path. This is needed for two features:

1. **TAG gradients** — backpropagation from predictive model through generative model embeddings
2. **LinearProbe** — extracting and caching deep embeddings for probe training

### Abstract methods

| Method | Signature | Purpose |
|--------|-----------|---------|
| `differentiable_embedding` | `(ohe_seq_SPT: FloatTensor) → FloatTensor` | OHE → deep embeddings (through embedding layer + transformer body) |
| `embedding_to_outputs` | `(embedding_SPD: FloatTensor) → Any` | Deep embeddings → raw model output (same type as `forward`) |

### Subclass requirements

- Set `EMB_DIM: int` — embedding dimensionality (e.g. 960 for ESMC, 1536 for ESM3)
- The output of `embed()` must be shape `(S, P, EMB_DIM)`

### How OHE flows through

`embed()` creates a one-hot encoding at `tokenizer.vocab_size` (e.g. 33 for ESM). If the model's actual embedding table is wider (e.g. 64 for ESM's alignment padding), `differentiable_embedding` handles the mismatch internally via `ohe @ embed.weight`. Gradients flow back through the matmul to the vocab-sized OHE — TAG only sees vocab-sized gradients.

## LogitFormatter

A `@runtime_checkable Protocol` defining `__call__(logits, input_ids) → FloatTensor`.

### MaskedModelLogitFormatter

The standard formatter for masked language models. Key design decisions:

- **Direct indexing** (`mask_matrix[token_ids]`) instead of one-hot matmul — because `0.0 × (-inf) = NaN` (IEEE float)
- **Additive masking** (`0.0` pass-through, `-inf` block) instead of multiplicative — multiplying logits by `-inf` gives wrong signs for negative logits
- Uses `register_buffer` for the mask matrix (automatic device tracking, no gradients)

!!! note "`output_dim` vs `vocab_size`"
    `output_dim` can exceed `vocab_size` for alignment. For example, ESM models output 64 logits but only 33 are real vocabulary tokens. The extra columns are valid mask output positions.

### PassThroughLogitFormatter

Returns `logits.float()` unchanged. For models that don't need output masking.

## MPNNTokenizer

Wraps ProteinMPNN's amino acid vocabulary with an HF-compatible interface.

- Default: 20 standard AAs + UNK(X), indexed 0–20
- `include_mask_token=True` appends `<mask>` at idx 21 (needed when TAG guidance requires an explicit predictor-side mask token)
- `.vocab` returns `dict[str, int]` — same interface as HF tokenizers, used by `pca_embed_init` for cross-tokenizer vocabulary mapping
- No `cls_token_id`, `eos_token_id`, or `pad_token_id` (all `None`)

## LoRA support

LoRA adapter support lives on `TransitionModel`:

```python
model.apply_lora(target_modules=None, r=8, lora_alpha=16)
model.save_lora("adapters/my_adapter")
model.load_lora("adapters/my_adapter")
```

- `target_modules=None` auto-discovers all `nn.Linear` modules via `lora_target_modules()`
- After `apply_lora`, `self.model` becomes a `PeftModel` — attribute access delegates through PEFT
- `has_lora` property checks `isinstance(self.model, PeftModel)`
- Checkpointing is automatic: `save()` writes `lora_adapter/` if present, `from_checkpoint()` loads it

!!! warning "LoRA learning rate"
    LoRA-adapted large models (e.g. ESM3 1.4B) can collapse to constant predictions with lr=1e-3. Use lr=1e-4 or lower. Consider separate optimizer param groups for LoRA parameters vs. the prediction head.

!!! warning "Lazy module loading"
    If a model loads submodules lazily (e.g. ESM3's VQ-VAE encoder loaded on first `set_condition_()` call), those parameters won't be frozen by `apply_lora()`. After triggering lazy load, re-freeze all base params then re-enable `lora_` params.

---

## API Reference

::: proteingen.generative_modeling
