# Adding a Generative Model

Checklist for integrating a new generative (transition) model into ProteinGen. For the full step-by-step agent skill, see [`.agents/skills/add-generative-model/SKILL.md`](../.agents/skills/add-generative-model/SKILL.md).

## Choose a Base Class

- **`TransitionModel`** — wraps any `nn.Module` via composition. Use when you don't need gradient access to the embedding layer.
- **`TransitionModelWithEmbedding`** — ABC extending `TransitionModel` with a differentiable embedding path. Required for TAG guidance, `LinearProbe`, and any workflow needing gradients through the embedding step. **Most protein language models should use this.**

## Directory Layout

```
src/proteingen/models/<provider>/
├── __init__.py        # re-exports only
├── <model>.py         # one file per model class
├── utils.py           # optional shared helpers
└── <provider>.md      # design doc (dependencies, used-by, architecture, gotchas)
```

## Implementation Checklist

### TransitionModelWithEmbedding (most common)

Reference: `src/proteingen/models/esm/esmc.py`

- [ ] `__init__`: call `super().__init__(model, tokenizer, logit_formatter)`, set `EMB_DIM` and `OUTPUT_DIM`
- [ ] `differentiable_embedding(ohe_seq_SPT)` — OHE → deep embeddings via `ohe @ embed.weight` + transformer body
- [ ] `embedding_to_outputs(embedding_SPD)` — deep embeddings → raw model output (same type as `forward`)
- [ ] Override `format_raw_to_logits` if model returns a dataclass instead of a raw logit tensor
- [ ] Override `preprocess_observations` / `collate_observations` if model accepts conditioning (e.g. structure)
- [ ] Implement `_save_args()` for checkpointing support

### TransitionModel (composition)

- [ ] Pass `model`, `tokenizer`, `logit_formatter` to constructor
- [ ] Override `format_raw_to_logits` if the model's forward returns non-tensor output

### TypedDicts

- [ ] Define `TypedDict` for conditioning input if the model accepts conditioning variables
- [ ] Define `TypedDict` for raw output if the model returns structured output beyond raw logits
- [ ] Match field names with existing models where the content is the same

## Required Tests

Create `tests/test_<name>.py`:

- [ ] **Construction & forward** — model loads, `forward()` returns expected shape
- [ ] **Output matching** — same real protein input through original library AND ProteinGen wrapper, `torch.allclose` on outputs ← most important test
- [ ] **Embedding path** (TransitionModelWithEmbedding only) — `embed()` shape, `embedding_to_outputs(embed(seq))` ≈ `forward(seq)`, gradients flow
- [ ] **Log probabilities** — valid (all ≤ 0, sum to ~1 after exp), temperature scaling works
- [ ] **Batching** — single vs batched results are consistent
- [ ] **Conditioning** — if applicable: `set_condition_()`, `conditioned_on()`, `collate_observations`
- [ ] **LoRA** — if applicable: `apply_lora()`, `save_lora()` / `load_lora()` round-trip
- [ ] **Checkpointing** — if applicable: `save()` + `from_checkpoint()` round-trip

## Documentation

- [ ] Design doc at `src/proteingen/models/<provider>/<provider>.md`
- [ ] Add to `src/proteingen/models/AGENTS.md` registry
- [ ] Add to `docs/models.md` with code examples and conditioning docs
- [ ] Export from `src/proteingen/models/__init__.py`
- [ ] Update `mkdocs.yml` if new pages needed

## Common Gotchas

- `from_pretrained` device arg — some models require `torch.device("cpu")`, not string `"cpu"`
- Output dim vs vocab_size — pass `output_dim` to `MaskedModelLogitFormatter` if padded
- Lazy module loading — re-freeze after triggering lazy loads (e.g. VQ-VAE) when using LoRA
- `nn.Module.train()` recurses — override `train()` to protect frozen modules

## Using an Agent

Prompt your agent with:

> "Read the skill file at `.agents/skills/add-generative-model/SKILL.md` and follow it to add **[model name]** to ProteinGen."
