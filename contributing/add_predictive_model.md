# Adding a Predictive Model

Checklist for integrating a new predictive model into ProtStar. For the full step-by-step agent skill, see [`.agents/skills/add-predictive-model/SKILL.md`](../.agents/skills/add-predictive-model/SKILL.md).

## The Four Layers

Every predictive model integration decomposes into four separable layers. Identify which already exist in the library and only build what's missing.

1. **Raw Predictor** — the original pretrained model, ported with minimal changes. Not protstar-specific.
2. **Binary Logit Function** — converts raw output to `(B, 2)` binary logits. Independent of the model.
3. **Template Model Class** *(optional)* — a reusable architecture pattern (e.g. `LinearProbe`, `OneHotMLP`, `EmbeddingMLP`). Skip if the architecture is one-off.
4. **PredictiveModel Subclass** — thin glue wiring layers 1–3 together with conditioning, tokenizer, and OHE basis.

## Available Binary Logit Functions

| Function | Input | TAG compatible? |
|----------|-------|-----------------|
| `binary_logits(logit_B, target)` | Single logit | ✅ |
| `categorical_binary_logits(logits_BC, target_class)` | Multi-class logits | ✅ |
| `point_estimate_binary_logits(pred_B, threshold, k)` | Scalar prediction | ⚠️ Large k saturates gradients |
| `gaussian_binary_logits(mu_B, log_var_B, threshold)` | Gaussian params | ✅ |

If none fit, write a new function in `src/protstar/predictive_modeling.py`.

## Available Template Classes

| Template | Architecture | Use case |
|----------|-------------|----------|
| `LinearProbe` | Frozen `GenerativeModelWithEmbedding` + linear head | Probe on pretrained embeddings |
| `OneHotMLP` | Flattened OHE → MLP | Simple baseline |
| `EmbeddingMLP` | Differentiable embedding → MLP | PCA-init from pretrained models |
| `PairwiseLinearModel` | Pairwise OHE features → linear | Epistatic models |

## Directory Layout

```
src/protstar/models/<provider>/
├── __init__.py        # re-exports only
├── <model>.py         # raw predictor + PredictiveModel subclass
├── utils.py           # optional helpers (data loading, featurization)
└── <provider>.md      # design doc
```

## Implementation Checklist

Reference: `src/protstar/models/rocklin_ddg/stability_predictor.py`

### Layer 1: Raw Predictor
- [ ] Port original model code with minimal changes
- [ ] If TAG is needed, make embedding lookup differentiable: replace `self.embed(token_ids)` with `ohe @ self.embed.weight`
- [ ] Preserve encode/decode split for conditioning

### Layer 2: Binary Logit Function
- [ ] Select existing function, or write a new one if needed
- [ ] New functions: add to `predictive_modeling.py`, export from `protstar.__init__`

### Layer 3: Template Class (optional)
- [ ] Use existing template if architecture matches
- [ ] New templates: add to `predictive_modeling.py`, export from `protstar.__init__`

### Layer 4: PredictiveModel Subclass
- [ ] `__init__`: call `super().__init__(tokenizer=tokenizer)`, set default target
- [ ] `forward(ohe_seq_SPK, **kwargs)` — takes OHE features (not token IDs), delegates to raw predictor
- [ ] `format_raw_to_logits(raw_output, seq_SPK, **kwargs)` — returns `(B, 2)` using binary logit function
- [ ] Override `token_ohe_basis()` if tokenizer needs reduced feature space (register as buffer)
- [ ] Override `preprocess_observations` (expensive, runs once) / `collate_observations` (cheap, runs each forward)
- [ ] Define conditioning `TypedDict` if applicable

## Required Tests

Create `tests/test_<name>.py`:

- [ ] **Construction & forward** — model loads, `forward(ohe, **conditioning)` returns expected shape
- [ ] **Output matching** — same real input through original library AND ProtStar wrapper, `torch.allclose` ← most important test
- [ ] **Binary logit pattern** — `format_raw_to_logits` returns `(B, 2)`, target switching works
- [ ] **Gradient flow** — `grad_log_prob(seq_SP)` returns `(B, P, K)` with non-zero gradients (TAG) or document DEG-only
- [ ] **Conditioning** — `set_condition_()`, `conditioned_on()`, `collate_observations` tiling
- [ ] **Batching** — single vs batched results consistent
- [ ] **Token OHE basis** — if overridden: correct shape, special tokens map correctly
- [ ] **Checkpointing** — if applicable: `save()` + `from_checkpoint()` round-trip

## Documentation

- [ ] Design doc at `src/protstar/models/<provider>/<provider>.md`
- [ ] Add to `src/protstar/models/AGENTS.md` registry
- [ ] Add to `docs/models.md` with code examples, conditioning docs, TAG/DEG compatibility
- [ ] Export from `src/protstar/models/__init__.py`
- [ ] New binary logit functions / templates documented in `docs/reference/predictive_modeling.md`

## Common Gotchas

- `forward` takes OHE, not token IDs — `PredictiveModel.get_log_probs` converts tokens to OHE before calling `forward`
- `format_raw_to_logits` must return `(B, 2)` — asserted in `get_log_probs`
- Target must be set before `get_log_probs` — set a sensible default in `__init__`
- Steep sigmoid kills TAG gradients — use small k (5–10) for `point_estimate_binary_logits`, or use DEG
- `preprocess_observations` should use `torch.no_grad()` — structure encoding is inference-only
- `collate_observations` must use `expand`, not `repeat` — `expand` shares memory

## Using an Agent

Prompt your agent with:

> "Read the skill file at `.agents/skills/add-predictive-model/SKILL.md` and follow it to add **[model name]** to ProtStar."
