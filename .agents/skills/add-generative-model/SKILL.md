---
name: add-generative-model
description: Step-by-step workflow for integrating a new generative (transition) model into the proteingen library. Covers choosing between GenerativeModel and GenerativeModelWithEmbedding, implementing abstract methods, writing tests, and avoiding common gotchas. This skill is for generative models only — not PredictiveModel subclasses.
---

# Add a Generative Model

Workflow for wrapping a new pretrained generative model (e.g. a protein language model) into proteingen's `GenerativeModel` hierarchy. This skill covers `GenerativeModel` (composition) and `GenerativeModelWithEmbedding` (ABC with differentiable embeddings). It does **not** cover `PredictiveModel` subclasses — those are a separate concern.

**Progress tracking**: When returning control to the user, include a status line showing all phases and which one you're currently on. Example:

`Phases: 1 Research → 2 Internals → 3 Plan → 4 Scaffold → 5 Implement → **6 Tests** → 7 Design doc → 8 Docs site → 9 PR

This ensures the user always knows where you are and how much remains.

## Phase 1: Get Access & Align on Scope

Before writing any code:

1. **Clone or access the original model repo** — get the upstream source code so you can read the actual implementation (forward pass, tokenizer, embedding layers, etc.). If it's a HuggingFace model, pull the relevant modeling files. If it's a private repo, ask the user for access.
2. **Get an example script** — ask the user for a specific script or notebook they want to replicate using the proteingen abstractions. This is the acceptance criterion: the example should work end-to-end with the new wrapper.
3. **Clarify integration details** with the user before proceeding:
   - Which inputs are **conditioning variables** (e.g. structure, chain info) vs. sequence inputs?
   - How should **weights be stored and loaded**? (HuggingFace hub, local checkpoint path, bundled with the package?)
   - Are there **multiple model variants/sizes** to support?
   - Any **licensing or access restrictions** on weights?

## Phase 2: Understand the Model Internals

With access to the source code, gather:

1. **Model architecture** — what does its forward pass return? (raw logits tensor, a dataclass, multiple heads?)
2. **Tokenizer** — does it use a HuggingFace tokenizer? What's the vocab size? Are there special tokens (CLS, EOS, PAD, MASK)?
3. **Embedding access** — can you get the embedding layer's weight matrix? Is `ohe @ embed.weight` equivalent to the embedding lookup?
4. **Output dim** — does the output logit dim match vocab_size, or is it padded for alignment (e.g. ESM uses 64-dim output for 33-token vocab)?
5. **Conditioning** — does the model support structure conditioning or other side inputs? If so, how expensive is preprocessing?
6. **Loading** — how is the model loaded? (`from_pretrained`, custom checkpoint, etc.) Any device arg quirks?

## Phase 3: Present Plan & Get Approval

Before writing any implementation code, present a summary to the user:

- **Which base class** (`GenerativeModel` vs `GenerativeModelWithEmbedding`) and why
- **Directory/file layout** — where the model files will live, what each file contains
- **Interface design** — constructor signature, what conditioning looks like, any limitations
- **What the example script will look like** end-to-end

Get explicit sign-off before proceeding to implementation.

## Phase 4: Create the Directory Structure

```bash
mkdir -p src/proteingen/models/<provider>/
```

Directory names should reflect the **provider or model family** (e.g. `esm/` for
EvolutionaryScale ESM models, `mpnn/` for ProteinMPNN). Multiple models from
the same family live in the same directory.

Files to create:
- `__init__.py` — **re-exports only**, no implementation. Imports classes from
  sibling modules to keep the public API clean (e.g. `from .esmc import ESMC`).
- One `.py` file per model class (e.g. `esmc.py`, `esm3.py`). Name matches the
  class in lowercase.
- `utils.py` (optional) — shared helpers across models in the family (e.g.
  common input/output parsing, tokenizer setup, conditioning utilities).
- `<provider>.md` — design doc (dependencies, used-by, architecture, gotchas).

Example layout for ESM family:
```
models/esm/
├── __init__.py      # from .esmc import ESMC; from .esm3 import ESM3; ...
├── esmc.py          # ESMC class
├── esm3.py          # ESM3 class
├── esm3_api.py      # ESM3API class (Forge API variant)
├── utils.py         # shared tokenizer/formatting helpers
└── esm.md           # design doc
```

Update `src/proteingen/models/__init__.py` to export the new class (and any conditioning `TypedDict`s added in Phase 5).
Update `src/proteingen/models/AGENTS.md` to add the model to the registry.

## Phase 5: Implement

### Step 1: Choose between GenerativeModel and GenerativeModelWithEmbedding

Read the design docs to understand the two options:

- `src/proteingen/generative_modeling.md` — GenerativeModel and GenerativeModelWithEmbedding contracts
- `src/proteingen/probability_model.md` — ProbabilityModel base (conditioning, checkpointing) that both inherit from

```
Can you access the model's embedding layer weights,
and does the model need to support TAG guidance, LinearProbe,
or any workflow requiring gradients through the embedding step?

  YES → GenerativeModelWithEmbedding (ABC)
        You implement: differentiable_embedding, embedding_to_outputs
        You get for free: forward, embed, format_raw_to_logits

  NO  → GenerativeModel (concrete, composition)
        You pass in: model, tokenizer, logit_formatter
        You may override: format_raw_to_logits (if raw output isn't a tensor)
```

Most protein language models should use `GenerativeModelWithEmbedding` — TAG guidance and embedding-based probes are core use cases.

### TypedDicts for structured inputs/outputs

Define `TypedDict`s for:

- **Conditioning input** — if the model accepts conditioning variables (structure coords, chain IDs, etc.), define a `TypedDict` for the conditioning dict so the interface is self-documenting and type-checkable. E.g. `class ESM3Conditioning(TypedDict): structure_coords: Tensor; ...`. If possible try to match these names with the other models in the library or recommend changes to the other names as part of your PR if you think they should be improved.
- **Raw forward output** — if `forward` or `embedding_to_outputs` returns something richer than a plain tensor (e.g. a dataclass with multiple heads), define a `TypedDict` for the raw output so downstream code can access fields by name. This replaces opaque `Any` return types. Again, if possible, try to have the names of these fields be similar across models when the content is the same.

Place these in the model's `.py` file alongside the class, or in `utils.py` if shared
across models in the family. Export them from the directory's `__init__.py` (the `models/__init__.py` export was already set up in Phase 4).

### Step 2a: GenerativeModelWithEmbedding (most common path)

Use ESMC as the reference implementation: `src/proteingen/models/esm/esmc.py`.

#### Constructor checklist

- [ ] Call `super().__init__(model, tokenizer, logit_formatter)`
- [ ] `model` — the loaded pretrained model (nn.Module)
- [ ] `tokenizer` — HF tokenizer or `MPNNTokenizer`
- [ ] `logit_formatter` — typically `MaskedModelLogitFormatter(tokenizer, OUTPUT_DIM)`
- [ ] Set `OUTPUT_DIM` class variable (or use vocab_size if no padding)
- [ ] Set `EMB_DIM` (read from model weights if dynamic, e.g. `model.embed.weight.shape[1]`)
- [ ] Stash any checkpoint identifier for `_save_args()` (e.g. `self._checkpoint_name = checkpoint`)

#### Abstract methods to implement

**`differentiable_embedding(ohe_seq_SPT: FloatTensor) -> FloatTensor`**

Takes a one-hot (or soft) distribution over tokens and returns deep embeddings after the transformer body.

Typical pattern:
```python
def differentiable_embedding(self, ohe_seq_SPT):
    # Differentiable embedding lookup
    emb = ohe_seq_SPT @ self.model.embed.weight  # (S, P, D)
    # Run through transformer
    output = self.model.transformer(emb)
    return output  # (S, P, EMB_DIM)
```

**`embedding_to_outputs(embedding_SPD: FloatTensor) -> Any`**

Takes deep embeddings and returns raw model outputs (same type as `forward` would return).

Typical pattern:
```python
def embedding_to_outputs(self, embedding_SPD):
    return self.model.output_head(embedding_SPD)
```

#### Optional overrides

**`format_raw_to_logits`** — override if `embedding_to_outputs` returns a dataclass rather than a raw tensor:
```python
def format_raw_to_logits(self, raw_output, seq_SP, **kwargs):
    logits = raw_output.sequence_logits.float()  # extract tensor from dataclass
    return self.logit_formatter(logits, seq_SP)
```

**`preprocess_observations` / `collate_observations`** — override for conditioning (structure, etc.). See ESM3's structure conditioning as the reference pattern.

**`_save_args`** — implement for checkpointing support.

### Step 2b: GenerativeModel (composition)

For simpler cases where you just wrap an existing model without needing embedding access:

```python
from proteingen import GenerativeModel, MaskedModelLogitFormatter

model = load_my_model(checkpoint)
tokenizer = load_my_tokenizer()
formatter = MaskedModelLogitFormatter(tokenizer, output_dim=model.output_dim)
tm = GenerativeModel(model, tokenizer, formatter)
```

Override `format_raw_to_logits` if the model's forward returns something other than a logit tensor.

## Phase 6: Write Tests

Create `tests/test_<name>.py`. Required test categories:

### Construction & basic forward
- Model loads without error
- `forward(seq_SP)` returns expected output shape

### Logit matching against original library
- Run the same **real protein sequence** (not random tokens) through both the original library's model and the ProteinGen wrapper
- Assert logits match within floating-point tolerance (`torch.allclose` with appropriate `atol`/`rtol`)
- This is the single most important test — it proves the wrapper faithfully reproduces the original model's behavior

### Embedding path (GenerativeModelWithEmbedding only)
- `embed(seq_SP)` returns shape `(S, P, EMB_DIM)`
- Embedding path matches forward path: `embedding_to_outputs(embed(seq))` ≈ `forward(seq)` (within float tolerance)
- Gradients flow through `embed()` — `_ohe.grad` is not None after backward

### Log probabilities
- Temperature scaling works: higher temp → flatter distribution
- `get_log_probs` output shape is `(S, P, vocab_size)` or `(S, P, OUTPUT_DIM)`
- Log probs are valid: all ≤ 0, `exp(log_probs).sum(dim=-1)` ≈ 1

### Batching
- Single sequence and batched sequences produce consistent results (first sequence in batch matches solo inference)

### Conditioning (if applicable)
- `set_condition_()` caches observations correctly
- `conditioned_on()` context manager reverts state
- Conditioned forward produces different output than unconditioned
- `collate_observations` tiles correctly to batch size

### LoRA (if applicable)
- `apply_lora()` makes some params trainable
- Forward still works after LoRA
- `save_lora` / `load_lora` round-trips

### Checkpointing (if applicable)
- `save()` + `from_checkpoint()` round-trips
- Loaded model produces same outputs as original


## Common Gotchas Checklist

Review these before considering the integration complete:

- [ ] **`from_pretrained` device arg** — some models require `torch.device("cpu")`, not the string `"cpu"`. ESM models call `device.type` which fails on strings.
- [ ] **Import shadowing** — if your wrapper class has the same name as the upstream class, import as `from upstream import Model as _Model`
- [ ] **Output dim vs vocab_size** — if the model pads output dim for alignment, pass `output_dim` to `MaskedModelLogitFormatter`
- [ ] **`mask_token_id` may be None** — some tokenizers (ESM v3.0.3) don't set `mask_token_id`. Use `tokenizer.vocab["<mask>"]` instead.
- [ ] **Lazy module loading** — if the model loads submodules lazily (e.g. VQ-VAE encoders), params loaded after `apply_lora()` won't be frozen. Re-freeze after triggering lazy load.
- [ ] **`nn.Module.train()` recurses** — if you freeze a pretrained backbone, calling `.train()` on the wrapper will set the backbone to train mode (enabling dropout). Override `train()` to protect frozen modules.
- [ ] **bfloat16 / float32** — some models default to bfloat16 on GPU. If you need float32 (e.g. CPU inference), call `.float()` after loading.
- [ ] **Geometric attention / pairwise ops OOM** — models with O(L²) attention may OOM at large batch sizes. Document the safe batch_size for your GPU.
- [ ] **Structure conditioning length-lock** — if `preprocess_observations` encodes to fixed-length tokens, all subsequent calls must match that length.
- [ ] **`from __future__ import annotations`** — if your file has forward references in type annotations, add this import at the top.

## Phase 7: Write the Design Doc

Create `src/proteingen/models/<provider>/<provider>.md` following the pattern in `models/esm/esm.md`:

- **Dependencies** — what core abstractions and external packages it uses
- **Used By** — which examples, tests, and downstream components consume it
- **Architecture** — internals (what are the sub-modules, how does forward work)
- **Checkpointing** — what gets saved, any stashed args
- **Gotchas** — model-specific traps (device args, lazy loading, dtype issues, OOM thresholds)

## Phase 8: Add Documentation

⚠️ **The model is NOT done until this phase is complete.** Passing tests = Phase 6. Don't stop here.

After the model is implemented and tests pass, add user-facing documentation:

1. **Update `docs/models.md`** — add the model to the appropriate table (Generative Models or Predictive Models) and write a section covering:
   - What the model does and where it comes from (paper, repo link)
   - How to load and use it with proteingen
   - Code snippets for common workflows (forward pass, sampling, conditioning, guidance)
   - If the model supports conditioning, document the conditioning `TypedDict` — list each field, its type, and what it represents. Show a code example of constructing and passing conditioning.
   - Any model-specific configuration (variants, checkpoint sizes, device requirements)
2. **Update `mkdocs.yml`** — add any new pages to the nav if needed.
3. **Verify `models/AGENTS.md`** — confirm the model was added to the registry in Phase 4.
4. **Verify exports** — confirm `src/proteingen/models/__init__.py` exports the class and any conditioning `TypedDict`s.

## Phase 9: Open a Pull Request

Create a clean branch with only the model integration changes and open a PR against `main`.

If your working branch has accumulated unrelated changes, isolate the model files onto a fresh branch:

```bash
git checkout main && git pull
git checkout -b pr/<model-name>
git checkout <feature-branch> -- \
  src/proteingen/models/<provider>/ \
  tests/test_<name>.py \
  docs/models.md \
  src/proteingen/models/__init__.py \
  src/proteingen/models/AGENTS.md
# Review staged files, then commit
git diff --cached --stat
git add -A && git commit -m "feat(models): add <ModelName> generative model wrapper"
git push -u origin pr/<model-name>
```

Before opening the PR, walk through every item in the checklist below and actually verify it — run the tests, check the exports, read the docs page. Don't just check the boxes from memory.

### Add direct code links in the PR body

Include a short **Code Links** section in the PR body that points reviewers to the key files and tests:

- wrapper class implementation
- conditioning `TypedDict`
- export wiring (`models/__init__.py`)
- logit-matching test
- docs page for the new model

Use **stable commit permalinks** (`.../blob/<commit-sha>/path#Lx-Ly`) rather than branch links so references don't break after rebases/force-pushes. In GitHub file view, press `y` to convert to a commit permalink.

Use the following template for the PR description — copy it directly into the GitHub PR body and fill in each section:

```markdown
## Summary
<!-- What model is being added? Link to paper/repo. What capabilities does this enable for proteingen users? -->

## Implementation
- **Base class**: <!-- GenerativeModel / GenerativeModelWithEmbedding — why? -->
- **Conditioning**: <!-- What observation variables are supported? Link to the conditioning TypedDict. -->
- **New dependencies**: <!-- Packages added to pyproject.toml, with versions -->

## Tests
<!-- Describe each test category and what it covers. Must include a logit-matching
     test that runs the same real protein sequence through both the original library
     and the ProteinGen wrapper, asserting outputs match within floating-point tolerance. -->

## Checklist
- [ ] Model class in `src/proteingen/models/<provider>/`
- [ ] Exported from `models/__init__.py` (class and conditioning `TypedDict` if applicable)
- [ ] Tests pass (`uv run python -m pytest tests/test_<name>.py -v`)
- [ ] Includes logit-matching test against original library on a real protein input
- [ ] Design doc at `src/proteingen/models/<provider>/<provider>.md`
- [ ] Listed in `docs/models.md` with code examples
- [ ] Conditioning `TypedDict` documented in `docs/models.md` (if applicable)
- [ ] `models/AGENTS.md` updated
- [ ] No breaking changes to existing APIs (or noted below)

## Known Limitations
<!-- GPU memory requirements, sequence length limits, missing features -->

## Example
<!-- Paste or link to the end-to-end example script from Phase 1 -->
```


