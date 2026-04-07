---
name: add-predictive-model
description: Step-by-step workflow for integrating a new predictive model into the proteingen library. Covers decomposing a pretrained predictor into four layers (raw model, binary logit function, template model class, PredictiveModel subclass), identifying what already exists vs what's missing, and only building the missing pieces. This skill is for predictive models only — not GenerativeModel subclasses.
---

# Add a Predictive Model

Workflow for integrating a new pretrained predictive model (e.g. a stability predictor, fitness oracle) into proteingen's `PredictiveModel` hierarchy. Predictive models answer "what is log p(target | sequence)?" and are used by TAG/DEG guidance to steer generative sampling.

**Progress tracking**: When returning control to the user, include a status line showing all phases and which one you're currently on. Example:

`Phases: 1 Research → 2 Internals → 3 Plan → 4 Scaffold → 5 Implement → **6 Tests** → 7 Design doc → 8 Docs site → 9 PR`

This ensures the user always knows where you are and how much remains.

## The Four Layers

Integrating a predictive model means decomposing it into four separable layers. The planning phase (Phase 3) is about figuring out which of these already exist in the library and which need to be built.

### Layer 1: Raw Predictor

The original pretrained model — its architecture, weights, and forward pass. This is **not proteingen-specific**. You port it with minimal changes, just enough to load and run inference. Examples: `StabilityPMPNN`, a pretrained CNN fitness predictor, a GNN binding model.

### Layer 2: Binary Logit Function

A standalone function that converts the raw predictor's output to `(B, 2)` binary logits `[false_logit, true_logit]`. This is **independent of the model** — the same predictor could use different functions depending on the use case.

Existing functions in `proteingen.predictive_modeling`:

| Function | Input → Output | Use case | TAG compatible? |
|----------|---------------|----------|-----------------|
| `binary_logits(logit_B, target)` | scalar → `[0, logit]` | Binary classification | ✅ Yes |
| `categorical_binary_logits(logits_BC, target_class)` | multi-class logits → `[rest, target]` | Classification | ✅ Yes |
| `point_estimate_binary_logits(pred_B, threshold, k)` | scalar → sigmoid thresholding | Regression | ⚠️ Large k saturates gradients |
| `gaussian_binary_logits(mu_B, log_var_B, threshold)` | mean + variance → CDF | Probabilistic regression | ✅ Yes |

If none of these fit, you'll need to write a new one. It's just a function `(raw_output, ...) -> FloatTensor(B, 2)`.

### Layer 3: Template Model Class (optional)

A reusable `PredictiveModel` subclass that defines an architecture pattern but leaves `format_raw_to_logits` abstract. The user instantiates it with configuration, not pretrained weights. Examples: `LinearProbe`, `OneHotMLP`, `EmbeddingMLP`, `PairwiseLinearModel`.

Most pretrained predictors don't need a new template — they have their own architecture. But if the predictor's pattern is generalizable (e.g. "frozen GNN encoder + linear head"), consider whether a new template would benefit future models.

### Layer 4: PredictiveModel Subclass

The final glue class that wires layers 1–3 together. If layers 1–2 are well-designed (and a suitable template exists for layer 3), this class should be thin — just constructor, `forward`, and `format_raw_to_logits`.

**The goal**: the `PredictiveModel` subclass should be mostly boilerplate. If it's getting complex, something in layers 1–3 is missing.

## Phase 1: Get Access & Align on Scope

Before writing any code:

1. **Clone or access the original model repo** — get the upstream source code and pretrained weights.
2. **Get an example script** — ask the user for a specific script they want to replicate using the proteingen abstractions. This is the acceptance criterion.
3. **Clarify integration details** with the user:
   - What does the model predict? (stability, fitness, binding, classification, regression)
   - What **conditioning** does the model need? (structure coords, chain info, ligand, etc.)
   - Is there an expensive preprocessing step that should run once per structure?
   - How should **weights be stored and loaded**?
   - Does the model need to work with **TAG** (gradient-based guidance)? If so, `forward` must be differentiable w.r.t. OHE input.

## Phase 2: Understand the Model Internals

With access to the source code, gather:

1. **Architecture** — what does the forward pass look like? What are the inputs (sequence tokens, coordinates, features)?
2. **Output format** — single scalar? Multi-class logits? Mean + variance? This determines which binary logit function to use.
3. **Tokenizer** — what amino acid encoding does it use? How many tokens? Are there special tokens?
4. **Encode/decode split** — does the model have an expensive structure encoding step that can be separated from the cheap sequence-dependent decoding? This maps to `preprocess_observations` / `forward`.
5. **Differentiability** — can you replace the token embedding lookup with a differentiable `ohe @ embed.weight` matmul? This is required for TAG.
6. **Loading** — how is the model loaded? Any device or dtype quirks?

## Phase 3: Decompose & Plan

This is the key planning phase. Present the decomposition to the user, identifying what exists vs what's missing:

### Layer 1: Raw Predictor
- How much porting is needed? Can you use the original code directly, or does it need adaptation?
- What's the encode/decode split for conditioning?

### Layer 2: Binary Logit Function
- Which existing function fits the model's output format?
- If none fit, what does the new function need to do? (Describe the math — it should be a simple, standalone function.)
- Is the chosen function TAG-compatible? If not, the model is DEG-only.

### Layer 3: Template Model Class
- Does the predictor's architecture match an existing template (`LinearProbe`, `OneHotMLP`, `EmbeddingMLP`, `PairwiseLinearModel`)?
- If not, is the pattern generalizable enough to warrant a new template? (e.g. "frozen structure encoder + sequence decoder + pooling + head" could be a `StructureConditionedPredictor` template)
- If the architecture is one-off, skip the template — subclass `PredictiveModel` directly.

### Layer 4: PredictiveModel Subclass
- What goes in the constructor? (tokenizer, model loading, default target)
- What's the OHE basis? (default identity, or reduced space with `<mask>` → all-zero?)
- What conditioning TypedDicts are needed?

### Summary for user approval
- Which layers are new vs existing
- TAG vs DEG compatibility
- Directory/file layout
- What the example script will look like end-to-end

Get explicit sign-off before proceeding to implementation.

## Phase 4: Create the Directory Structure

```bash
mkdir -p src/proteingen/models/<provider>/
```

Files to create:
- `__init__.py` — re-exports only
- One `.py` file per model class (raw predictor + PredictiveModel subclass can live in the same file if the raw predictor isn't reused elsewhere)
- `utils.py` (optional) — shared helpers (data loading, featurization)
- `<provider>.md` — design doc

Example layout (stability predictor):
```
models/rocklin_ddg/
├── __init__.py              # from .stability_predictor import PreTrainedStabilityPredictor, ...
├── stability_predictor.py   # StabilityPMPNN (raw) + PreTrainedStabilityPredictor (wrapper)
├── data_utils.py            # PDB loading, featurization
└── rocklin_ddg.md           # design doc
```

If adding a new binary logit function, it goes in `src/proteingen/predictive_modeling.py` alongside the existing ones. If adding a new template model class, it also goes in `predictive_modeling.py`.

Update `src/proteingen/models/__init__.py` to export the new class and any conditioning/output `TypedDict`s.
Update `src/proteingen/models/AGENTS.md` to add the model to the registry.

## Phase 5: Implement

Read the design docs:

- `src/proteingen/predictive_modeling.md` — PredictiveModel ABC, binary logit functions, template models
- `src/proteingen/probability_model.md` — ProbabilityModel base (conditioning, checkpointing)
- `src/proteingen/models/rocklin_ddg/stability_predictor.py` — reference implementation

### Step 1: Port the raw predictor (Layer 1)

Bring the original model code into the provider directory with minimal changes. The goal is a working `nn.Module` that can load pretrained weights and run inference.

- Keep the original architecture intact — don't refactor it to match proteingen patterns
- If the model has an encode/decode split, preserve it (you'll map this to preprocess/forward later)
- If TAG is needed, make the embedding lookup differentiable: replace `self.embed(token_ids)` with accepting OHE input and doing `ohe @ self.embed.weight`

### Step 2: Add or select the binary logit function (Layer 2)

If an existing function works, skip this step.

If you need a new function, add it to `src/proteingen/predictive_modeling.py`:

```python
def my_binary_logits(raw_output, ...) -> torch.FloatTensor:
    """Convert raw output to (B, 2) binary logits [false_logit, true_logit]."""
    ...
```

Export it from `proteingen.__init__` and add it to the table in `src/proteingen/predictive_modeling.md`.

### Step 3: Add or select the template class (Layer 3, optional)

If the predictor's pattern is one-off, skip this step and subclass `PredictiveModel` directly.

If you're adding a new template, add it to `src/proteingen/predictive_modeling.py`. A template is an ABC that:
- Subclasses `PredictiveModel`
- Defines the architecture (forward pass, pooling, head)
- Leaves `format_raw_to_logits` abstract (the user picks the binary logit function)

### Step 4: Wire it together (Layer 4)

The `PredictiveModel` subclass. If layers 1–3 are well-designed, this should be thin.

#### Constructor

```python
class MyPredictor(PredictiveModel):
    def __init__(self, ckpt_path, device="cuda"):
        tokenizer = ...  # set up tokenizer
        super().__init__(tokenizer=tokenizer)
        self.target = True  # default target (user can change with set_target_)
        self.model = ...    # load the raw predictor (Layer 1)
```

- Call `super().__init__(tokenizer=tokenizer)` — sets `self.tokenizer`, `self.target = None`, `self._ohe = None`
- Set a sensible default target (e.g. `True` for "is stable")

#### `forward(ohe_seq_SPK, **kwargs) -> Any`

Takes OHE features (not token IDs) and conditioning kwargs. Delegates to the raw predictor.

- Input is `ohe_seq_SPK` where K is the OHE feature dim (= vocab_size by default, or smaller if `token_ohe_basis` is overridden)
- `**kwargs` receives the output of `collate_observations`

#### `format_raw_to_logits(raw_output, seq_SPK, **kwargs) -> FloatTensor(B, 2)`

Calls the binary logit function from Layer 2:

```python
def format_raw_to_logits(self, raw_output, seq_SPK, **kwargs):
    return binary_logits(raw_output, self.target)
```

#### TypedDicts

Define `TypedDict`s for conditioning inputs and intermediate representations:

```python
class MyConditioningInput(TypedDict):
    coords: torch.Tensor       # (1, L, 4, 3) backbone atom coordinates
    mask: torch.Tensor         # (1, L) residue mask

class MyStructureEncoding(TypedDict):
    node_features: torch.Tensor   # (1, L, D) encoded node features
    edge_features: torch.Tensor   # (1, L, K, D) encoded edge features
```

Place these in the model's `.py` file. Export from the directory's `__init__.py` (the `models/__init__.py` export was already set up in Phase 4).

Try to match field names with existing models in the library. Recommend changes to existing names as part of your PR if you think they should be improved.

#### Conditioning (preprocess/collate split)

**`preprocess_observations(observations) -> processed`** — runs once per structure. Delegates to the raw predictor's expensive encoding step. Wrap in `torch.no_grad()`.

**`collate_observations(x_B, observations) -> dict`** — runs every forward pass. Tiles cached representation to batch size using `expand` (not `repeat`). Returns a dict whose keys become `**kwargs` to `forward`.

#### Token OHE basis

Override `token_ohe_basis()` if the tokenizer has tokens that should map to a reduced feature space:

```python
def token_ohe_basis(self) -> torch.FloatTensor:
    return self._token_ohe_basis_TK  # registered as buffer in __init__
```

Register the basis as a buffer (`register_buffer`) so it tracks device automatically.

#### Static helper for conditioning setup

Consider adding a static method that builds conditioning from raw inputs:

```python
@staticmethod
def prepare_conditioning(pdb_path, device="cpu") -> MyConditioningInput:
    """Load PDB and build conditioning tensors."""
    ...
```

Not required by the ABC but makes the model easier to use.

## Phase 6: Write Tests

Create `tests/test_<name>.py`. Required test categories:

### Construction & basic forward
- Model loads without error
- `forward(ohe_SPK, **conditioning)` returns expected output shape
- `predict(seq_SP)` returns raw output (no binary conversion)

### Output matching against original library
- Run the same **real protein input** (not random data) through both the original library's model and the ProteinGen wrapper
- Assert outputs match within floating-point tolerance (`torch.allclose` with appropriate `atol`/`rtol`)
- This is the single most important test — it proves the wrapper faithfully reproduces the original model's behavior

### Binary logit pattern
- `format_raw_to_logits` returns shape `(B, 2)`
- `get_log_probs` returns shape `(B,)` with values ≤ 0
- Target switching works: `set_target_(True)` vs `set_target_(False)` produce different log probs that sum correctly

### Gradient flow (TAG compatibility)
- `grad_log_prob(seq_SP)` returns shape `(B, P, K)` with non-zero gradients
- If the model is NOT TAG-compatible (non-differentiable forward), document this and test that DEG works instead

### Conditioning
- `set_condition_()` caches observations correctly
- `conditioned_on()` context manager reverts state
- `preprocess_observations` runs the expensive encoding once
- `collate_observations` tiles correctly to batch size

### Batching
- Single sequence and batched sequences produce consistent results

### Token OHE basis (if overridden)
- Basis shape is `(vocab_size, K)`
- Special tokens (e.g. `<mask>`) map to expected vectors (e.g. all-zero)

### Binary logit function (if new)
- Returns correct shape `(B, 2)` for various inputs
- Gradients flow through the function (if TAG-compatible)
- Edge cases (zero input, large values, boundary thresholds)

### Template model class (if new)
- Construction with mock tokenizer
- Forward pass shape
- `get_log_probs` pipeline works end-to-end
- `grad_log_prob` returns non-zero gradients

### Checkpointing (if applicable)
- `save()` + `from_checkpoint()` round-trips
- Loaded model produces same outputs as original


## Common Gotchas Checklist

Review these before considering the integration complete:

- [ ] **`forward` takes OHE, not token IDs** — `PredictiveModel.get_log_probs` converts tokens to OHE before calling `forward`. If you're testing `forward` directly, pass OHE features.
- [ ] **`format_raw_to_logits` must return `(B, 2)`** — asserted in `get_log_probs`. If your model returns a scalar, use `binary_logits(scalar, self.target)` which constructs `[0, logit]`.
- [ ] **Target must be set** before calling `get_log_probs` — asserts `self.target is not None`. Set a sensible default in `__init__`.
- [ ] **Steep sigmoid kills TAG gradients** — `point_estimate_binary_logits` with large `k` saturates the sigmoid, making gradients ≈ 0. Use k=5–10, or use DEG instead of TAG.
- [ ] **Tokenizer mismatch with generative model** — the predictor's tokenizer may differ from the generative model's. TAG's `GuidanceProjection` handles this mapping, but the OHE basis dimensions must be correct.
- [ ] **`token_ohe_basis` device** — if you override `token_ohe_basis()`, use `register_buffer` for the basis tensor so it automatically tracks the model's device.
- [ ] **`preprocess_observations` should use `torch.no_grad()`** — structure encoding is inference-only. Wrap to avoid building a computation graph for the cached representation.
- [ ] **`collate_observations` must use `expand`, not `repeat`** — `expand` shares memory (no copy), `repeat` allocates. For tiling a single structure to batch size, `expand` is correct.
- [ ] **Import shadowing** — if your wrapper class has the same name as the upstream class, import as `from upstream import Model as _Model`.
- [ ] **`from __future__ import annotations`** — if your file has forward references in type annotations, add this import at the top.
- [ ] **New binary logit functions must be exported** — add to `proteingen.__init__` and document in `predictive_modeling.md`.
- [ ] **New template classes must be exported** — add to `proteingen.__init__` and document in `predictive_modeling.md` and `docs/reference/predictive_modeling.md`.

## Phase 7: Write the Design Doc

Create `src/proteingen/models/<provider>/<provider>.md` following the pattern in `models/rocklin_ddg/rocklin_ddg.md`:

- **Dependencies** — what core abstractions and external packages it uses
- **Used By** — which examples, tests, and downstream components consume it
- **Architecture** — internals (encode/decode split, prediction head, pooling)
- **Tokenizer / OHE** — how tokens map to features, any `token_ohe_basis` override
- **Conditioning** — what goes into `preprocess_observations` vs `forward`
- **Gotchas** — model-specific traps

## Phase 8: Add Documentation

⚠️ **The model is NOT done until this phase is complete.** Passing tests = Phase 6. Don't stop here.

After the model is implemented and tests pass, add user-facing documentation:

1. **Update `docs/models.md`** — add the model to the Predictive Models table and write a section covering:
   - What the model predicts and where it comes from (paper, repo link)
   - How to load and use it with proteingen
   - Code snippets for common workflows (prediction, conditioning, use with TAG/DEG)
   - Document the conditioning `TypedDict` — list each field, its type, and what it represents
   - Which binary logit function is used and what the target means
   - TAG vs DEG compatibility
2. **If you added a new binary logit function or template class**, update `docs/reference/predictive_modeling.md` to document it.
3. **Update `mkdocs.yml`** — add any new pages to the nav if needed.
4. **Verify `models/AGENTS.md`** — confirm the model was added to the registry in Phase 4.
5. **Verify exports** — confirm `src/proteingen/models/__init__.py` exports the class and any conditioning `TypedDict`s. Confirm any new binary logit functions or template classes are exported from `proteingen.__init__`.

## Phase 9: Open a Pull Request

Create a clean branch with only the model integration changes and open a PR against `main`.

If your working branch has accumulated unrelated changes, isolate the model files onto a fresh branch:

```bash
git checkout main && git pull
git checkout -b pr/<model-name>
git checkout <feature-branch> -- \
  src/proteingen/models/<provider>/ \
  src/proteingen/predictive_modeling.py \
  tests/test_<name>.py \
  docs/models.md \
  src/proteingen/models/__init__.py \
  src/proteingen/models/AGENTS.md
# Review staged files, then commit
git diff --cached --stat
git add -A && git commit -m "feat(models): add <ModelName> predictive model wrapper"
git push -u origin pr/<model-name>
```

Note: include `src/proteingen/predictive_modeling.py` in the checkout if you added a new binary logit function or template class.

Before opening the PR, walk through every item in the checklist below and actually verify it — run the tests, check the exports, read the docs page. Don't just check the boxes from memory.

Use the following template for the PR description — copy it directly into the GitHub PR body and fill in each section:

```markdown
## Summary
<!-- What model is being added? Link to paper/repo. What does it predict? -->

## Decomposition
<!-- How was the pretrained predictor decomposed into the four layers? -->
- **Raw predictor**: <!-- What was ported? How much adaptation was needed? -->
- **Binary logit function**: <!-- Existing or new? Which one and why? TAG-compatible? -->
- **Template class**: <!-- Existing, new, or N/A (direct PredictiveModel subclass)? -->
- **PredictiveModel subclass**: <!-- How thin is the glue? What does it add beyond wiring? -->

## Implementation
- **Conditioning**: <!-- What observation variables are supported? Link to the conditioning TypedDict. -->
- **OHE basis**: <!-- Default identity or custom? If custom, what does <mask> map to? -->
- **New dependencies**: <!-- Packages added to pyproject.toml, with versions -->

## Tests
<!-- Describe each test category and what it covers. Must include an output-matching
     test that runs the same real input through both the original library
     and the ProteinGen wrapper, asserting outputs match within floating-point tolerance. -->

## Checklist
- [ ] Model class in `src/proteingen/models/<provider>/`
- [ ] Exported from `models/__init__.py` (class and conditioning `TypedDict` if applicable)
- [ ] Tests pass (`uv run python -m pytest tests/test_<name>.py -v`)
- [ ] Includes output-matching test against original library on real input
- [ ] Design doc at `src/proteingen/models/<provider>/<provider>.md`
- [ ] Listed in `docs/models.md` with code examples
- [ ] Conditioning `TypedDict` documented in `docs/models.md` (if applicable)
- [ ] `models/AGENTS.md` updated
- [ ] TAG/DEG compatibility documented
- [ ] New binary logit function exported and documented (if applicable)
- [ ] New template class exported and documented (if applicable)
- [ ] No breaking changes to existing APIs (or noted below)

## Known Limitations
<!-- TAG compatibility, GPU memory requirements, sequence length limits -->

## Example
<!-- Paste or link to the end-to-end example script from Phase 1 -->
```
