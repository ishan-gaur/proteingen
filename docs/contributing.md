# Contributing

ProteinGen is designed to make model contributions straightforward — whether you're using an AI coding agent or adding one by hand.

## Using an agent

The recommended way to add a model is to use an AI coding agent (e.g. Claude Code) with our skill files. The skill walks the agent through a multi-phase process — from reading the original model's source code to opening a pull request.

To kick it off, prompt your agent with:

> "Read the skill file at `.agents/skills/add-generative-model/SKILL.md` and follow it to add **[model name]** to ProteinGen."


or for predictive models:

> "Read the skill file at `.agents/skills/add-predictive-model/SKILL.md` and follow it to add **[model name]** to ProteinGen."

### What the agent does

**Generative models** (9 phases): The agent reads the original source, chooses the right base class (`GenerativeModel` vs `GenerativeModelWithEmbedding`), scaffolds the directory, implements the wrapper, writes tests, updates docs, and opens a PR.

**Predictive models** (9 phases): The agent decomposes the pretrained predictor into four separable layers — the raw model, the binary logit function, an optional template class, and the final `PredictiveModel` subclass — then identifies which layers already exist in the library and only builds what's missing. See [the four layers](#the-four-layers) below.

### What to check

The agent pauses at checkpoints throughout the process. These are the moments where your review matters most.

**During the design phase** — The agent presents its plan before writing any code. This is the most important review point.

- **Simplicity** — wrapping an existing model shouldn't require a complex class hierarchy. If the agent is proposing multiple helper classes or deep abstractions, push back.
- **Feature preservation** — make sure nothing important was dropped to fit the API. The wrapper should preserve the original model's capabilities (conditioning inputs, multiple output heads, etc.), not simplify them away.
- **Conditioning `TypedDict`** — if the model accepts conditioning, the agent will define a `TypedDict` for it. Check that the field names are clear, the types are right, and the format is consistent with existing models.
- **Output `TypedDict`** — same review if the model returns structured output beyond raw logits.
- *(Predictive models only)* **Four-layer decomposition** — check that the agent isn't reinventing something that already exists. Does an existing binary logit function work? Does an existing template class fit?

**After tests** — The agent writes an output-matching test that runs the same real protein input through both the original library and the ProteinGen wrapper, asserting outputs match with `torch.allclose`. This is the single most important test — if it passes, the wrapper is faithful. Confirm it ran and passed.

**Before the PR** — Skim the final code for organization. Models follow a consistent layout: `__init__.py` for re-exports, one `.py` per model class, optional `utils.py` for shared helpers, and a `<provider>.md` design doc. Make sure the new model fits the pattern.

## Adding a model manually

If you're adding a model without an agent, here's what you need to know.

### Generative models

ProteinGen has two generative model base classes (see [generative_modeling](reference/generative_modeling.md) for the full API):

**[`GenerativeModel`](reference/generative_modeling.md#generativemodel)** — wraps any `nn.Module` via composition. You pass in the model, tokenizer, and a logit formatter. Use this when you don't need gradient access to the embedding layer.

```python
from proteingen import GenerativeModel, MaskedModelLogitFormatter

model = load_my_model(checkpoint)
tokenizer = load_my_tokenizer()
formatter = MaskedModelLogitFormatter(tokenizer, output_dim=model.output_dim)
tm = GenerativeModel(model, tokenizer, formatter)
```

**[`GenerativeModelWithEmbedding`](reference/generative_modeling.md#generativemodelwithembedding)** — an ABC extending `GenerativeModel` for models that expose a differentiable embedding path. Required for TAG guidance (backprop through embeddings) and `LinearProbe` (cached deep embeddings). Most protein language models should use this.

You implement two abstract methods:

- `differentiable_embedding(ohe_seq_SPT)` — takes a one-hot (or soft) distribution, returns deep embeddings after the transformer body
- `embedding_to_outputs(embedding_SPD)` — takes deep embeddings, returns raw model output

And set two class variables: `OUTPUT_DIM` and `EMB_DIM`.

Use `MaskedModelLogitFormatter` for masked language models. Use `PassThroughLogitFormatter` for models that don't need output masking. Override `format_raw_to_logits` if the model's forward returns a dataclass instead of a raw tensor.

### Predictive models

Predictive models answer "what is log p(target | sequence)?" and are used by TAG/DEG guidance. See [predictive_modeling](reference/predictive_modeling.md) for the full API.

#### The four layers

Integrating a predictive model means decomposing it into four separable layers:

**Layer 1: Raw Predictor** — the original pretrained model, ported with minimal changes. This isn't proteingen-specific — just get it loading and running inference.

**Layer 2: Binary Logit Function** — a standalone function that converts the raw predictor's output to `(B, 2)` binary logits. This is independent of the model — the same predictor could use different functions depending on the use case. Existing functions:

| Function | Use case | TAG compatible? |
|----------|----------|-----------------|
| `binary_logits` | Single logit output | ✅ |
| `categorical_binary_logits` | Multi-class logits | ✅ |
| `point_estimate_binary_logits` | Thresholded regression | ⚠️ Large k saturates gradients |
| `gaussian_binary_logits` | Mean + variance | ✅ |

If none of these fit, write a new one — it's just a function that returns `(B, 2)`. Add it to `src/proteingen/predictive_modeling.py`.

**Layer 3: Template Model Class** *(optional)* — a reusable `PredictiveModel` subclass that defines an architecture pattern but leaves `format_raw_to_logits` abstract. Existing templates: `LinearProbe`, `OneHotMLP`, `EmbeddingMLP`, `PairwiseLinearModel`. If the predictor's architecture is generalizable, consider adding a new template. If it's one-off, skip this and subclass `PredictiveModel` directly.

**Layer 4: PredictiveModel Subclass** — the glue that wires layers 1–3 together with conditioning, OHE basis, and tokenizer. If layers 1–3 are well-designed, this class should be thin.

#### Key implementation points

**`forward` takes OHE, not token IDs** — `PredictiveModel` converts tokens to one-hot features before calling `forward`. If the original model uses an embedding lookup, replace it with `ohe @ embed.weight` for TAG differentiability.

**`format_raw_to_logits` must return `(B, 2)`** — use one of the binary logit functions above.

**Conditioning split** — `preprocess_observations` runs expensive work once (e.g. structure encoding), `collate_observations` tiles cached results to batch size each forward pass. See `PreTrainedStabilityPredictor` in `models/rocklin_ddg/` for the reference pattern.

**Token OHE basis** — override `token_ohe_basis()` if the tokenizer has tokens that should map to a reduced feature space (e.g. `<mask>` → all-zero vector for TAG compatibility). Register the basis as a buffer.

### Directory layout

Each model family lives under `src/proteingen/models/<provider>/`:

```
models/<provider>/
├── __init__.py        # re-exports only
├── <model>.py         # one file per model class
├── utils.py           # optional shared helpers
└── <provider>.md      # design doc
```

Update `src/proteingen/models/__init__.py` to export the new class and any conditioning `TypedDict`s.

### Required tests

Create `tests/test_<name>.py` covering:

- **Construction & forward** — model loads, `forward()` returns expected shape
- **Output matching** — same real protein input through original library and ProteinGen wrapper, `torch.allclose` on outputs. This is the most important test.
- **Log probabilities** — valid (all ≤ 0, sum to ~1 after exp), temperature scaling works
- **Batching** — single vs batched results are consistent
- **Conditioning** — if applicable: `set_condition_()`, `conditioned_on()`, `collate_observations`
- *(Generative only)* **Embedding path** — `embed()` shape, `embedding_to_outputs(embed(seq))` ≈ `forward(seq)`, gradients flow
- *(Predictive only)* **Binary logit pattern** — `(B, 2)` shape, target switching, gradient flow for TAG

### Documentation

- Add the model to the table and write a section in [Models](models.md) with code examples
- Document conditioning `TypedDict`s (fields, types, meaning) if applicable
- Write a design doc at `src/proteingen/models/<provider>/<provider>.md`
- Update `src/proteingen/models/AGENTS.md`

## Code standards

- Follow the patterns in existing models — look at `src/proteingen/models/esm/` for generative, `models/rocklin_ddg/` for predictive
- Use type hints on all function signatures
- Keep docstrings concise — don't restate what the code already says
- Annotate tensor shapes on every intermediate variable (e.g. `# [B, L, D] - transformer output`)
- Run `uv run ruff check` and `uv run ruff format` before submitting
- Run `uv run python -m pytest tests/ -v` to verify nothing is broken

## Getting help

Open an issue on [GitHub](https://github.com/ishan-gaur/proteingen/issues).
