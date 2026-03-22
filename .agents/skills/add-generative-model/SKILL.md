---
name: add-generative-model
description: Step-by-step workflow for integrating a new generative (transition) model into the proteingen library. Covers choosing between TransitionModel and TransitionModelWithEmbedding, implementing abstract methods, writing tests, and avoiding common gotchas. This skill is for generative models only — not PredictiveModel subclasses.
---

# Add a Generative Model

Workflow for wrapping a new pretrained generative model (e.g. a protein language model) into proteingen's `TransitionModel` hierarchy. This skill covers `TransitionModel` (composition) and `TransitionModelWithEmbedding` (ABC with differentiable embeddings). It does **not** cover `PredictiveModel` subclasses — those are a separate concern.

## Phase 1: Understand the New Model

Before writing any code, gather:

1. **Model architecture** — what does its forward pass return? (raw logits tensor, a dataclass, multiple heads?)
2. **Tokenizer** — does it use a HuggingFace tokenizer? What's the vocab size? Are there special tokens (CLS, EOS, PAD, MASK)?
3. **Embedding access** — can you get the embedding layer's weight matrix? Is `ohe @ embed.weight` equivalent to the embedding lookup?
4. **Output dim** — does the output logit dim match vocab_size, or is it padded for alignment (e.g. ESM uses 64-dim output for 33-token vocab)?
5. **Conditioning** — does the model support structure conditioning or other side inputs? If so, how expensive is preprocessing?
6. **Loading** — how is the model loaded? (`from_pretrained`, custom checkpoint, etc.) Any device arg quirks?

Present your findings and confirm with the user before proceeding.

## Phase 2: Create the Directory Structure

```bash
mkdir -p src/proteingen/models/<name>/
```

Files to create:
- `__init__.py` — model class(es), exports
- `<name>.md` — design doc (dependencies, used-by, gotchas)

Update `src/proteingen/models/__init__.py` to export the new class.
Update `src/proteingen/models/AGENTS.md` to add the model to the registry.

## Phase 3: Implement

### Step 1: Choose between TransitionModel and TransitionModelWithEmbedding

Read the design docs to understand the two options:

- `src/proteingen/generative_modeling.md` — TransitionModel and TransitionModelWithEmbedding contracts
- `src/proteingen/probability_model.md` — ProbabilityModel base (conditioning, checkpointing) that both inherit from

```
Can you access the model's embedding layer weights,
and does the model need to support TAG guidance, LinearProbe,
or any workflow requiring gradients through the embedding step?

  YES → TransitionModelWithEmbedding (ABC)
        You implement: differentiable_embedding, embedding_to_outputs
        You get for free: forward, embed, format_raw_to_logits

  NO  → TransitionModel (concrete, composition)
        You pass in: model, tokenizer, logit_formatter
        You may override: format_raw_to_logits (if raw output isn't a tensor)
```

Most protein language models should use `TransitionModelWithEmbedding` — TAG guidance and embedding-based probes are core use cases.

### Step 2a: TransitionModelWithEmbedding (most common path)

Use ESMC as the reference implementation: `src/proteingen/models/esm/__init__.py`.

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

### Step 2b: TransitionModel (composition)

For simpler cases where you just wrap an existing model without needing embedding access:

```python
from proteingen import TransitionModel, MaskedModelLogitFormatter

model = load_my_model(checkpoint)
tokenizer = load_my_tokenizer()
formatter = MaskedModelLogitFormatter(tokenizer, output_dim=model.output_dim)
tm = TransitionModel(model, tokenizer, formatter)
```

Override `format_raw_to_logits` if the model's forward returns something other than a logit tensor.

## Phase 4: Write Tests

Create `tests/test_<name>.py`. Required test categories:

### Construction & basic forward
- Model loads without error
- `forward(seq_SP)` returns expected output shape
- `get_log_probs(seq_SP)` returns valid log probabilities (sum to ~1 after exp)

### Embedding path (TransitionModelWithEmbedding only)
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

## Phase 5: Write the Design Doc

Create `src/proteingen/models/<name>/<name>.md` following the pattern in `models/esm/esm.md`:

- **Dependencies** — what core abstractions and external packages it uses
- **Used By** — which examples, tests, and downstream components consume it
- **Architecture** — internals (what are the sub-modules, how does forward work)
- **Checkpointing** — what gets saved, any stashed args
- **Gotchas** — model-specific traps (device args, lazy loading, dtype issues, OOM thresholds)

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
