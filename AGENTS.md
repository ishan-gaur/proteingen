# Agent Guidelines

## Project Management

- Use `uv` for all package management and running Python code
  - Install dependencies: `uv add <package>`
  - Run scripts: `uv run python <script>`
  - Sync environment: `uv sync`
  - Install package in editable mode: `uv pip install -e .`
  - Run tests: `uv run pytest tests/ -v`
  - Run formatter: `uv run ruff format`
  - Run lineter: `uv run ruff check`
- Use this file to note down project-related info important to know across sessions
- **Discuss design decisions before implementing** - especially for abstractions and class structures
- **Ask for clarification when instructions seem contradictory** - don't guess intent, surface the confusion

## Code Style

- Follow PEP 8 conventions
- Use type hints for function signatures
- Keep functions focused and modular
- Make user-provided inputs required, not optional - if the user needs to provide something, the interface should demand it
- Prefer strict interfaces that prevent misuse over permissive ones that raise errors at runtime
- Let code crash naturally rather than wrapping in try/catch - silent failures hide bugs
- Use assert statements to validate inputs/outputs in complex pipelines and tricky functions - they serve as executable documentation and catch issues early
- Follow John Ousterhouts A Philosophy of Software Design
- Use `raise ValueError` for input validation at API boundaries, asserts for internal invariants
- Prefer torch.Tensor over numpy arrays when data will be used in training
- Keep return structures flat and minimal - only include what's actually needed
- Implement functionality or remove it - no silent no-ops (e.g. don't accept a parameter and ignore it)
- Direct indexing over clever fallbacks - if something should be indexable, just index it and let it crash if not

## Project Structure

- `src/dfm/` — core library (installed as editable package; run `uv pip install -e .` after changes)
  - `probability_model.py` — `ProbabilityModel` ABC (shared base: temp, conditioning, abstract forward/format_raw_to_logits/preprocess_observations/collate_observations, concrete get_log_probs)
  - `generative_modeling.py` — `TransitionModel` (concrete, inherits ProbabilityModel), `LogitFormatter` protocol, `MaskedModelLogitFormatter`, `PassThroughLogitFormatter`, `MPNNTokenizer`
  - `predictive_modeling.py` — `PredictiveModel` (inherits `ProbabilityModel`), `ClassValuedPredictiveModel`, `RealValuedPredictiveModel`, `OneHotMLP`, `EmbeddingMLP`, `LinearProbe`, `pca_embed_init`
  - `guide.py` — `TAG`, `DEG`, `TokenizerTranslator` (guidance algorithms)
  - `sampling.py` — `sample_any_order_ancestral` (uses `model.get_log_probs`)
  - `data.py` — `GuidanceDataset` base class, `NoiseSchedule` type alias, schedule functions
  - `models/esm.py` — `ESMC(TransitionModel)`, `ESMCEmbedding(PreTrainedEmbeddingModel)` — ESMC as masked LM and as frozen embedding extractor
  - `models/rocklin_ddg/` — stability predictor (StabilityPMPNN, PreTrainedStabilityPredictor), data_utils, guidance_utils
  - `models/utils.py` — `pdb_to_atom37_and_seq` (incomplete)
- `examples/unconditional_sampling.py` — working end-to-end ESMC sampling example
- `examples/stability_guidance/main.py` — cleaned-up stability guidance example (uses dfm abstractions, many TODOs)
- `examples/pca_embedding_init.py` — end-to-end example: ESMC PCA → EmbeddingMLP initialization
- `examples/trpb_linear_probe.py` — trains LinearProbe on ESMC embeddings for TrpB fitness prediction (HF dataset: SaProtHub/Dataset-TrpB_fitness_landsacpe)
- `tests/` — pytest tests (`test_guidance_data.py`, `test_logit_formatter.py`, `test_transition_model.py`, `test_embedding_mlp.py`, `test_pca_embed_init.py`)
- `TODO.md` — phased roadmap (Phase 1 done, Phase 2–4 pending)
- **Deleted**: `mixins.py` (conditioning folded into ProbabilityModel), `ConditionalTransitionModel`, `ConditionalPredictiveModel`

## TransitionModel Design

- `TransitionModel(ProbabilityModel)` — **concrete class** (not ABC), uses composition pattern
- `__init__` takes `model: nn.Module`, `tokenizer`, `logit_formatter` — wraps any model
- `forward` calls `self.model(seq_SP, **kwargs)` and returns raw output (may be non-tensor, e.g. ESMCOutput dataclass)
- `format_raw_to_logits` applies `self.logit_formatter(raw_output, seq_SP)` — default assumes raw output is already a tensor
- Subclasses override `format_raw_to_logits` when the wrapped model returns non-tensor output (e.g. ESMC extracts `.sequence_logits.float()` then applies logit_formatter)
- `preprocess_observations` and `collate_observations` have sensible defaults (pass-through and tile-to-batch-size) — override for structure conditioning
- `generative_modeling.py` uses `from __future__ import annotations` for lazy annotation eval (needed because `TransitionModel` references `LogitFormatter` which is defined later in the file)
- Run tests with `uv run python -m pytest` (not `uv run pytest` — pytest not on PATH directly) [×1]
- The ESM child class in `~/PALM/esm-cath/src/esm_cath/model.py` is an older consumer — may need updating to new composition pattern

## LogitFormatter / MaskedModelLogitFomatter Design

- `LogitFormatter` is a `@runtime_checkable Protocol` — defines `__call__(logits, input_ids) -> FloatTensor`
- `MaskedModelLogitFomatter` inherits `(nn.Module, LogitFormatter)` — **nn.Module must come first** in MRO or Protocol's `__call__` shadows nn.Module's (returns None instead of dispatching to `forward`)
- Uses `nn.Module.__init__(self)` instead of `super().__init__()` because Protocol in the MRO breaks cooperative `super()` chain for nn.Module init
- The mask matrix uses `register_buffer` for device tracking (no gradients, moves with `.to(device)`)
- Uses **direct indexing** (`mask_matrix[token_ids]`) NOT one-hot matmul — `0.0 * (-inf) = NaN` in IEEE floats kills the matmul approach
- Uses **additive masking** (0.0 pass-through, -inf block) NOT multiplicative — multiplying logits by -inf gives wrong signs for negative logits
- `output_dim` can exceed `vocab_size` for memory alignment (e.g. ESM's 33-token vocab → 64-dim output)
- ESM tokenizer v3.0.3: `mask_token_id` is `None` — always use `tokenizer.vocab["<mask>"]` instead
- Current constructor is HF-tokenizer-specific (uses `.vocab`, `.added_tokens_decoder`) — TODO to add a general constructor taking primitive ids
- Reference consumer: `~/PALM/esm-cath/src/esm_cath/model.py` ESM class
- Tests in `tests/test_logit_formatter.py` (24 tests) cover ESM and BERT tokenizers

## ProbabilityModel Design

- `ProbabilityModel(nn.Module, ABC)` in `probability_model.py` — shared base for ALL models in the library
- **Conditioning built in** — `observations`, `set_condition_()`, `set_condition()`, `conditioned_on()` context manager (with revert)
- **Abstract methods** (4): `forward(x_B, **kwargs) -> Any`, `format_raw_to_logits(raw_output, x_B, **kwargs) -> FloatTensor`, `preprocess_observations(obs) -> obs`, `collate_observations(x_B, obs) -> obs`
- **Concrete methods**: `get_log_probs(x_B)`, `with_temp()`, `set_temp_()`, `set_temp()`, conditioning methods
- `get_log_probs` pipeline: `collate_observations(x_B, self.observations)` → `forward(x_B, **obs)` → `format_raw_to_logits(raw, x_B, **obs)` → `log_softmax(logits / temp)`
- `forward` returns `Any` (not just tensors) — allows dataclass outputs like ESMCOutput
- `format_raw_to_logits` receives `x_B` and `**kwargs` so it has full context (e.g. seq_SP for logit formatting)
- All 4 abstract methods are **intentionally abstract** even for unconditional models — forces implementers to explicitly consider each concern
- `device` property: `next(self.parameters()).device`

## PredictiveModel Design

- `PredictiveModel(ProbabilityModel, ABC)` — adds `tokenizer` to init
- Still has `with_target` (abstract), `target_log_probs_given_ohe` (on subclasses), `get_log_prob_target_given_seq` (convenience)
- `ClassValuedPredictiveModel` and `RealValuedPredictiveModel` exist but will be rewritten as `TargetProbabilityMixin` variants
- Child classes must set `self.input_dim` for `get_log_prob_target_given_seq` to work

## ESMC Model

- `ESMC(TransitionModel)` in `models/esm.py` — single clean subclass
- Imports ESM as `from esm.models.esmc import ESMC as _ESMC` to avoid name shadowing (the dfm class is also called ESMC)
- Overrides `format_raw_to_logits` to extract `.sequence_logits.float()` from `ESMCOutput` dataclass, then applies logit_formatter
- `OUTPUT_DIM = 64` — ESM's 33-token vocab padded to 64-dim output for memory alignment
- `model.embed` is an `nn.Embedding(64, 960)` — to load for PCA: `ESMC.from_pretrained("esmc_300m", device=torch.device("cpu"))` (must pass `torch.device`, not string "cpu" — ESM calls `.type` on it)
- `MaskedModelLogitFormatter(tokenizer, OUTPUT_DIM)` — takes 2 args (tokenizer, output_dim), NOT 3 (old code erroneously passed `"<mask>"` as second arg)
- ESM3IF stub removed — left as TODO[pi]

## ESMCEmbedding Model

- `ESMCEmbedding(PreTrainedEmbeddingModel)` in `models/esm.py` — frozen ESMC feature extractor for downstream probes
- `EMB_DIM = 960` — mean-pooled over non-special positions (CLS, EOS, PAD masked out before pooling)
- All ESMC parameters frozen (`requires_grad = False`) — pure feature extractor
- `forward(seq_SP)` takes token IDs, returns pooled embeddings `(S, D)`
- `forward_ohe(ohe_seq_SPT)` satisfies `PreTrainedEmbeddingModel` ABC, returns `(log_probs, pooled_embeddings)`
- `tokenize(sequences)` convenience method: raw AA strings → token IDs with CLS/EOS framing
- Compatible with `LinearProbe(embed_model, output_dim)` — probe calls `embed_model(token_ids)` which dispatches to `forward`
- Exported from `dfm.models`

## EmbeddingMLP / pca_embed_init Design

- `EmbeddingMLP` in `predictive_modeling.py` — MLP with learned `nn.Embedding` layer, alternative to `OneHotMLP`'s frozen identity embedding
- **`init_embed_from_pretrained_pca(source, source_vocab, target_vocab)`** — method on EmbeddingMLP that initializes embedding from PCA of a pretrained `nn.Embedding`. Uses `self.embed_dim` as n_components, `self.vocab_size` as target size — no redundant params. Zeroes the padding row after copy.
- `pca_embed_init()` is now an **internal helper** (not exported from `dfm.__init__`), called by the method above
- Token matching is by string key (e.g. `"A"`, `"C"`) — shared tokens are the intersection of `source_vocab.keys()` and `target_vocab.keys()`. Unmatched tokens (e.g. UNK `"X"` in MPNN vs `"<unk>"` in ESM) naturally get zero rows — no need to filter vocabs before passing
- PCA is computed ONLY over the shared tokens' embeddings (not the full pretrained vocab) — special tokens like `<cls>`, `<mask>` etc. are excluded from the centering and SVD
- Uses `torch.linalg.svd` (NOT `np.linalg.svd`) to keep everything in torch
- ESMC `embed` layer: `Embedding(64, 960)` — 64 tokens (33 real vocab + 31 alignment padding), 960-dim embeddings
- First 20 PCs of ESMC's 20 AA embeddings capture ~100% variance (20 tokens in 960-d = rank 19 after centering, so 20 components is exact)
- Exported from `dfm.__init__`: `EmbeddingMLP` (not `pca_embed_init`)
- Tests in `tests/test_pca_embed_init.py` (21 tests): synthetic fast tests + ESMC integration tests (module-scoped fixture loads model once)
- Example in `examples/pca_embedding_init.py`
- **Design decision**: PCA init is a post-construction method, NOT a constructor param — avoids redundant shape args and lets the model exist before deciding on initialization. Old `initial_embed_weights` constructor param was removed.
- Consumer: `~/kortemme_tyrosine_kinase_design/train_ohe_mlp.py` uses the old API (pre-method `pca_embed_init` + `initial_embed_weights` constructor) — needs updating

## Stale Tests / Broken Imports

- `test_guidance_data.py::TestGuidanceDataset` — 3 tests fail because they construct `GuidanceDataset` without the now-required `tokenize`, `noise_schedule`, `mask_token` args
- `tests/test_esm.py` — may need updating for new TransitionModel composition pattern + ESMC changes
- `guide.py` imports from `dfm.predictive_model` (should be `dfm.predictive_modeling`), also references `ConditionalTransitionModel` which no longer exists
- `sampling.py` now uses `model.get_log_probs` (previously `model.transition_log_probs`)
- `PassThroughLogitFormatter` has its own `__init__(self)` to avoid inheriting `LogitFormatter` Protocol's `__init__` (which requires a tokenizer arg)
- `MaskedModelLogitFormatter` constructor takes `(tokenizer, output_dim=None)` — no `mask_token` string arg (old tests passed `"<mask>"` as second positional, which is wrong)
- `MaskedModelLogitFormatter` extra columns beyond `vocab_size` (when `output_dim > vocab_size`) are blocked (`-inf`) for ALL input tokens including mask — they don't correspond to real tokens so no probability mass should flow there

## Stability Predictor (rocklin_ddg)

- `StabilityPMPNN` in `models/rocklin_ddg/stability_predictor.py` — PMPNN-based stability predictor with encode/decode split
- `encode_structure()` is expensive (runs once per structure), `decode()` is cheap (runs per sample) — this is the conditioning pattern formalized by ProbabilityModel's `preprocess_observations`
- `PreTrainedStabilityPredictor(ClassValuedPredictiveModel)` wraps StabilityPMPNN — syntax errors fixed (missing self, `class` keyword), but forward/conditioning not yet implemented
- The old working example (`models/rocklin_ddg/example_usage.py`) uses local `data_utils.py` and `guidance_utils.py` — these do NOT use dfm abstractions
- `data_utils.py` has ~300 lines of PMPNN-specific featurization (featurize, prepare_conditioning_inputs, token conversion, PDB loading via biotite)
- `guidance_utils.py` has flow matching Euler sampling + TAG guidance + ESM3 inverse folding wrappers — most of this is replicated by `guide.py` (TAG/DEG) and `sampling.py`
- The new example (`examples/stability_guidance/main.py`) uses dfm abstractions but has many unresolved TODOs
- **Next steps**: implement PreTrainedStabilityPredictor.forward using ProbabilityModel conditioning (preprocess_observations = encode_structure, forward uses cached embeddings + decoder), finish pdb_to_atom37_and_seq in models/utils.py, get the new example working, then delete the old code

## External Dependencies

- ProteinMPNN via Foundry: `rc-foundry[all]` — provides `mpnn` and `atomworks` packages
- `MPNNTokenizer` in `generative_model.py` wraps PMPNN's `MPNN_TOKEN_ENCODING` (21 tokens: 20 standard AAs + UNK at idx 20)
- Importing from `atomworks` prints env var warnings (CCD_MIRROR_PATH, PDB_MIRROR_PATH) — these are harmless

## Tokenization

- PMPNN vocabulary: 20 standard amino acids + UNK (X), indexed 0–20
- Mapping: one-letter AA → three-letter code (atomworks `DICT_THREE_TO_ONE`) → PMPNN index (`MPNN_TOKEN_ENCODING.token_to_idx`)
- `MPNNTokenizer()`: encode("ACDE") → [0,4,3,6], decode([0,4,3,6]) → "ACDE", __call__(["ACDE"]) → {"input_ids": tensor}
- ESM tokenizer (`EsmSequenceTokenizer`): vocab_size=33, indices 0–3 are special (`<cls>`, `<pad>`, `<eos>`, `<unk>`), AAs at 4–23 (e.g. A=5, L=4, C=23), non-standard at 24–31, `<mask>`=32
- Both tokenizers expose `.vocab` as `dict[str, int]` (token string → index) — this is the interface `pca_embed_init` uses for cross-tokenizer mapping
- Simple 20-AA vocab for predictive models: `{aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}` with padding at index 20, vocab_size=21
