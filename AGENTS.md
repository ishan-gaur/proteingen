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
  - `generative_modeling.py` — `TransitionModel` (concrete), `TransitionModelWithEmbedding` (ABC, adds differentiable embedding), `LogitFormatter` protocol, `MaskedModelLogitFormatter`, `PassThroughLogitFormatter`, `MPNNTokenizer`
  - `predictive_modeling.py` — `PredictiveModel`, `CategoricalPredictiveModel`, `BinaryPredictiveModel`, `PointEstimatePredictiveModel`, `GaussianPredictiveModel`, `LinearProbe`, `OneHotMLP`, `EmbeddingMLP`, `pca_embed_init`
  - `guide.py` — `TAG`, `DEG`, `GuidanceProjection`/`LinearGuidanceProjection` (guidance algorithms)
  - `sampling.py` — `sample_any_order_ancestral` (uses `model.get_log_probs`)
  - `data.py` — `GuidanceDataset` base class, `NoiseSchedule` type alias, schedule functions
  - `models/esm.py` — `ESMC(TransitionModelWithEmbedding)` — ESMC as masked LM + embedding extractor via differentiable embedding
  - `models/rocklin_ddg/` — stability predictor (StabilityPMPNN, PreTrainedStabilityPredictor), data_utils, guidance_utils
  - `models/utils.py` — `pdb_to_atom37_and_seq` (incomplete)
- `examples/unconditional_sampling.py` — working end-to-end ESMC sampling example
- `examples/stability_guidance/main.py` — cleaned-up stability guidance example (uses dfm abstractions, many TODOs)
- `examples/pca_embedding_init.py` — end-to-end example: ESMC PCA → EmbeddingMLP initialization
- `examples/trpb_linear_probe.py` — `TrpBFitnessPredictor(PointEstimatePredictiveModel)` with LinearProbe + ESMC. Trains on cached embeddings, full differentiable forward for guidance. (HF dataset: SaProtHub/Dataset-TrpB_fitness_landsacpe). SLURM job submitted (job 31419).
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

## TransitionModelWithEmbedding Design

- `TransitionModelWithEmbedding(TransitionModel, ABC)` — extends TransitionModel with differentiable embedding support
- Two abstract methods subclasses implement: `differentiable_embedding(ohe_SPT) → emb_SPD` and `embedding_to_logits(emb_SPD) → logits_SPT`
- Public `embed(seq_SP) → emb_SPD` — handles OHE creation from token IDs, padding (vocab_size → OUTPUT_DIM), and calls `differentiable_embedding`. Stashes `self._ohe` for gradient access.
- `forward(seq_SP)` calls `embed(seq_SP)` then `embedding_to_logits(emb)` — two lines
- `format_raw_to_logits` applies `self.logit_formatter` — inherited from TransitionModel pattern
- **OHE padding**: OHE is created at `tokenizer.vocab_size` (e.g. 33 for ESM), then zero-padded to `OUTPUT_DIM` (e.g. 64) before `differentiable_embedding`. Gradients flow back through pad to vocab-sized `_ohe`. TAG only sees vocab-sized gradients.
- `EMB_DIM: int` — subclasses must set (e.g. 960 for ESMC)
- `PreTrainedEmbeddingModel` ABC was deleted — functionality replaced by this class
- LinearProbe now takes `TransitionModelWithEmbedding` as embed_model, calls `embed_model.embed(seq_SP)` for both caching and forward

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
- **Abstract methods** (2): `forward(x_B, **kwargs) -> Any`, `format_raw_to_logits(raw_output, x_B, **kwargs) -> FloatTensor`
- **Concrete methods with defaults**: `preprocess_observations` (pass-through), `collate_observations` (tile-to-batch), `get_log_probs(x_B)`, `with_temp()`, `set_temp_()`, `set_temp()`, conditioning methods
- `preprocess_observations` and `collate_observations` defaults live on ProbabilityModel (not duplicated in TransitionModel/PredictiveModel) — override for custom behavior (e.g. stability predictor's structure encoding)
- `get_log_probs` asserts `self.temp > 0` before dividing
- `get_log_probs` pipeline: `collate_observations(x_B, self.observations)` → `forward(x_B, **obs)` → `format_raw_to_logits(raw, x_B, **obs)` → `log_softmax(logits / temp)`
- `forward` returns `Any` (not just tensors) — allows dataclass outputs like ESMCOutput
- `format_raw_to_logits` receives `x_B` and `**kwargs` so it has full context (e.g. seq_SP for logit formatting)
- `device` property: `next(self.parameters()).device`

## PredictiveModel Design

- `PredictiveModel(ProbabilityModel, ABC)` — adds `tokenizer`, `target` to init. No `model` arg — user subclasses directly.
- **Binary logit pattern**: `format_raw_to_logits` must return `(B, 2)` binary logits `[false_logit, true_logit]`. Parent's `get_log_probs` applies `log_softmax(logits / temp)` → `(B, 2)`. PredictiveModel's `get_log_probs` takes `[:, 1]` → scalar `log p(target | x)`.
- **Binary logit functions** (replacing old subclasses): `categorical_binary_logits(logits_BC, target_class)`, `binary_logits(logit_B, target)`, `point_estimate_binary_logits(pred_B, threshold, k)`, `gaussian_binary_logits(mu_B, log_var_B, threshold)`. Users call these from their `format_raw_to_logits` implementation.
- **Deleted classes**: `CategoricalPredictiveModel`, `BinaryPredictiveModel`, `PointEstimatePredictiveModel`, `GaussianPredictiveModel` — replaced by the standalone functions above.
- **Target management**: `set_target_()` (in-place), `set_target()` (returns self), `with_target()` (context manager with revert). Asserts target is set before `get_log_probs`.
- `get_log_probs(seq_SP)` takes token IDs, creates vocab-sized OHE internally, stashes `self._ohe`, calls `super().get_log_probs(ohe)` → forward → format_raw_to_logits → log_softmax → [:, 1].
- `grad_log_prob(seq_SP)` — runs `get_log_probs`, backprops, returns `self._ohe.grad` (shape B, P, vocab_size). Uses `torch.enable_grad()` context. This is what TAG calls.
- `forward` takes OHE (from get_log_probs). All template models (LinearProbe, OneHotMLP, EmbeddingMLP) are now `PredictiveModel` subclasses — their forward receives OHE and must be differentiable for TAG.
- TODO[pi] on the class: think about tokenizer mismatch — predictor's tokenizer (used for OHE) may differ from underlying model's tokenizer (ESMC's 64-wide embed table vs 33 vocab_size)
- `PredictiveModel` now exposes `token_ohe_basis() -> (T,K)` and `tokens_to_ohe(seq_SP)`.
- Default `token_ohe_basis` is identity (`K=T`), but subclasses can define reduced/structured OHE features (`K!=T`).
- `get_log_probs`/`predict` now go through `tokens_to_ohe`; `grad_log_prob` returns gradients in feature space `(B,P,K)` (not always `(B,P,vocab_size)`).

## LinearProbe Design

- `LinearProbe(nn.Module)` — wraps a `TransitionModelWithEmbedding` + `nn.Linear`
- Constructor takes `embed_model: TransitionModelWithEmbedding`, `output_dim`, optional `pooling_fn(emb_SPD, seq_SP) -> pooled_SD`
- Default pooling: `emb_SPD.mean(dim=1)` — override for masked pooling (e.g. exclude special tokens)
- `pooling_fn` takes **two args**: `(embeddings_SPD, seq_SP)` — seq_SP needed for masking special tokens
- Freezes `embed_model` parameters in `__init__`
- `compute_embeddings(sequences, batch_size, device)` — calls `embed_model.embed(token_ids)` to get pooled embeddings for training
- `forward(seq_SP)` — `embed_model.embed(seq_SP)` → `pooling_fn` → `linear`
- Both `compute_embeddings` and `forward` use the public `embed()` method (not `_embedding` or `differentiable_embedding` directly)

## ESMC Model

- `ESMC(TransitionModelWithEmbedding)` in `models/esm.py` — very concise implementation
- Imports ESM as `from esm.models.esmc import ESMC as _ESMC` to avoid name shadowing
- `OUTPUT_DIM = 64`, `EMB_DIM = 960`
- `differentiable_embedding(ohe_SPT)` — `ohe @ embed.weight` → transformer → returns deep embeddings. ~4 lines.
- `embedding_to_logits(emb_SPD)` — `return self.model.sequence_head(emb_SPD)`. 1 line.
- `forward`, `format_raw_to_logits`, `embed` — all inherited from `TransitionModelWithEmbedding`
- `embed(seq_SP)` (inherited) creates vocab-sized OHE (33), pads to OUTPUT_DIM (64), calls `differentiable_embedding`
- ESM tokenizer `all_special_ids` returns 6 IDs: `<cls>=0, <pad>=1, <eos>=2, <unk>=3, |=31, <mask>=32` — use these for masked pooling
- `_ESMC` internals: `self.model.embed` (nn.Embedding(64,960)), `self.model.transformer` (TransformerStack), `self.model.sequence_head` (Sequential). `_use_flash_attn` is False on this setup.
- `MaskedModelLogitFormatter(tokenizer, OUTPUT_DIM)` — takes 2 args (tokenizer, output_dim)
- No `forward_ohe`, `ESMCEmbedding`, or `PreTrainedEmbeddingModel` — all replaced by TransitionModelWithEmbedding

## ESM3 Model

- `ESM3(TransitionModelWithEmbedding)` in `models/esm.py` — sequence-only masked LM
- `OUTPUT_DIM = 64`, `EMB_DIM = 1536` (esm3-open)
- Loads with `.float()` — ESM3 uses bfloat16 on GPU by default, but needs float32 on CPU
- `from_pretrained` accepts `device=torch.device("cpu")` (same as ESMC)
- ESM3 encoder sums embeddings from 7 tracks: sequence + structure + ss8 + sasa + function + residue + plddt. For sequence-only use, non-sequence tracks use default padding values.
- `_non_sequence_embedding(seq_SP)` computes constant non-sequence track embeddings. Needs `seq_SP` to set structure tokens correctly at special positions (CLS→STRUCTURE_BOS, EOS→STRUCTURE_EOS, PAD→STRUCTURE_PAD).
- `differentiable_embedding`: `ohe @ encoder.sequence_embed.weight` + `_non_sequence_embedding` → transformer (with NaN coords → geom attn disabled)
- `embedding_to_outputs`: `self.model.output_heads.sequence_head(emb)`
- `forward(seq_SP)` must pass `sequence_tokens=seq_SP` as keyword arg — ESM3's forward uses keyword-only params (`*`)
- `format_raw_to_logits` extracts `.sequence_logits.float()` from ESMOutput
- ESM3 constants: `esm.utils.constants.esm3` (imported as `ESM3_CONSTANTS`), `rbf` in `esm.utils.misc`
- `build_affine3d_from_coordinates` from `esm.utils.structure.affine3d` — NaN coords produce all-False affine_mask
- ESM3 internals: `self.model.encoder` (EncodeInputs), `self.model.transformer` (TransformerStack), `self.model.output_heads` (OutputHeads)
- **Structure conditioning**: `set_condition_({"coords_RAX": tensor/np.array})` or `conditioned_on(...)` context manager. `preprocess_observations` runs VQ-VAE encoder once (expensive) → caches `structure_tokens` + `coordinates` (both with BOS/EOS padding). Both `differentiable_embedding` and `forward` read from `self.observations`.
- `_non_sequence_embedding` takes optional `structure_tokens` — if provided (from conditioning), uses them directly; otherwise defaults to STRUCTURE_MASK_TOKEN with special-position overrides
- `differentiable_embedding` uses cached coordinates for `build_affine3d_from_coordinates` → geometric attention enabled; NaN coords when unconditioned → geom attn disabled
- `forward` passes `structure_tokens` and `structure_coords` kwargs to ESM3's native forward when conditioned
- `collate_observations` tiles structure_tokens and coordinates to batch size
- ESM3's structure VQ-VAE encoder: `self.model.encode(ESMProtein(coordinates=coords))` — coordinates must be atom37 format `(L, 37, 3)`. BOS/EOS coordinate padding uses `inf` (not `nan`).
- VQ-VAE warning: `torch.cuda.amp.autocast` deprecation — harmless
- Tests in `tests/test_esm3.py` (21 tests) — construction, embed path vs forward match, gradient flow, log probs, batching, structure conditioning (8 tests), temperature
- Same tokenizer as ESMC: `EsmSequenceTokenizer` (vocab_size=33)

## OneHotMLP / EmbeddingMLP Design

- Both are now `PredictiveModel, ABC` subclasses — user implements `format_raw_to_logits` using the binary logit functions
- `OneHotMLP(PredictiveModel, ABC)` — takes `tokenizer` (derives `vocab_size` from it), no embedding layer. Forward receives OHE, flattens, passes through MLP.
- `EmbeddingMLP(PredictiveModel, ABC)` — takes `tokenizer` + optional `padding_idx` (defaults to `tokenizer.pad_token_id`). Forward does `ohe @ self.embed.weight` for differentiable embedding lookup (enables TAG gradients), flattens, passes through MLP.
- **`init_embed_from_pretrained_pca(source, source_vocab, target_vocab)`** — method on EmbeddingMLP that initializes embedding from PCA of a pretrained `nn.Embedding`. Uses `self.embed_dim` as n_components, `self.vocab_size` as target size — no redundant params. Zeroes the padding row after copy.
- `pca_embed_init()` is now an **internal helper** (not exported from `dfm.__init__`), called by the method above
- Token matching is by string key (e.g. `"A"`, `"C"`) — shared tokens are the intersection of `source_vocab.keys()` and `target_vocab.keys()`. Unmatched tokens (e.g. UNK `"X"` in MPNN vs `"<unk>"` in ESM) naturally get zero rows — no need to filter vocabs before passing
- PCA is computed ONLY over the shared tokens' embeddings (not the full pretrained vocab) — special tokens like `<cls>`, `<mask>` etc. are excluded from the centering and SVD
- Uses `torch.linalg.svd` (NOT `np.linalg.svd`) to keep everything in torch
- ESMC `embed` layer: `Embedding(64, 960)` — 64 tokens (33 real vocab + 31 alignment padding), 960-dim embeddings
- First 20 PCs of ESMC's 20 AA embeddings capture ~100% variance (20 tokens in 960-d = rank 19 after centering, so 20 components is exact)
- Exported from `dfm.__init__`: `OneHotMLP`, `EmbeddingMLP` (not `pca_embed_init`)
- Tests in `tests/test_embedding_mlp.py` (49 tests) and `tests/test_pca_embed_init.py` (21 tests): construction, forward, gradients, predictive pipeline (get_log_probs, grad_log_prob), PCA init, ESMC integration
- Example in `examples/pca_embedding_init.py`
- For tests, use `SimpleNamespace(vocab_size=N, pad_token_id=M)` as a mock tokenizer
- **Design decision**: PCA init is a post-construction method, NOT a constructor param — avoids redundant shape args and lets the model exist before deciding on initialization. Old `initial_embed_weights` constructor param was removed.
- Consumer: `~/kortemme_tyrosine_kinase_design/train_ohe_mlp.py` uses the old API (pre-method `pca_embed_init` + `initial_embed_weights` constructor) — needs updating

## LoRA Adapter Support

- `TransitionModel` has LoRA methods: `apply_lora()`, `has_lora`, `lora_target_modules()`, `save_lora()`, `load_lora()`
- `apply_lora(target_modules, r, lora_alpha, lora_dropout, bias, **kwargs)` — named params, constructs `peft.LoraConfig` internally
- When `target_modules=None`, auto-discovers all `nn.Linear` modules in `self.model` and targets them all
- `lora_target_modules()` returns `dict[pattern, (in_features, out_features, count)]` — collapses block indices to `*` via regex `(?<=\.)\d+(?=\.)`
- After `apply_lora`, `self.model` is a `PeftModel` wrapping the original — attribute access delegates through PEFT's `__getattr__`
- `differentiable_embedding` works with PEFT: `self.model.embed.weight` delegates to base model, transformer Linear layers are LoRA-adapted automatically
- `LinearProbe.__init__` has `freeze_embed_model: bool = True` param — set to `False` when using LoRA so PEFT's freeze/unfreeze state is preserved
- `peft>=0.13.0` added as a regular dependency in `pyproject.toml`
- **LoRA lr must be much lower than head lr** — lr=1e-3 causes mode collapse (constant predictions) with ESM3 LoRA; lr=1e-4 works. Consider separate param groups with different LRs for LoRA vs head.
- **ESM3 + LoRA + structure conditioning**: after `set_condition_()` (loads VQ-VAE lazily), must re-freeze all params then re-enable LoRA params — VQ-VAE's ~30M params would otherwise be trainable
- **ESM3 LoRA OOMs at batch_size≥32** on 48GB GPU due to geometric attention. batch_size=16 with bfloat16 AMP is optimal (~32s/epoch)
- ESM3 LoRA param count: r=8 → ~9.8M trainable (0.69% of 1.4B), r=4 → ~4.9M, r=2 → ~2.5M

## Checkpointing (save / from_checkpoint)

- `ProbabilityModel` defines the protocol: `_save_args() -> dict`, `save(path)`, `from_checkpoint(path)` classmethod
- `save(path)` writes `config.json` with constructor kwargs from `_save_args()`
- `from_checkpoint(path)` reads `config.json`, calls `cls(**args)` to reconstruct
- `TransitionModel.save` adds LoRA adapter to `path/lora_adapter/` if present
- `TransitionModel.from_checkpoint` loads LoRA adapter if `path/lora_adapter/` exists
- `LinearProbe.save` adds `head.pt` (head state dict) + delegates to `embed_model.save(path/embed_model/)`
- `LinearProbe.from_checkpoint` reconstructs via `cls(**config)`, loads LoRA onto embed_model, loads head weights
- Subclasses implement `_save_args()` returning JSON-serializable constructor kwargs
- ESMC stashes `self._esmc_checkpoint`, ESM3 stashes `self._esm3_checkpoint` for `_save_args()`
- Tests: `tests/test_lora.py` (23 tests), `tests/test_esmc_lora.py` (11 tests)

## Stale Tests / Broken Imports

- `test_guidance_data.py::TestGuidanceDataset` — 3 tests fail because they construct `GuidanceDataset` without the now-required `tokenize`, `noise_schedule`, `mask_token` args
- `tests/test_esm.py` — needs updating: ESMC now inherits `TransitionModelWithEmbedding` (not plain `TransitionModel`), `ESMCEmbedding`/`PreTrainedEmbeddingModel` deleted
- `tests/test_sampling.py` — 31 errors (pre-existing, not from this session's changes)
- `models/seki_tyrosine_kinase.py` — manually sets `self.input_dim` which shadows the new property; needs updating
- `guide.py` imports from `dfm.predictive_model` (should be `dfm.predictive_modeling`), also references `ConditionalTransitionModel` which no longer exists
- `sampling.py` now uses `model.get_log_probs` (previously `model.transition_log_probs`)
- Core tests pass: 127 passed (logit_formatter, transition_model, embedding_mlp, pca_embed_init)

## Stability Predictor (rocklin_ddg)

- `StabilityPMPNN` in `models/rocklin_ddg/stability_predictor.py` — PMPNN-based stability predictor with encode/decode split
- `encode_structure()` is expensive (runs once per structure), `decode()` is cheap (runs per sample) — this is the conditioning pattern formalized by ProbabilityModel's `preprocess_observations`
- `PreTrainedStabilityPredictor(PredictiveModel)` wraps StabilityPMPNN — uses binary logit pattern `[0, logit]`, sets `_target = True` by default, no longer overrides `get_log_probs`
- `PreTrainedStabilityPredictor` now uses `MPNNTokenizer(include_mask_token=True)` so TAG can pass an explicit predictor-side `<mask>` token.
- Stability predictor overrides `token_ohe_basis()` so predictor `<mask>` maps to an all-zero OHE row; this preserves original PMPNN masking semantics while making mask behavior explicit in the interface.
- The old working example (`models/rocklin_ddg/example_usage.py`) uses local `data_utils.py` and `guidance_utils.py` — these do NOT use dfm abstractions
- `data_utils.py` has ~300 lines of PMPNN-specific featurization (featurize, prepare_conditioning_inputs, token conversion, PDB loading via biotite)
- `guidance_utils.py` has flow matching Euler sampling + TAG guidance + ESM3 inverse folding wrappers — most of this is replicated by `guide.py` (TAG/DEG) and `sampling.py`
- The new example (`examples/stability_guidance/main.py`) uses dfm abstractions but has many unresolved TODOs
- **Next steps**: implement PreTrainedStabilityPredictor.forward using ProbabilityModel conditioning (preprocess_observations = encode_structure, forward uses cached embeddings + decoder), finish pdb_to_atom37_and_seq in models/utils.py, get the new example working, then delete the old code

## TAG / Guidance Gotchas

- **Gradient vanishes on <mask> tokens**: predictor trained on real AA embeddings → gradients through frozen ESMC transformer vanish ~10^6x when input contains <mask> tokens. Fix: fill mask positions with gen model's argmax before computing predictor gradients.
- **Steep sigmoid saturates gradient**: with `k=100` and prediction above threshold, `sigmoid(k*(pred-threshold))` → 1, gradient → 0. Use lower k (5-10) or enumeration-based guidance (DEG) instead.
- **Enumeration > gradient for frozen-LM probes**: TAG gradients through 30-layer frozen transformer are unreliable. DEG-style enumeration (evaluate all 20 AAs at each position) gives much better guidance because it only needs correct rankings, not accurate gradients.
- **Gen temperature is key for guidance**: ESMC prior at well-determined positions (e.g. G at pos 227 with log prob -0.00) is essentially unoverridable. Higher gen temp (2-3) flattens the prior, giving guidance room to steer. Combined with guidance scale (10-20), this produces significant improvements.
- **TrpB results**: Unguided mean=0.48, Enum(scale=20,temp=3) mean=0.62, %>0.7 from 0.5% to 32.5% (N=200, 10 runs)
- **ESMC-300m vs 600m**: essentially identical ρ on TrpB (0.38 vs 0.39). Varpos-concat pooling (4 positions × D) is better than full mean pool (0.38 vs 0.28 for MLP).
- **No separate `guidance_scale`** — predictive model's temperature controls guidance strength (lower temp → steeper log_softmax → larger gradient/log_prob magnitude). Gen model temp controls prior flatness.
- TAG/DEG `argmax_masked_positions` flag: when True, fills mask tokens with gen argmax before evaluating predictor. Needed when predictor wasn't trained on noisy/masked inputs.
- DEG requires position info via `at_position()` context manager — `sample_any_order_ancestral` passes this automatically
- `sampling.py` `any_order_ancestral_step` now takes model (not just callable) — picks positions first, then sets DEG positions via `at_position` before computing log probs
- Cached varpos embeddings: `data/trpb_embeddings/{esmc_300m,esmc_600m}_varpos/{train,valid,test}.pt`
- ESMC `EMB_DIM` is now dynamic (instance variable set from model weights), not a class constant
- TAG architecture now keeps tokenizer/OHE mismatch logic in a TAG-owned projection layer (`GuidanceProjection`) rather than inside predictive models.
- Default TAG path uses `LinearGuidanceProjection` with explicit first-order math: `delta(t) = g · (M[t] - M[baseline])`, where `M` maps gen tokens to predictor OHE features.
- `LinearGuidanceProjection` handles token mapping, optional fallback token for unmapped symbols, and CLS/EOS window stripping when predictor does not model those positions.
- New focused tests in `tests/test_tag_projection.py` validate prepare-step token/position mapping and the Taylor delta math.

## Stability Demo Reimplementation Notes

- `examples/stability_guidance/main.py` now mirrors `examples/original_stability_guidance/example_usage.py` sampling settings more closely.
- Reimplementation uses flow-matching Euler sampling (`dt=0.01`, `x1_temp=0.1`, `num_samples=100`, `batch_size=50`) rather than `sample_euler`.
- Denoising logits in the reimplementation apply original-style token constraints: fixed CLS/EOS behavior and inner-position restriction to canonical 20 AAs.
- Guided run uses DFM `TAG` with the projection-based cross-tokenizer path, so behavior is closer to the original demo while staying inside DFM abstractions.

## SLURM

- General-purpose submit script: `~/slurm/run_python.sh` — supports `--uv` and `--conda <env>` modes
- Usage: `bash ~/slurm/run_python.sh --uv examples/trpb_linear_probe.py --device cuda`
- Output goes to `~/slurm/output/<job_name>.out` / `.err`
- Single node with 4x NVIDIA RTX 6000 Ada (49GB each), 128 CPUs, 500GB RAM, partition=long

## External Dependencies

- ProteinMPNN via Foundry: `rc-foundry[all]` — provides `mpnn` and `atomworks` packages
- `MPNNTokenizer` in `generative_model.py` wraps PMPNN's `MPNN_TOKEN_ENCODING` (21 tokens: 20 standard AAs + UNK at idx 20)
- Importing from `atomworks` prints env var warnings (CCD_MIRROR_PATH, PDB_MIRROR_PATH) — these are harmless

## MkDocs Documentation

- MkDocs Material theme with `pymdownx.tabbed` (alternate_style), `pymdownx.highlight` (anchor_linenums), `pymdownx.superfences`, `admonition`, `attr_list`
- Content tabs: `=== "Tab Name"` syntax — content must be indented 4 spaces under the tab header
- Code block line highlighting: `` ```python hl_lines="2 3 11" `` — works inside tabs (indented code fences)
- Build: `uv run mkdocs build` — site output in `site/`
- Serve locally: `uv run mkdocs serve` — site is served at `/proteingen/` prefix (from `site_url`), not root `/`
- True side-by-side columns (CSS grid) not feasible in MkDocs Markdown — content tabs are the idiomatic alternative
- Known warning: `index.md` link `ishangaur.com` flagged as broken (missing `https://` prefix)
- **Repo renamed to `proteingen`** — display name is **ProteinGen** (capitalized in all prose), but URLs/paths/package-slug stay lowercase `proteingen`
- Git remote updated: `git@github.com:ishan-gaur/proteingen.git`
- Python package is still `dfm` internally — TODO[pi] to rename to `proteingen` in pyproject.toml/src/imports/tests
- `mkdocstrings[python]` configured with `paths: [src]` so it resolves `dfm.*` modules from `src/dfm/`
- API reference pages are thin stubs with `::: dfm.<module>` directives — mkdocstrings auto-generates from docstrings
- griffe warnings about missing type annotations in `generative_modeling.py` (line 152 `**kwargs`, line 408 return) — harmless, fix by adding annotations
- `PLAN.md` at repo root documents the full docs roadmap and what's implemented vs TODO
- Nav structure: Home | Setup | Examples | Models | Workflows (Overview, ProteinGuide) | Contributing | Reference (Design Philosophy, API Reference per-module)
- Workflow sub-pages are all stubs with "Coming soon" admonitions and TODO[pi] markers
- `site/` is in `.gitignore`

## Tokenization

- PMPNN vocabulary: 20 standard amino acids + UNK (X), indexed 0–20
- Mapping: one-letter AA → three-letter code (atomworks `DICT_THREE_TO_ONE`) → PMPNN index (`MPNN_TOKEN_ENCODING.token_to_idx`)
- `MPNNTokenizer()`: encode("ACDE") → [0,4,3,6], decode([0,4,3,6]) → "ACDE", __call__(["ACDE"]) → {"input_ids": tensor}
- ESM tokenizer (`EsmSequenceTokenizer`): vocab_size=33, indices 0–3 are special (`<cls>`, `<pad>`, `<eos>`, `<unk>`), AAs at 4–23 (e.g. A=5, L=4, C=23), non-standard at 24–31, `<mask>`=32
- Both tokenizers expose `.vocab` as `dict[str, int]` (token string → index) — this is the interface `pca_embed_init` uses for cross-tokenizer mapping
- Simple 20-AA vocab for predictive models: `{aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}` with padding at index 20, vocab_size=21
