# ESM Models — Design Notes

ESMC and ESM3 wrappers in `models/esm/__init__.py`. Both subclass `GenerativeModelWithEmbedding`.

## Dependencies

- [generative_modeling.md](../../generative_modeling.md) — `GenerativeModelWithEmbedding`, `MaskedModelLogitFormatter`
- [probability_model.md](../../probability_model.md) — conditioning, checkpointing protocol
- External: `esm` package (EvolutionaryScale)

## Used By

- [predictive_modeling.md](../../predictive_modeling.md) — `LinearProbe` wraps these as `embed_model`
- [guide.md](../../guide.md) — TAG/DEG use these as the generative model
- Examples: `unconditional_sampling.py`, `trpb_linear_probe.py`, `esm3_structure_conditioned_sampling.py`
- Tests: `test_esm.py` (ESMC), `test_esm3.py` (21 tests), `test_esmc_lora.py` (11 tests)

## Shared

- Both use `EsmSequenceTokenizer` (vocab_size=33)
- `OUTPUT_DIM = 64` (ESM's embed table is 64-wide for alignment, only 33 tokens are real)
- `MaskedModelLogitFormatter(tokenizer, OUTPUT_DIM)` — takes 2 args
- ESM tokenizer `all_special_ids`: `<cls>=0, <pad>=1, <eos>=2, <unk>=3, |=31, <mask>=32` — use for masked pooling
- Imports ESM as `from esm.models.esmc import ESMC as _ESMC` (and similar for ESM3) to avoid name shadowing

## ESMC

- `EMB_DIM = 960`
- Very concise — `differentiable_embedding` is ~4 lines: `ohe @ embed.weight` → transformer → deep embeddings
- `embedding_to_logits`: `self.model.sequence_head(emb_SPD)` — 1 line
- `forward`, `format_raw_to_logits`, `embed` — all inherited from `GenerativeModelWithEmbedding`
- `_ESMC` internals: `self.model.embed` (nn.Embedding(64,960)), `self.model.transformer` (TransformerStack), `self.model.sequence_head` (Sequential). `_use_flash_attn` is False on this setup.
- `EMB_DIM` is dynamic (instance variable set from model weights), not a class constant

## ESM3

- `EMB_DIM = 1536` (esm3-open)
- Loads with `.float()` — ESM3 uses bfloat16 on GPU by default, but needs float32 on CPU
- ESM3 encoder sums embeddings from 7 tracks: sequence + structure + ss8 + sasa + function + residue + plddt. For sequence-only use, non-sequence tracks use default padding values.
- `_non_sequence_embedding(seq_SP)` computes constant non-sequence track embeddings. Needs `seq_SP` to set structure tokens correctly at special positions (CLS→STRUCTURE_BOS, EOS→STRUCTURE_EOS, PAD→STRUCTURE_PAD).
- `differentiable_embedding`: `ohe @ encoder.sequence_embed.weight` + `_non_sequence_embedding` → transformer (with NaN coords → geom attn disabled)
- `embedding_to_logits`: `self.model.output_heads.sequence_head(emb)`
- `forward(seq_SP)` must pass `sequence_tokens=seq_SP` as keyword arg — ESM3's forward uses keyword-only params (`*`)
- `format_raw_to_logits` extracts `.sequence_logits.float()` from ESMOutput
- ESM3 internals: `self.model.encoder` (EncodeInputs), `self.model.transformer` (TransformerStack), `self.model.output_heads` (OutputHeads)
- ESM3 constants: `esm.utils.constants.esm3` (imported as `ESM3_CONSTANTS`), `rbf` in `esm.utils.misc`

### Structure Conditioning

- `set_condition_({"coords_RAX": tensor/np.array})` or `conditioned_on(...)` context manager
- `preprocess_observations` runs VQ-VAE encoder once (expensive) → caches `structure_tokens` + `coordinates` (both with BOS/EOS padding)
- Both `differentiable_embedding` and `forward` read from `self.observations`
- `_non_sequence_embedding` takes optional `structure_tokens` — if provided, uses directly; otherwise defaults to STRUCTURE_MASK_TOKEN with special-position overrides
- `differentiable_embedding` uses cached coordinates for `build_affine3d_from_coordinates` → geometric attention enabled; NaN coords when unconditioned → geom attn disabled
- `forward` passes `structure_tokens` and `structure_coords` kwargs to ESM3's native forward when conditioned
- `collate_observations` tiles structure_tokens and coordinates to batch size
- VQ-VAE encoder: `self.model.encode(ESMProtein(coordinates=coords))` — coordinates must be atom37 format `(L, 37, 3)`. BOS/EOS coordinate padding uses `inf` (not `nan`).
- `build_affine3d_from_coordinates` from `esm.utils.structure.affine3d` — NaN coords produce all-False affine_mask
- VQ-VAE warning: `torch.cuda.amp.autocast` deprecation — harmless

## Checkpointing

- ESMC stashes `self._esmc_checkpoint`, ESM3 stashes `self._esm3_checkpoint` for `_save_args()`
- `from_pretrained` accepts `device=torch.device("cpu")` — must be `torch.device`, NOT string `"cpu"` (ESM code calls `device.type`)

## Gotchas

- **ESM tokenizer v3.0.3**: `mask_token_id` is `None` — always use `tokenizer.vocab["<mask>"]` instead
- **ESM3 structure conditioning is length-locked** — `set_condition_()` preprocesses coords to fixed-length structure tokens (L+2 with BOS/EOS). All subsequent `embed()`/`differentiable_embedding()` calls must use sequences of exactly that length or you get shape mismatch.
- **ESM3 `_structure_encoder` loaded lazily** — `ESM3.encode()` (called by `set_condition_()`) loads the VQ-VAE encoder on first use. If you froze params before calling `set_condition_()`, the encoder's ~30M params won't be frozen. Always freeze AFTER conditioning setup.
- **ESM3 geometric attention OOMs at large batch sizes** — `geom_attn` computes pairwise (L×L) tensors. batch_size≥32 OOMs on 48GB GPU for L=297. Use batch_size≤16 with bfloat16 AMP.

## LoRA

- See also [generative_modeling.md](../../generative_modeling.md) for the general LoRA API
- `differentiable_embedding` works with PEFT: `self.model.embed.weight` delegates to base model, transformer Linear layers are LoRA-adapted automatically
- **ESM3 + LoRA + structure conditioning**: after `set_condition_()` (loads VQ-VAE lazily), must re-freeze all params then re-enable LoRA params — VQ-VAE's ~30M params would otherwise be trainable
- **LoRA lr must be much lower than head lr** — lr=1e-3 causes mode collapse with ESM3 LoRA; lr=1e-4 works. Consider separate param groups.
- ESM3 LoRA param count: r=8 → ~9.8M trainable (0.69% of 1.4B), r=4 → ~4.9M, r=2 → ~2.5M
- ESM3 LoRA OOMs at batch_size≥32 on 48GB GPU. batch_size=16 with bfloat16 AMP is optimal (~32s/epoch)
