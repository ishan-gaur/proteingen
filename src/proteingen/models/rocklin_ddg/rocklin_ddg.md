# Stability Predictor (rocklin_ddg) — Design Notes

PMPNN-based stability predictor for protein design guidance.

## Dependencies

- [predictive_modeling.md](../../predictive_modeling.md) — `PreTrainedStabilityPredictor(PredictiveModel)`
- [generative_modeling.md](../../generative_modeling.md) — `MPNNTokenizer`
- [probability_model.md](../../probability_model.md) — conditioning protocol (`preprocess_observations`)
- External: `rc-foundry[all]` (provides `mpnn`, `atomworks`)

## Used By

- [guide.md](../../guide.md) — TAG/DEG use this as the predictive model for stability guidance
- Examples: `stability_guidance/main.py`, `original_stability_guidance/`
- Tests: within `test_tag_projection.py` (indirectly)

## Architecture

- `StabilityPMPNN` — raw PMPNN-based predictor with encode/decode split
  - `encode_structure()` is expensive (runs once per structure)
  - `decode()` is cheap (runs per sample)
  - This split maps to ProbabilityModel's `preprocess_observations` / `forward` pattern
- `PreTrainedStabilityPredictor(PredictiveModel)` — wraps StabilityPMPNN into the proteingen interface
  - Uses binary logit pattern `[0, logit]`
  - Sets `_target = True` by default
  - No longer overrides `get_log_probs` — uses inherited pipeline

## Tokenizer / OHE

- Uses `MPNNTokenizer(include_mask_token=True)` so TAG can pass an explicit predictor-side `<mask>` token
- Overrides `token_ohe_basis()` so predictor `<mask>` maps to an all-zero OHE row — preserves original PMPNN masking semantics while making mask behavior explicit in the interface
- This is a key cross-tokenizer interface point with TAG — see also [guide.md](../../guide.md)

## Supporting Files

- `data_utils.py` — ~300 lines of PMPNN-specific featurization (featurize, prepare_conditioning_inputs, token conversion, PDB loading via biotite)
- `guidance_utils.py` — flow matching Euler sampling + TAG guidance + ESM3 inverse folding wrappers (mostly replicated by core `guide.py` and `sampling.py`)
- `models/utils.py` — `pdb_to_atom37_and_seq` (implemented, has TODOs for multi-chain support)

## Gotchas

- `data_utils.py` and `guidance_utils.py` in this directory are **legacy** — they predate the proteingen abstractions. The old example (`examples/original_stability_guidance/`) uses them directly. The new example (`examples/stability_guidance/main.py`) uses proteingen abstractions.
- Importing from `atomworks` prints env var warnings (CCD_MIRROR_PATH, PDB_MIRROR_PATH) — harmless
