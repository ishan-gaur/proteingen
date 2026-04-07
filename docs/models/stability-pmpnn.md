# StabilityPMPNN

ProteinMPNN-based stability predictor from the [ProteinGuide paper](https://arxiv.org/abs/2505.04823). Trained on the [Rocklin Megascale stability dataset](https://www.nature.com/articles/s41586-022-04604-z) to predict thermodynamic stability (ΔΔG) from structure + sequence.

- **Encode/decode split**: `encode_structure()` runs once per structure (expensive), `decode()` runs per sequence sample (cheap). This maps naturally to ProbabilityModel's `preprocess_observations` / `forward` pattern.
- **Tokenizer**: `MPNNTokenizer` with 21 tokens (20 standard AAs + UNK). When used with TAG, `include_mask_token=True` adds `<mask>` at idx 21.

## Cross-tokenizer behavior

The stability predictor overrides `token_ohe_basis()` so that the `<mask>` token maps to an all-zero OHE row — preserving original PMPNN masking semantics while making mask behavior explicit in the interface. This is a key integration point with TAG's `GuidanceProjection`.

<!-- TODO[pi]: figure out model storage strategy — should we convert all models to HuggingFace format? Use torch hub cache as default (like evodiff loads from zenodo)? Or extend the save/from_checkpoint interface to support both HF and custom loading? This affects how contributed models are distributed. -->
