# Frame2seq — Design Notes

Structure-conditioned masked inverse folding model from Akpinaroglu et al.

- **Paper**: [Structure-conditioned masked language models for protein sequence design generalize beyond the native sequence space](https://doi.org/10.1101/2023.12.15.571823)
- **Repo**: [dakpinaroglu/Frame2seq](https://github.com/dakpinaroglu/Frame2seq)
- **PyPI package**: `frame2seq>=0.0.8` (includes 3 bundled `.ckpt` ensemble members)

## Dependencies

- [generative_modeling.md](../../../generative_modeling.md) — `GenerativeModelWithEmbedding`, `LogitFormatter`
- `frame2seq` package:
  - `frame2seq.model.Frame2seq.frame2seq` checkpoint class
  - `frame2seq.utils.pdb2input.get_inference_inputs` for PDB/chain preprocessing
  - `frame2seq.utils.featurize.make_s_init`, `make_z_init`
  - `frame2seq.utils.rigid_utils.Rigid`

## Used By

- `tests/test_frame2seq.py` — construction, conditioning, embedding path, batching, temperature, upstream logit matching
- docs: `docs/models/frame2seq.md`

## Architecture

`Frame2seq` wraps an ensemble of Frame2seq checkpoints and exposes a 22-token proteingen interface:

- Frame2seq native vocab (`U=21`): 20 standard AAs + `X` unknown
- Protstar wrapper vocab (`T=22`): native 21 + `<mask>` input token

The wrapper maps `<mask>` probability mass into Frame2seq's unknown channel before running the network.

### Conditioning

Conditioning is required via:

```python
{"pdb_path": "path/to/structure.pdb", "chain_id": "A"}
```

`preprocess_observations` calls upstream `get_inference_inputs`, caches:

- `X_LA3` — `(L, 5, 3)` coordinates for `N, CA, C, CB, O`
- `seq_mask_L` — `(L,)` valid-residue mask
- `native_seq_L` — `(L,)` native sequence token IDs (Frame2seq indexing)

### Embedding path

For each ensemble member:

1. `make_s_init` and `make_z_init` build single/pair initial features from structure + masked sequence input.
2. IPA + transition + edge-update stack produces final single representation `s_BPD`.
3. `single_to_sequence` projects to 21-token logits.

Wrapper embedding is concatenated over ensemble members (`EMB_DIM = 128 * n_models`), while outputs are averaged logits over members.

## Checkpointing

- Constructor accepts `checkpoint_paths`; default is all packaged checkpoints under `frame2seq/trained_models/*.ckpt`
- `_save_args()` stores these checkpoint paths for `from_checkpoint()` reconstruction

## Gotchas

- **Conditioning is mandatory** — no unconditional mode.
- **Frame2seq upstream uses `X` as unknown/masked input** — wrapper adds explicit `<mask>` token and folds it into `X` internally.
- **Output constraints** — wrapper blocks both unknown (`X`) and mask columns in logits so probabilities normalize over the 20 canonical AAs.
- **Lightning checkpoint warning** — loading checkpoints may print an automatic upgrade notice (`v1.8 -> v2.x`); this is harmless.
