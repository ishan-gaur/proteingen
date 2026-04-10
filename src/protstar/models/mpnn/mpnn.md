# ProteinMPNN — Design Notes

Structure-conditioned autoregressive sequence design model from the Foundry (`rc-foundry[all]`) package.

## Dependencies

- **Core abstractions**: `GenerativeModelWithEmbedding`, `MaskedModelLogitFormatter`, `MPNNTokenizer` from `protstar.generative_modeling`
- **Structure loading**: `load_pdb`, `PDBStructure` from `protstar.models.utils` — see [../utils.md](../utils.md) for the two-layer API
- **External**: `mpnn` package (via `rc-foundry[all]`) — provides `ProteinMPNN` model, `load_legacy_weights`, `cat_neighbors_nodes`, `gather_nodes`
- **Checkpoint registry**: `foundry.inference_engines.checkpoint_registry.REGISTERED_CHECKPOINTS`

## Used By

- `tests/test_protein_mpnn.py` — 20 tests (19 pass, 1 skipped for missing checkpoint)
- All tests use `PDBStructure`-based conditioning (synthetic via `_make_structure()` or real via `load_pdb`)
- Includes a Foundry reference test on PDB 1YCR (p53/MDM2, 2 chains, 98 residues) — produces bitwise-identical logits (0.0 max diff, 100% argmax match)
- `test_design_chains` verifies `design_chains` restricts `residue_mask` to specified chains
- Sampling, TAG guidance, LinearProbe via `GenerativeModelWithEmbedding` interface

## Structure Conditioning

PMPNN is the reference implementation for the structure conditioning pattern described in [../utils.md](../utils.md).

Users pass a `PDBStructure` directly — encoding happens inside `preprocess_observations`:

```python
model.conditioned_on({"structure": load_pdb("1YCR.pdb"), "design_chains": ["B"]})
```

- **`preprocess_observations`** accepts `{"structure": PDBStructure, "design_chains": [...]}`
- **`_encode_structure`** (static, internal) — converts `PDBStructure` → `ProteinMPNNCondition` using `MPNN_TOKEN_ENCODING` with `default_coord=0.0`, `occupancy_threshold=0.5`. 0.0 logit diff vs Foundry on 1YCR multimer.
- **`design_chains`** — list of chain IDs to design (e.g. `["B"]`). Sets `residue_mask` to True only for those chains. If None, all chains are designable.

## Architecture

ProteinMPNN is a graph neural network for structure-conditioned protein sequence design:

1. **Graph featurization** (`ProteinFeatures`): backbone coordinates → edge features (RBF distances between backbone + virtual atoms) + neighbor graph (K-nearest neighbors, K=48)
2. **Encoder** (3 `EncLayer`): message passing on the structure graph → node features `h_V` (B, L, 128) and edge features `h_E` (B, L, K, 128)
3. **Decoder** (3 `DecLayer`): autoregressive sequence prediction conditioned on encoder output. Uses causal masking to control information flow between positions.
4. **Output head** (`W_out`): linear projection from hidden dim (128) → vocab size (21)

### Wrapper design

The protstar wrapper splits the model into two cached stages:

- **`preprocess_observations`** (called once via `set_condition_()`): runs graph featurization + encoder. Caches `h_V`, `h_E`, `E_idx`, `residue_mask`. Uses `S=0` (ALA) as placeholder for backbone atom indexing — safe because backbone atoms N/CA/C/O are at positions 0-3 for all amino acids.
- **`_run_decoder`** (called every forward): runs decoder with cached encoder features and the current sequence. Uses **conditional_minus_self** causality: each position sees all other positions' sequence embeddings but not its own. This gives a pseudo-likelihood at every position.

### Logit formatting

MPNN outputs 21-dim logits (20 AAs + UNK). The wrapper:
1. Pads to 22-dim by appending -inf at index 21 (mask token column)
2. Uses a simple `_BlockUNKLogitFormatter` that sets UNK (index 20) to -inf

Unlike `MaskedModelLogitFormatter`, there is no delta-forcing at non-mask positions. With conditional-minus-self decoding, every position already gets a meaningful conditional distribution P(residue_i | structure, all other residues). Forcing delta would destroy this signal and make sequence scoring useless — `get_log_probs` would return 0 at every non-mask position.

## Checkpointing

- `_save_args()` returns `{"checkpoint": "<name>"}` where name is a Foundry registry key
- Available: `"proteinmpnn"` (standard), `"solublempnn"` (soluble-only training)
- Weights loaded via `load_legacy_weights` (handles legacy → new parameter name mapping)

## Gotchas

- **`torch.utils.checkpoint` must be imported** before `mpnn.model.mpnn` — the MPNN encoder uses `torch.utils.checkpoint.checkpoint()` which requires the submodule to be loaded
- **`atomworks` env var warnings** — importing from `atomworks` prints CCD_MIRROR_PATH/PDB_MIRROR_PATH warnings; harmless
- **Backbone atom positions are token-independent** — N/CA/C/O at indices 0-3 for all amino acids, so using S=0 (ALA) for graph featurization is safe
- **`structure_noise=0.0`** in `preprocess_observations` — no Gaussian noise added to coordinates during inference (noise is a training augmentation)
- **All positions get logits** — unlike autoregressive inference where positions are decoded one-at-a-time, `conditional_minus_self` teacher forcing gives logits at all positions simultaneously
- **Small model (1.7M params)** — fits easily on CPU, no OOM concerns even for long sequences
- **Graph featurization mutates `input_features` dict** — adds keys like `X_pre_noise`, `X_backbone`, etc. in-place. The wrapper uses a fresh dict each time.
