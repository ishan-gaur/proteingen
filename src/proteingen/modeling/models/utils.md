# Model Utilities — Design Notes

Shared utilities for loading structures and preparing model inputs.

## Dependencies

- **External (Foundry/atomworks)**: `atomworks.io.parse`, `atomworks.ml.transforms.encoding.atom_array_to_encoding`, `biotite.structure`
- **Used by**: all structure-conditioned models (PMPNN, ESM3, future models)

## Structure Loading — Two-Layer API

### Layer 1: `PDBStructure` (user-facing)

```python
from proteingen.models.utils import load_pdb, PDBStructure

structure = load_pdb("1YCR.pdb")
# structure.atom_array   — biotite AtomArray (preserved for model-specific encoding)
# structure.chain_ids    — (L,) per-residue chain ID strings, e.g. ['A','A','B']
# structure.sequence     — full sequence across all chains
```

`PDBStructure` is **encoding-agnostic** — it holds the raw biotite `AtomArray`, not encoded coordinates. This is the object users pass around. Model-specific code re-encodes the atom_array with the appropriate atom layout.

### Layer 2: `atom_array_to_encoding()` (model-implementer-facing)

Inside each model's `condition_from_structure()` (or equivalent), call atomworks' encoding function with the **model's specific encoding constant**:

```python
from atomworks.ml.transforms.encoding import atom_array_to_encoding  # [foundry]

encoded = atom_array_to_encoding(
    structure.atom_array,
    encoding=MODEL_SPECIFIC_ENCODING,  # see table below
    default_coord=0.0,                 # model-dependent (0.0 or NaN)
    occupancy_threshold=0.5,           # model-dependent (0.0 or 0.5)
)
```

The `encoded` dict contains:

| Field | Shape | Description |
|---|---|---|
| `xyz` | `(L, N_atoms, 3)` | Coordinates. `N_atoms` depends on encoding. Missing atoms get `default_coord`. |
| `mask` | `(L, N_atoms)` | True where atom exists and occupancy > threshold |
| `seq` | `(L,)` | Integer token index in the encoding's vocabulary |
| `chain_id` | `(L,)` | Integer chain label |
| `chain_id_to_int` | `dict` | String→int mapping (e.g. `{'A': 0, 'B': 1}`) |
| `chain_entity` | `(L,)` | Integer entity ID (distinguishes unique sequences) |
| `molecule_iid` | `(L,)` | Integer molecule instance ID |
| `chain_iid` | `(L,)` | Integer chain instance ID |
| `transformation_id` | `(L,)` | Integer assembly transformation ID |
| `token_is_atom` | `(L,)` | True for atomized tokens (ligands), False for standard residues |

### Available Encodings

| Constant | Source | Tokens × Atoms | Used by |
|---|---|---|---|
| `MPNN_TOKEN_ENCODING` | `mpnn.transforms.feature_aggregation.mpnn` | 21 × 37 | ProteinMPNN |
| `UNIFIED_ATOM37_ENCODING` | `atomworks.ml.encoding_definitions` | 33 × 37 | ESM3 |
| `AF2_ATOM37_ENCODING` | `atomworks.ml.encoding_definitions` | 21 × 37 | AlphaFold2-style models |
| `AF2_ATOM14_ENCODING` | `atomworks.ml.encoding_definitions` | 21 × 14 | Compact AF2 representation |

**Critical**: different encodings place atoms at different slot indices. E.g. MPNN puts O at slot 3, CB at slot 4; UNIFIED puts CB at slot 3, O at slot 4. Using the wrong encoding produces wrong coordinates and non-trivial logit differences.

## Pattern for Adding Structure Conditioning to a New Model

When integrating a new structure-conditioned model, follow the PMPNN pattern:

1. **Accept `PDBStructure` in `preprocess_observations`** — users pass `{"structure": pdb_structure, ...}` to `set_condition_()` / `conditioned_on()`. The model's `preprocess_observations` handles encoding internally.
2. **Encode inside the model** — call `atom_array_to_encoding(structure.atom_array, YOUR_ENCODING, ...)` to get coords/mask. Derive chain labels, residue indices from the encoded dict. Accept user-facing options like `design_chains` from the observations dict and map them to internal fields (e.g. `residue_mask`).
3. **Keep internal conditioning TypedDicts internal** (prefixed with `_`) — users interact with `PDBStructure` + options, not raw tensor dicts. Tests should also use `PDBStructure` (see `_make_structure()` in `test_protein_mpnn.py` for building synthetic structures with biotite).

See `proteingen.models.mpnn.protein_mpnn` (`preprocess_observations` and `_encode_structure`) for the reference implementation.

## Legacy

- `pdb_to_atom37_and_seq()` — older single-chain utility using `UNIFIED_ATOM37_ENCODING`. Still used by ESM3 examples. Returns `(coords_RAX, sequence)` without chain info. Prefer `load_pdb()` for new code.
