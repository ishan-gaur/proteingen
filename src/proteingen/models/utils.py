from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import biotite.structure as bts
import atomworks.io as aio
from atomworks.ml.transforms.encoding import atom_array_to_encoding
from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING


@dataclass
class PDBStructure:
    """Parsed PDB with per-residue chain and sequence info.

    The atom_array is kept so model-specific code can re-encode
    coordinates with the appropriate atom layout (e.g. MPNN vs ESM).
    """

    atom_array: bts.AtomArray  # full biotite atom array
    chain_ids: np.ndarray  # (L,) per-residue chain ID strings, e.g. ['A','A','B']
    sequence: str  # full sequence across all chains


def load_pdb(pdb_path: Path | str) -> PDBStructure:
    """Parse a PDB file into a PDBStructure.

    Assumes a single biological assembly. Handles multi-chain structures.
    """
    parsed = aio.parse(str(pdb_path))
    atom_array = parsed["assemblies"]["1"][0]

    residue_starts = bts.get_residue_starts(atom_array)
    chain_ids = atom_array.chain_id[residue_starts]  # (L,) str
    sequence = bts.to_sequence(atom_array)[0][0]

    return PDBStructure(
        atom_array=atom_array,
        chain_ids=chain_ids,
        sequence=sequence,
    )


def pdb_to_atom37_and_seq(pdb_path: Path | str, backbone_only: bool = False):
    # TODO[pi] instead of backbone_only
    # Assumes the structure file has only one assembly and one chain TODO[pi] modify the interface to be appropriately general e.g. for use with multi-chain inverse-folding conditioning, etc.
    structure_data_dict = aio.parse(pdb_path)
    chain_atom_array = structure_data_dict["assemblies"]["1"][0]
    wt_seq = bts.to_sequence(chain_atom_array)[0][0]
    coords_atom37_RAX = atom_array_to_encoding(
        chain_atom_array, UNIFIED_ATOM37_ENCODING
    )["xyz"]  # Residue, atom, position
    if backbone_only:
        coords_bb_RAX = coords_atom37_RAX
        coords_bb_RAX[:, 3, :] = float("nan")
        coords_bb_RAX[:, 5:, :] = float("nan")
        return coords_bb_RAX, wt_seq
    else:
        return coords_atom37_RAX, wt_seq
