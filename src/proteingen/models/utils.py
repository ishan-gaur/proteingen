from pathlib import Path
import biotite.structure as bts
import atomworks.io as aio
from atomworks.ml.transforms.encoding import atom_array_to_encoding
from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING


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
