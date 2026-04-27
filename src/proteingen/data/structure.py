from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import atomworks.io as aio
import biotite.structure as bts
import numpy as np
from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING
from atomworks.ml.transforms.encoding import atom_array_to_encoding


@dataclass
class PDBStructure:
    """Parsed PDB with per-residue chain and sequence info.

    The atom_array is kept so model-specific code can re-encode
    coordinates with the appropriate atom layout (e.g. MPNN vs ESM).
    """

    atom_array: bts.AtomArray  # full biotite atom array
    chain_ids: np.ndarray  # (L,) per-residue chain ID strings, e.g. ['A','A','B']
    sequence: str  # full sequence across all chains


def _repo_root_from_file() -> Path:
    """Best-effort repo root detection using this module's __file__."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
    return here.parents[3]


def _default_pdb_cache_dir() -> Path:
    return _repo_root_from_file() / "data" / "pdbs"


def _infer_pdb_id(pdb_path: Path | str) -> str:
    path = Path(pdb_path)
    token = path.stem if path.suffix else path.name
    token = token.strip().upper()
    if len(token) != 4 or not token.isalnum():
        raise FileNotFoundError(
            f"PDB file not found at {path}. Could not infer a valid 4-char PDB id from '{token}'."
        )
    return token


def _download_pdb_to_path(pdb_id: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        urlretrieve(url, destination)
    except HTTPError as exc:
        raise FileNotFoundError(
            f"Failed to download PDB {pdb_id} from RCSB ({url}): {exc}"
        ) from exc
    except URLError as exc:
        raise FileNotFoundError(
            f"Network error while downloading PDB {pdb_id} from RCSB ({url}): {exc}"
        ) from exc
    return destination


def _resolve_pdb_path(
    pdb_path: Path | str, cache_dir: Path | str | None = None
) -> Path:
    """Resolve a local PDB path, downloading from RCSB if missing.

    If ``pdb_path`` exists, returns it unchanged.
    If not, infers a 4-character PDB id from filename/stem and downloads to:
    ``<repo_root>/data/pdbs/<PDB_ID>.pdb`` by default.
    """
    path = Path(pdb_path)
    if path.exists():
        return path

    pdb_id = _infer_pdb_id(path)
    cache_root = Path(cache_dir) if cache_dir is not None else _default_pdb_cache_dir()
    cached_path = cache_root / f"{pdb_id}.pdb"

    if not cached_path.exists():
        _download_pdb_to_path(pdb_id, cached_path)
    return cached_path


def load_pdb(
    pdb_path: Path | str,
    cache_dir: Path | str | None = None,
) -> PDBStructure:
    """Parse a PDB file into a PDBStructure.

    If the file is not present locally, tries to infer a PDB id from
    ``pdb_path`` and downloads it from RCSB into ``data/pdbs`` at repo root.

    Assumes a single biological assembly. Handles multi-chain structures.
    """
    resolved_path = _resolve_pdb_path(pdb_path, cache_dir=cache_dir)
    parsed = aio.parse(str(resolved_path))
    atom_array = parsed["assemblies"]["1"][0]

    residue_starts = bts.get_residue_starts(atom_array)
    chain_ids = atom_array.chain_id[residue_starts]  # (L,) str
    sequence = bts.to_sequence(atom_array)[0][0]

    return PDBStructure(
        atom_array=atom_array,
        chain_ids=chain_ids,
        sequence=sequence,
    )


def pdb_to_atom37_and_seq(
    pdb_path: Path | str,
    backbone_only: bool = False,
    cache_dir: Path | str | None = None,
):
    # TODO[pi] instead of backbone_only
    # Assumes the structure file has only one assembly and one chain TODO[pi] modify the interface to be appropriately general e.g. for use with multi-chain inverse-folding conditioning, etc.
    resolved_path = _resolve_pdb_path(pdb_path, cache_dir=cache_dir)
    structure_data_dict = aio.parse(str(resolved_path))
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
