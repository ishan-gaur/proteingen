from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import biotite.structure.io.pdb as pdb_io
import biotite.structure.io.pdbx as pdbx
import torch
from esm.utils.structure.protein_chain import ProteinChain


def cif_to_atom37(cif_path: str | Path, chain_id: str = "A") -> torch.Tensor:
    """Convert a CIF structure file to atom37 coordinates ``(L, 37, 3)``.

    The CIF is converted through a temporary PDB because ``ProteinChain`` currently
    consumes PDB input.
    """
    cif_path = Path(cif_path)
    f = pdbx.CIFFile.read(str(cif_path))
    atoms = pdbx.get_structure(f, model=1, extra_fields=["b_factor", "occupancy"])

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        pdb_file = pdb_io.PDBFile()
        pdb_file.set_structure(atoms)
        pdb_file.write(tmp.name)
        tmp_path = Path(tmp.name)

    try:
        protein_chain = ProteinChain.from_pdb(str(tmp_path), chain_id=chain_id)
        return torch.from_numpy(protein_chain.atom37_positions).float()
    finally:
        tmp_path.unlink(missing_ok=True)


def af3_result_cif_path(
    result_output_dir: str,
    result_name: str,
    *,
    container_output_root: str = "/app/af_output",
    host_output_root: str | Path = "/data/af3_server_output",
) -> Path:
    """Map AF3 server output dir (container path) to host CIF path."""
    output_dir = str(result_output_dir)
    host_root = Path(host_output_root)
    if output_dir.startswith(container_output_root):
        suffix = output_dir[len(container_output_root) :].lstrip("/")
        mapped_dir = host_root / suffix
    else:
        mapped_dir = Path(output_dir)
    return mapped_dir / f"{result_name}_model.cif"


def fold_sequence_and_download_cif(
    client: Any,
    sequence: str,
    name: str,
    cif_path: str | Path,
):
    """Fold one sequence with AF3 server and download the resulting CIF file."""
    result = client.fold(sequence=sequence, name=name)
    cif_path = Path(cif_path)
    cif_path.parent.mkdir(parents=True, exist_ok=True)
    client.download_cif(result.job_id, cif_path)
    return result, cif_path


def fold_sequence_to_atom37(
    client: Any,
    sequence: str,
    name: str,
    *,
    container_output_root: str = "/app/af_output",
    host_output_root: str | Path = "/data/af3_server_output",
    chain_id: str = "A",
):
    """Fold one sequence with AF3 server and return ``(result, coords_atom37)``."""
    result = client.fold(sequence=sequence, name=name)
    cif_path = af3_result_cif_path(
        result_output_dir=result.output_dir,
        result_name=result.name,
        container_output_root=container_output_root,
        host_output_root=host_output_root,
    )
    coords = cif_to_atom37(cif_path, chain_id=chain_id)
    return result, coords
