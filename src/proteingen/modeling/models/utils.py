"""Backward-compatible import path for structure utilities.

Moved to ``proteingen.data.structure``.
"""

from proteingen.data.structure import PDBStructure, load_pdb, pdb_to_atom37_and_seq

__all__ = ["PDBStructure", "load_pdb", "pdb_to_atom37_and_seq"]
