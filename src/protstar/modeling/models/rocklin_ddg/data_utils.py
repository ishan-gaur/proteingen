"""Data loading, featurization, and token conversion utilities.

Self-contained module (no imports from pdg) providing:
- Constants and alphabets for PMPNN and ESM tokenization
- Token conversion functions (single-sample and batched)
- PDB loading via biotite
- featurize() for ProteinMPNN-style graph dicts
- prepare_conditioning_inputs() for inference
- format_coords_to_esm3() for ESM3 atom37 format
- Sequence distance utilities
"""

from __future__ import print_function

import copy
import random
from pathlib import Path

import numpy as np
import torch

import biotite.structure as struc
from biotite.structure.io import pdb
from biotite.structure import get_chains
from biotite.structure.residues import get_residues
from biotite.sequence import ProteinSequence


# ---------------------------------------------------------------------------
# 1. Constants and alphabets
# ---------------------------------------------------------------------------

PMPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"  # vocab=21, last is mask token

ESM_ALPHABET = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N",
    "F", "Y", "M", "H", "W", "C",
    "X", "B", "U", "Z", "O", ".", "-", "|", "<mask>",
]

ESM3_ALPHABET_SIZE = 33


# ---------------------------------------------------------------------------
# 2. Token conversion functions
# ---------------------------------------------------------------------------

def esm_tok_to_pmpnn_tok(sample):
    """Strip cls/eos from ESM tokens and map to PMPNN alphabet.

    Args:
        sample: numpy array of ESM token indices (with cls at start, eos at end)

    Returns:
        numpy array of PMPNN token indices (no special tokens)
    """
    pmpnn_tok = np.asarray(
        [PMPNN_ALPHABET.index(ESM_ALPHABET[t]) for t in sample.astype(int)[1:-1]]
    )
    return pmpnn_tok


def pmpnn_tok_to_esm_tok(sample):
    """Convert PMPNN tokens to ESM tokens with cls/eos.

    Args:
        sample: numpy array of PMPNN token indices

    Returns:
        numpy array of ESM token indices with cls at start, eos at end
    """
    esm_tok = np.asarray(
        [0] + [ESM_ALPHABET.index(PMPNN_ALPHABET[t]) for t in sample.astype(int)] + [2]
    )
    return esm_tok


def esm_tokens_to_pmpnn_tokens_batch(esm_tokens: torch.Tensor) -> torch.Tensor:
    """Convert batch of ESM3 integer tokens (B, D) to PMPNN tokens (B, L).

    Strips cls/eos, maps ESM amino acid indices to PMPNN indices.
    """
    # Build lookup table: ESM index -> PMPNN index
    # Only valid for amino acid tokens (indices 4-23 in ESM = L,A,G,V,S,E,R,T,I,D,P,K,Q,N,F,Y,M,H,W,C)
    max_esm_idx = max(i for i, tok in enumerate(ESM_ALPHABET) if tok in PMPNN_ALPHABET)
    lookup = torch.zeros(max_esm_idx + 1, dtype=torch.long, device=esm_tokens.device)
    for esm_idx, tok in enumerate(ESM_ALPHABET):
        if tok in PMPNN_ALPHABET:
            lookup[esm_idx] = PMPNN_ALPHABET.index(tok)
    # Strip cls (first) and eos (last)
    inner = esm_tokens[:, 1:-1]
    return lookup[inner.long()]


def esm_ohe_to_pmpnn_ohe(esm_ohe: torch.Tensor) -> torch.Tensor:
    """Convert ESM3 one-hot (B, D, 33) to PMPNN one-hot (B, L, 21).

    Strips cls/eos positions, maps ESM alphabet columns to PMPNN alphabet columns.
    Preserves gradients for TAG.
    """
    # Strip cls and eos positions
    inner = esm_ohe[:, 1:-1, :]  # (B, L, 33)

    # Build column mapping: for each PMPNN index, which ESM index has that amino acid
    B, L, _ = inner.shape
    pmpnn_ohe = torch.zeros(B, L, len(PMPNN_ALPHABET), device=esm_ohe.device, dtype=esm_ohe.dtype)
    for pmpnn_idx, aa in enumerate(PMPNN_ALPHABET):
        if aa in ESM_ALPHABET:
            esm_idx = ESM_ALPHABET.index(aa)
            pmpnn_ohe[:, :, pmpnn_idx] = inner[:, :, esm_idx]

    return pmpnn_ohe


# ---------------------------------------------------------------------------
# 3. PDB loading function
# ---------------------------------------------------------------------------

def _get_atom_coords_residuewise(struct, atoms=None):
    """Extract per-residue coordinates for specified backbone atoms.

    Args:
        struct: biotite AtomArray
        atoms: list of atom names, default ["N", "CA", "C", "O"]

    Returns:
        numpy array of shape (n_residues, len(atoms), 3)
    """
    if atoms is None:
        atoms = ["N", "CA", "C", "O"]

    def filterfn(s, axis=None):
        filters = np.stack([s.atom_name == name for name in atoms], axis=1)
        sum_ = filters.sum(0)
        if not np.all(sum_ <= np.ones(filters.shape[1])):
            raise RuntimeError("structure has multiple atoms with same name")
        index = filters.argmax(0)
        coords = s[index].coord
        coords[sum_ == 0] = float("nan")
        return coords

    return struc.apply_residue_wise(struct, struct, filterfn)


def load_pdb_to_graph_dict(pdb_path):
    """Load a PDB file and return a ProteinMPNN-style graph dict.

    Returns dict with keys:
        seq_chain_A: amino acid sequence string
        coords_chain_A: dict with N_chain_A, CA_chain_A, C_chain_A, O_chain_A
                        (each numpy array of shape [L, 3])
        name: PDB name (stem of file)
        num_of_chains: 1
        seq: same as seq_chain_A
        masked_list: ["A"]
        visible_list: []
    """
    pdb_path = Path(pdb_path)

    # Load structure
    with open(pdb_path) as fin:
        pdbf = pdb.PDBFile.read(fin)
    structure = pdb.get_structure(pdbf, model=1)

    # Filter to first chain
    all_chains = get_chains(structure)
    if len(all_chains) == 0:
        raise ValueError("No chains found in the input file.")
    chain_id = all_chains[0]
    chain_filter = [a.chain_id == chain_id for a in structure]
    structure = structure[chain_filter]

    # Extract sequence
    residue_identities = get_residues(structure)[1]
    seq = "".join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities])

    # Extract backbone coordinates residue-wise
    bb_coords = _get_atom_coords_residuewise(structure, atoms=["N", "CA", "C", "O"])
    # bb_coords shape: (L, 4, 3)

    coords_dict = {
        "N_chain_A": bb_coords[:, 0],    # (L, 3)
        "CA_chain_A": bb_coords[:, 1],   # (L, 3)
        "C_chain_A": bb_coords[:, 2],    # (L, 3)
        "O_chain_A": bb_coords[:, 3],    # (L, 3)
    }

    graph = {
        "seq_chain_A": seq,
        "coords_chain_A": coords_dict,
        "name": pdb_path.stem,
        "num_of_chains": 1,
        "seq": seq,
        "masked_list": ["A"],
        "visible_list": [],
    }
    return graph


# ---------------------------------------------------------------------------
# 4. featurize() function
# ---------------------------------------------------------------------------

def featurize(batch, device, use_esm=False):
    """Convert batch of graph dicts to tensors for StabilityPMPNN.

    Args:
        batch: list of graph dicts (from load_pdb_to_graph_dict or similar)
        device: torch device string or torch.device
        use_esm: if True, use ESM alphabet (adds cls/eos tokens and pads accordingly)

    Returns:
        X: coordinates tensor (B, L_max, 4, 3)
        S: sequence token tensor (B, L_max)
        mask: validity mask (B, L_max)
        lengths: numpy array of sequence lengths
        chain_M: chain mask (B, L_max)
        residue_idx: residue indices (B, L_max)
        mask_self: self-interaction mask (B, L_max, L_max)
        chain_encoding_all: chain encoding (B, L_max)
    """
    if use_esm:
        alphabet = ESM_ALPHABET
    else:
        alphabet = PMPNN_ALPHABET

    B = len(batch)
    lengths = np.array(
        [len(b["seq"]) for b in batch], dtype=np.int32
    )  # sum of chain seq lengths

    if use_esm:
        # Add 2 for the start and end tokens
        L_max = max([len(b["seq"]) for b in batch]) + 2
    else:
        L_max = max([len(b["seq"]) for b in batch])

    X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100 * np.ones(
        [B, L_max], dtype=np.int32
    )  # residue idx with jumps across chains
    chain_M = np.zeros(
        [B, L_max], dtype=np.int32
    )  # 1.0 for the bits that need to be predicted, 0.0 for the bits that are given
    mask_self = np.ones(
        [B, L_max, L_max], dtype=np.int32
    )  # for interface loss calculation - 0.0 for self interaction, 1.0 for other
    chain_encoding_all = np.zeros(
        [B, L_max], dtype=np.int32
    )  # integer encoding for chains 0, 0, 0,...0, 1, 1,..., 1, 2, 2, 2...
    S = np.zeros([B, L_max], dtype=np.int32)  # sequence AAs integers
    init_alphabet = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    ]
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_letters = init_alphabet + extra_alphabet
    for i, b in enumerate(batch):
        masked_chains = b["masked_list"]
        visible_chains = b["visible_list"]
        all_chains = masked_chains + visible_chains
        visible_temp_dict = {}
        masked_temp_dict = {}
        for step, letter in enumerate(all_chains):
            chain_seq = b[f"seq_chain_{letter}"]
            if letter in visible_chains:
                visible_temp_dict[letter] = chain_seq
            elif letter in masked_chains:
                masked_temp_dict[letter] = chain_seq
        for km, vm in masked_temp_dict.items():
            for kv, vv in visible_temp_dict.items():
                if vm == vv:
                    if kv not in masked_chains:
                        masked_chains.append(kv)
                    if kv in visible_chains:
                        visible_chains.remove(kv)
        all_chains = masked_chains + visible_chains
        random.shuffle(all_chains)  # randomly shuffle chain order
        num_chains = b["num_of_chains"]
        mask_dict = {}
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        l0 = 0
        l1 = 0
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                chain_seq = b[f"seq_chain_{letter}"]
                chain_length = len(chain_seq)
                chain_coords = b[f"coords_chain_{letter}"]  # this is a dictionary
                chain_mask = np.zeros(chain_length)  # 0.0 for visible chains
                x_chain = np.stack(
                    [
                        chain_coords[c_key]
                        for c_key in [
                            f"N_chain_{letter}",
                            f"CA_chain_{letter}",
                            f"C_chain_{letter}",
                            f"O_chain_{letter}",
                        ]
                    ],
                    1,
                )  # [chain_length,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
            elif letter in masked_chains:
                chain_seq = b[f"seq_chain_{letter}"]
                chain_length = len(chain_seq)
                chain_coords = b[f"coords_chain_{letter}"]  # this is a dictionary
                chain_mask = np.ones(chain_length)  # 1.0 for masked chains
                x_chain = np.stack(
                    [
                        chain_coords[c_key]
                        for c_key in [
                            f"N_chain_{letter}",
                            f"CA_chain_{letter}",
                            f"C_chain_{letter}",
                            f"O_chain_{letter}",
                        ]
                    ],
                    1,
                )  # [chain_length,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
        x = np.concatenate(x_chain_list, 0)  # [L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(
            chain_mask_list, 0
        )  # [L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list, 0)

        l = len(all_sequence)
        if use_esm:
            padding_start_idx = 1
            padding_end_idx = L_max - l - 1
        else:
            padding_start_idx = 0
            padding_end_idx = L_max - l

        x_pad = np.pad(
            x,
            [[padding_start_idx, padding_end_idx], [0, 0], [0, 0]],
            "constant",
            constant_values=(np.nan,),
        )
        X[i, :, :, :] = x_pad

        m_pad = np.pad(
            m,
            [[padding_start_idx, padding_end_idx]],
            "constant",
            constant_values=(0.0,),
        )
        chain_M[i, :] = m_pad

        chain_encoding_pad = np.pad(
            chain_encoding,
            [[padding_start_idx, padding_end_idx]],
            "constant",
            constant_values=(0.0,),
        )
        chain_encoding_all[i, :] = chain_encoding_pad

        # Convert sequences to tokens
        if use_esm:
            indices = np.asarray(
                (
                    [alphabet.index("<cls>")]
                    + [alphabet.index(a) for a in all_sequence]
                    + [alphabet.index("<eos>")]
                ),
                dtype=np.int32,
            )
            # +2 for special tokens
            S[i, : l + 2] = indices
        else:
            indices = np.asarray(
                [alphabet.index(a) for a in all_sequence], dtype=np.int32
            )
            S[i, :l] = indices

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.0

    # Conversion
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    mask_self = torch.from_numpy(mask_self).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(
        dtype=torch.long, device=device
    )
    return X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all


# ---------------------------------------------------------------------------
# 5. prepare_conditioning_inputs()
# ---------------------------------------------------------------------------

def prepare_conditioning_inputs(pdb_path, batch_size, device="cuda"):
    """Load PDB and prepare conditioning inputs for StabilityPMPNN.

    Uses PMPNN alphabet (use_esm=False) since all predictor models use vocab=21.

    Args:
        pdb_path: path to PDB file
        batch_size: number of copies in the batch
        device: torch device

    Returns:
        dict with keys: X, S1, mask, chain_M, residue_idx, chain_encoding_all, wt_seq
    """
    graph = load_pdb_to_graph_dict(pdb_path)
    # Replicate graph for batch
    batch = [copy.deepcopy(graph) for _ in range(batch_size)]
    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(
        batch, device, use_esm=False
    )
    return dict(
        X=X,
        S1=S,
        mask=mask,
        chain_M=chain_M,
        residue_idx=residue_idx,
        chain_encoding_all=chain_encoding_all,
        wt_seq=graph["seq_chain_A"],
    )


# ---------------------------------------------------------------------------
# 6. format_coords_to_esm3()
# ---------------------------------------------------------------------------

def format_coords_to_esm3(coords, device="cuda"):
    """Convert (L, 4, 3) backbone coords to ESM3 atom37 format (L, 37, 3).

    Mapping: N->0, CA->1, C->2, O->4

    Args:
        coords: tensor or numpy array of shape (L, 4, 3)
        device: target device

    Returns:
        tensor of shape (L, 37, 3) with NaN for unmapped atom slots
    """
    if isinstance(coords, np.ndarray):
        coords = torch.from_numpy(coords)
    batch_size = coords.shape[0]
    coords_esm = torch.full(
        (batch_size, 37, 3), float("nan"), dtype=torch.float32, device=device
    )
    atom_mapping = {0: 0, 1: 1, 2: 2, 3: 4}  # N, CA, C, O
    coords = coords.to(dtype=torch.float32, device=device)
    for src_idx, tgt_idx in atom_mapping.items():
        coords_esm[:, tgt_idx, :] = coords[:, src_idx, :]
    return coords_esm


# ---------------------------------------------------------------------------
# 7. Sequence utilities
# ---------------------------------------------------------------------------

def hamming_distance(s1, s2):
    """Compute Hamming distance between two equal-length strings."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def compute_seq_id(seq1, seq2):
    """Compute sequence identity as 1 - (hamming / length)."""
    assert len(seq1) == len(seq2)
    return 1 - hamming_distance(seq1, seq2) / len(seq1)


def pairwise_hamming(seqs, mean=True):
    """Compute all pairwise Hamming distances among a list of sequences.

    Args:
        seqs: list of equal-length strings
        mean: if True return the mean distance, else return the array

    Returns:
        float (mean) or numpy array of pairwise distances
    """
    dists = []
    for i in range(len(seqs) - 1):
        for j in range(i + 1, len(seqs)):
            dists.append(hamming_distance(seqs[i], seqs[j]))
    dists = np.asarray(dists)
    return dists.mean() if mean else dists
