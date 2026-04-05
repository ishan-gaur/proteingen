"""Tests for ProteinMPNN wrapper (TransitionModelWithEmbedding)."""

import torch
import torch.utils.checkpoint
import numpy as np
import biotite.structure as bts
import pytest
import urllib.request
from pathlib import Path
from proteingen.models.mpnn import ProteinMPNN
from proteingen.models.utils import load_pdb, PDBStructure


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_structure(L: int = 20, n_chains: int = 1):
    """Create a synthetic PDBStructure with random backbone coordinates."""
    atoms_per_res = 4  # N, CA, C, O
    n_atoms = L * atoms_per_res
    residues_per_chain = L // n_chains

    atom_array = bts.AtomArray(n_atoms)
    atom_array.coord = np.random.randn(n_atoms, 3).astype(np.float32)
    atom_array.res_id = np.repeat(np.arange(1, L + 1), atoms_per_res)
    atom_array.res_name = np.array(["ALA"] * n_atoms)
    atom_array.atom_name = np.tile(["N", "CA", "C", "O"], L)
    atom_array.element = np.tile(["N", "C", "C", "O"], L)
    atom_array.set_annotation(
        "occupancy", np.ones(n_atoms, dtype=np.float32)
    )
    atom_array.set_annotation(
        "atomize", np.zeros(n_atoms, dtype=bool)
    )

    # Assign chain IDs
    chain_ids_per_res = []
    for c in range(n_chains):
        chain_ids_per_res.extend([chr(ord("A") + c)] * residues_per_chain)
    # Handle remainder
    chain_ids_per_res.extend([chr(ord("A") + n_chains - 1)] * (L - len(chain_ids_per_res)))
    atom_array.chain_id = np.repeat(chain_ids_per_res, atoms_per_res)

    residue_starts = bts.get_residue_starts(atom_array)
    chain_ids = atom_array.chain_id[residue_starts]
    sequence = "A" * L

    return {
        "structure": PDBStructure(
            atom_array=atom_array,
            chain_ids=chain_ids,
            sequence=sequence,
        )
    }


@pytest.fixture
def model():
    m = ProteinMPNN("proteinmpnn")
    return m


@pytest.fixture
def conditioned_model(model):
    """Model with structure conditioning set."""
    structure = _make_structure(20)
    model.set_condition_(structure)
    return model, structure


# ── Construction & basic properties ───────────────────────────────────────


def test_construction(model):
    assert hasattr(model, "model")
    assert hasattr(model, "tokenizer")
    assert hasattr(model, "logit_formatter")
    assert model.OUTPUT_DIM == 22
    assert model.EMB_DIM == 128
    assert model.tokenizer.vocab_size == 22
    assert model.tokenizer.mask_token_id == 21


def test_tokenizer(model):
    tok = model.tokenizer
    assert tok.cls_token_id is None
    assert tok.eos_token_id is None
    assert tok.pad_token_id is None

    encoded = tok.encode("ACDEF")
    assert len(encoded) == 5
    decoded = tok.decode(encoded)
    assert decoded == "ACDEF"

    # Mask token
    assert "<mask>" in tok.vocab
    assert tok.vocab["<mask>"] == 21


def test_save_args(model):
    args = model._save_args()
    assert args == {"checkpoint": "proteinmpnn"}


# ── Conditioning ──────────────────────────────────────────────────────────


def test_set_condition(model):
    structure = _make_structure(15)
    model.set_condition_(structure)
    assert model.observations is not None
    assert "h_V" in model.observations
    assert "h_E" in model.observations
    assert "E_idx" in model.observations
    assert "residue_mask" in model.observations

    # Cached features should be unbatched (squeezed)
    assert model.observations["h_V"].shape == (15, 128)
    assert model.observations["residue_mask"].shape == (15,)


def test_conditioned_on_context_manager(model):
    structure = _make_structure(10)
    assert model.observations is None
    with model.conditioned_on(structure):
        assert model.observations is not None
    assert model.observations is None


def test_requires_conditioning(model):
    """Forward/embed should fail without conditioning."""
    tokens = model.tokenizer("AAAAA")["input_ids"]
    with pytest.raises(ValueError, match="structure conditioning"):
        model.get_log_probs(tokens)


# ── Forward pass & log probs ──────────────────────────────────────────────


def test_forward_shape(conditioned_model):
    model, structure = conditioned_model
    L = 20
    tokens = model.tokenizer("A" * L)["input_ids"]

    with torch.no_grad():
        raw = model.forward(tokens)
    assert raw.shape == (1, L, 22)


def test_get_log_probs(conditioned_model):
    model, _ = conditioned_model
    L = 20
    tokens = model.tokenizer("A" * L)["input_ids"]

    with torch.no_grad():
        log_probs = model.get_log_probs(tokens)

    assert log_probs.shape == (1, L, 22)
    assert log_probs.dtype == torch.float32
    assert not torch.any(torch.isnan(log_probs))
    assert torch.all(log_probs <= 0.0)

    # Probabilities sum to 1
    probs = torch.exp(log_probs)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_log_probs_with_mask_tokens(conditioned_model):
    """Mask tokens should produce a distribution over AAs, not a delta."""
    model, _ = conditioned_model
    L = 20
    tokens = model.tokenizer("A" * L)["input_ids"]
    tokens[0, 5:10] = model.tokenizer.mask_token_id

    with torch.no_grad():
        log_probs = model.get_log_probs(tokens)

    # Mask positions: distribution over 20 standard AAs
    mask_probs = log_probs[0, 5:10].exp()
    assert torch.allclose(mask_probs.sum(dim=-1), torch.ones(5), atol=1e-4)
    # Should have non-trivial distribution (not all mass on one token)
    assert (mask_probs[:, :20] > 1e-6).sum() > 5  # multiple AAs have probability

    # Non-mask positions: still get a real conditional distribution
    # (conditional_minus_self means each position predicts based on all
    # others, so non-mask positions have meaningful log-probs, not delta)
    non_mask_probs = log_probs[0, 0:5].exp()
    assert torch.allclose(non_mask_probs.sum(dim=-1), torch.ones(5), atol=1e-4)
    # Not all mass on input token — it's a real distribution
    assert (non_mask_probs[:, :20] > 1e-6).sum() > 5

    # UNK (20) and mask (21) should always be -inf
    assert torch.all(log_probs[..., 20] == float("-inf"))
    assert torch.all(log_probs[..., 21] == float("-inf"))


def test_temperature_scaling(conditioned_model):
    model, _ = conditioned_model
    L = 20
    tokens = model.tokenizer("A" * L)["input_ids"]
    tokens[0, :] = model.tokenizer.mask_token_id  # all masked

    with torch.no_grad():
        log_probs_t1 = model.get_log_probs(tokens)
        with model.with_temp(0.1):
            log_probs_low = model.get_log_probs(tokens)
        with model.with_temp(10.0):
            log_probs_high = model.get_log_probs(tokens)

    # Lower temp → sharper distribution (higher max prob)
    max_t1 = log_probs_t1.exp().max(dim=-1).values.mean()
    max_low = log_probs_low.exp().max(dim=-1).values.mean()
    max_high = log_probs_high.exp().max(dim=-1).values.mean()
    assert max_low > max_t1
    assert max_t1 > max_high


# ── Embedding path ────────────────────────────────────────────────────────


def test_embed_shape(conditioned_model):
    model, _ = conditioned_model
    L = 20
    tokens = model.tokenizer("A" * L)["input_ids"]

    with torch.no_grad():
        emb = model.embed(tokens)
    assert emb.shape == (1, L, 128)


def test_embed_forward_consistency(conditioned_model):
    """embedding_to_outputs(embed(x)) should match forward(x)."""
    model, _ = conditioned_model
    L = 20
    tokens = model.tokenizer("A" * L)["input_ids"]

    with torch.no_grad():
        emb = model.embed(tokens)
        out_emb = model.embedding_to_outputs(emb)
        out_fwd = model.forward(tokens)

    # Compare finite values (mask column is -inf in both)
    finite = torch.isfinite(out_emb) & torch.isfinite(out_fwd)
    assert torch.allclose(out_emb[finite], out_fwd[finite], atol=1e-5)
    # Mask column is -inf in both
    assert (out_emb[:, :, 21] == float("-inf")).all()
    assert (out_fwd[:, :, 21] == float("-inf")).all()


def test_gradient_flow(conditioned_model):
    """Gradients should flow back through embed to the OHE input."""
    model, _ = conditioned_model
    L = 20
    tokens = model.tokenizer("A" * L)["input_ids"]

    ohe = torch.nn.functional.one_hot(
        tokens, num_classes=model.tokenizer.vocab_size
    ).float()
    ohe.requires_grad_(True)

    emb = model.differentiable_embedding(ohe)
    loss = emb.sum()
    loss.backward()

    assert ohe.grad is not None
    assert ohe.grad.norm().item() > 0


# ── Batching ──────────────────────────────────────────────────────────────


def test_batching_consistency(conditioned_model):
    """Single sequence should match first element of batched output."""
    model, _ = conditioned_model
    L = 20

    tokens_single = model.tokenizer("A" * L)["input_ids"]
    tokens_batch = model.tokenizer(["A" * L, "G" * L])["input_ids"]

    with torch.no_grad():
        lp_single = model.get_log_probs(tokens_single)
        lp_batch = model.get_log_probs(tokens_batch)

    assert torch.allclose(lp_single[0], lp_batch[0], atol=1e-5)


# ── Different structures produce different outputs ────────────────────────


def test_different_structures(model):
    """Different structures should produce different log probs."""
    L = 15
    struct1 = _make_structure(L)
    struct2 = _make_structure(L)  # random coords → different structure

    tokens = model.tokenizer("A" * L)["input_ids"]
    tokens[0, :] = model.tokenizer.mask_token_id

    with torch.no_grad():
        with model.conditioned_on(struct1):
            lp1 = model.get_log_probs(tokens)
        with model.conditioned_on(struct2):
            lp2 = model.get_log_probs(tokens)

    assert not torch.allclose(lp1, lp2, atol=1e-4)


# ── Multi-chain support ──────────────────────────────────────────────────


def test_multi_chain(model):
    """Model should handle multi-chain structures."""
    L = 20
    structure = _make_structure(L, n_chains=2)

    model.set_condition_(structure)
    tokens = model.tokenizer("A" * L)["input_ids"]

    with torch.no_grad():
        log_probs = model.get_log_probs(tokens)
    assert log_probs.shape == (1, L, 22)
    assert not torch.any(torch.isnan(log_probs))


def test_design_chains(model):
    """design_chains should restrict which positions are designable."""
    L = 20
    structure = _make_structure(L, n_chains=2)

    # All chains designable
    model.set_condition_(structure)
    mask_all = model.observations["residue_mask"]
    assert mask_all.all()

    # Only chain A designable
    structure_a = {**structure, "design_chains": ["A"]}
    model.set_condition_(structure_a)
    mask_a = model.observations["residue_mask"]
    assert mask_a.sum() == 10  # first 10 residues
    assert mask_a[:10].all()
    assert not mask_a[10:].any()


# ── Soluble MPNN variant ──────────────────────────────────────────────────


def test_soluble_mpnn():
    """SolubleMPNN checkpoint loads and runs."""
    from pathlib import Path
    from foundry.inference_engines.checkpoint_registry import REGISTERED_CHECKPOINTS

    ckpt_path = Path(str(REGISTERED_CHECKPOINTS["solublempnn"].get_default_path()))
    if not ckpt_path.exists():
        pytest.skip(f"SolubleMPNN checkpoint not found at {ckpt_path}")

    model = ProteinMPNN("solublempnn")
    assert model.EMB_DIM == 128

    L = 10
    structure = _make_structure(L)
    model.set_condition_(structure)

    tokens = model.tokenizer("A" * L)["input_ids"]
    with torch.no_grad():
        log_probs = model.get_log_probs(tokens)
    assert log_probs.shape == (1, L, 22)
    assert not torch.any(torch.isnan(log_probs))


# ── LoRA ──────────────────────────────────────────────────────────────────


def test_lora(conditioned_model):
    """LoRA should make some params trainable and forward should still work."""
    model, _ = conditioned_model
    assert not model.has_lora

    model.apply_lora(r=4, lora_alpha=8)
    assert model.has_lora

    L = 20
    tokens = model.tokenizer("A" * L)["input_ids"]
    with torch.no_grad():
        log_probs = model.get_log_probs(tokens)
    assert log_probs.shape == (1, L, 22)
    assert not torch.any(torch.isnan(log_probs))

    # Some params should be trainable
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert trainable > 0


# ── Foundry reference comparison on real multimer PDB ─────────────────────


PDB_1YCR = Path("/tmp/test_1YCR.pdb")


@pytest.fixture(scope="module")
def real_multimer_data():
    """Download 1YCR (p53/MDM2, 2 chains, 98 residues) and process via Foundry."""
    if not PDB_1YCR.exists():
        urllib.request.urlretrieve(
            "https://files.rcsb.org/download/1YCR.pdb", str(PDB_1YCR)
        )

    from mpnn.model.mpnn import ProteinMPNN as _ProteinMPNN
    from mpnn.utils.weights import load_legacy_weights

    # Our API
    structure = load_pdb(str(PDB_1YCR))

    # Foundry reference logits
    foundry_model = _ProteinMPNN()
    load_legacy_weights(foundry_model, "/data/ishan/foundry/proteinmpnn_v_48_020.pt")
    foundry_model.eval()

    # Need Foundry pipeline for reference logits (ground truth)
    from mpnn.pipelines.mpnn import build_mpnn_transform_pipeline
    from mpnn.collate.feature_collator import FeatureCollator
    from atomworks.io import parse

    parsed = parse(str(PDB_1YCR))
    atom_array = parsed["assemblies"]["1"][0]
    pipeline = build_mpnn_transform_pipeline(
        model_type="protein_mpnn",
        is_inference=True,
        minimal_return=False,
    )
    pipeline_output = pipeline(
        {
            "atom_array": atom_array.copy(),
            "structure_noise": 0.0,
            "decode_type": "teacher_forcing",
            "causality_pattern": "conditional_minus_self",
            "initialize_sequence_embedding_with_ground_truth": True,
            "atomize_side_chains": False,
            "repeat_sample_num": None,
            "features_to_return": None,
        }
    )
    network_input = FeatureCollator()([pipeline_output])

    with torch.no_grad():
        foundry_output = foundry_model(network_input)

    return {
        "structure": structure,
        "network_input": network_input,
        "foundry_logits": foundry_output["decoder_features"]["logits"],
    }


def test_matches_foundry_on_real_multimer(real_multimer_data):
    """Our wrapper produces identical logits to Foundry on 1YCR (2-chain PDB)."""
    structure = real_multimer_data["structure"]
    ni = real_multimer_data["network_input"]
    foundry_logits = real_multimer_data["foundry_logits"]

    S = ni["input_features"]["S"]
    chain_labels = ni["input_features"]["chain_labels"]
    assert chain_labels.unique().numel() == 2, "Expected 2 chains in 1YCR"

    model = ProteinMPNN("proteinmpnn")
    model.set_condition_({"structure": structure})

    with torch.no_grad():
        our_logits_22 = model.embedding_to_outputs(model.embed(S))
        our_logits_21 = our_logits_22[:, :, :21]

    max_diff = (foundry_logits - our_logits_21).abs().max().item()
    assert max_diff == 0.0, f"Logits differ by {max_diff} on real multimer 1YCR"
    assert foundry_logits.argmax(-1).eq(our_logits_21.argmax(-1)).all()
