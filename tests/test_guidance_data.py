"""Tests for proteingen.data module."""

import tempfile
from unittest.mock import MagicMock

import torch
import pytest

from proteingen.data import (
    ProteinDataset,
    uniform_mask_noise,
    no_noise,
    fully_unmasked,
    uniform_time,
    read_fasta,
    aligned_sequences_to_raw,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_mock_tokenizer():
    """Mock tokenizer with ESM-like behavior."""
    tok = MagicMock()
    tok.vocab = {
        "<cls>": 0,
        "<pad>": 1,
        "<eos>": 2,
        "<mask>": 32,
        "A": 5,
        "C": 6,
        "D": 7,
        "E": 8,
    }
    tok.all_special_ids = [0, 1, 2, 32]
    tok.pad_token_id = 1

    def tokenize_fn(sequences, padding=True, return_tensors="pt"):
        # Simple mock: CLS + char indices + EOS + PAD
        max_len = max(len(s) for s in sequences) + 2  # +CLS, +EOS
        batch = []
        for s in sequences:
            ids = [0] + [tok.vocab.get(c, 5) for c in s] + [2]
            ids += [1] * (max_len - len(ids))
            batch.append(ids)
        return {"input_ids": torch.tensor(batch, dtype=torch.long)}

    tok.side_effect = tokenize_fn
    tok.__call__ = tokenize_fn
    return tok


def _make_mock_model(tokenizer=None):
    """Mock model with tokenizer and preprocess_observations."""
    model = MagicMock()
    model.tokenizer = tokenizer or _make_mock_tokenizer()

    def preprocess_obs(obs):
        # Just stack tensors or return as-is
        result = {}
        for k, v in obs.items():
            if isinstance(v[0], torch.Tensor):
                result[k] = torch.stack(v)
            else:
                result[k] = v
        return result

    model.preprocess_observations = preprocess_obs
    return model


# ── Time samplers ────────────────────────────────────────────────────────────


class TestTimeSamplers:
    def test_fully_unmasked_returns_one(self):
        for _ in range(100):
            assert fully_unmasked() == 1.0

    def test_uniform_time_in_range(self):
        for _ in range(1000):
            t = uniform_time()
            assert 0.0 <= t <= 1.0


# ── Noise functions ──────────────────────────────────────────────────────────


class TestNoiseFunctions:
    def test_no_noise_identity(self):
        ids = torch.tensor([0, 5, 6, 7, 2])
        result = no_noise(ids, 0.5)
        assert torch.equal(result, ids)

    def test_uniform_mask_noise_at_t1_no_masking(self):
        tok = _make_mock_tokenizer()
        noise_fn = uniform_mask_noise(tok)
        ids = torch.tensor([0, 5, 6, 7, 8, 2])  # CLS A C D E EOS
        result = noise_fn(ids, 1.0)
        assert torch.equal(result, ids)

    def test_uniform_mask_noise_at_t0_masks_all_non_special(self):
        tok = _make_mock_tokenizer()
        noise_fn = uniform_mask_noise(tok)
        ids = torch.tensor([0, 5, 6, 7, 8, 2])  # CLS A C D E EOS
        result = noise_fn(ids, 0.0)
        # CLS and EOS should be preserved
        assert result[0].item() == 0  # CLS
        assert result[-1].item() == 2  # EOS
        # All middle positions should be masked
        assert (result[1:-1] == 32).all()

    def test_uniform_mask_noise_preserves_special_tokens(self):
        tok = _make_mock_tokenizer()
        noise_fn = uniform_mask_noise(tok)
        # Include PAD tokens
        ids = torch.tensor([0, 5, 6, 2, 1, 1])  # CLS A C EOS PAD PAD
        result = noise_fn(ids, 0.0)
        assert result[0].item() == 0  # CLS
        assert result[3].item() == 2  # EOS
        assert result[4].item() == 1  # PAD
        assert result[5].item() == 1  # PAD

    def test_uniform_mask_noise_does_not_modify_original(self):
        tok = _make_mock_tokenizer()
        noise_fn = uniform_mask_noise(tok)
        ids = torch.tensor([0, 5, 6, 7, 2])
        original = ids.clone()
        noise_fn(ids, 0.0)
        assert torch.equal(ids, original)


# ── ProteinDataset ───────────────────────────────────────────────────────────


class TestProteinDataset:
    def test_length(self):
        ds = ProteinDataset(sequences=["ACDE", "FGHI", "KLMN"])
        assert len(ds) == 3

    def test_getitem_sequence_only(self):
        ds = ProteinDataset(sequences=["ACDE", "FGHI"])
        item = ds[0]
        assert item["sequence"] == "ACDE"
        assert "observations" not in item
        assert "labels" not in item

    def test_getitem_with_labels(self):
        labels = torch.tensor([1.0, 2.0])
        ds = ProteinDataset(sequences=["ACDE", "FGHI"], labels=labels)
        item = ds[1]
        assert item["labels"] == 2.0

    def test_getitem_with_observations(self):
        obs = {"coords": [torch.randn(10, 3), torch.randn(10, 3)]}
        ds = ProteinDataset(sequences=["ACDE", "FGHI"], observations=obs)
        item = ds[0]
        assert "observations" in item
        assert "coords" in item["observations"]
        assert torch.equal(item["observations"]["coords"], obs["coords"][0])

    def test_mismatched_sequence_labels_raises(self):
        with pytest.raises(ValueError, match="Labels"):
            ProteinDataset(
                sequences=["ACDE", "FGHI", "KLMN"],
                labels=torch.tensor([1.0, 2.0]),
            )

    def test_mismatched_observation_length_raises(self):
        with pytest.raises(ValueError, match="Observation.*coords"):
            ProteinDataset(
                sequences=["ACDE", "FGHI"],
                observations={"coords": [1, 2, 3]},
            )


# ── Collator ─────────────────────────────────────────────────────────────────


class TestCollator:
    def test_collator_basic_tokenization(self):
        ds = ProteinDataset(sequences=["ACDE", "ACDE"])
        model = _make_mock_model()
        collate_fn = ds.collator(model, noise_fn=no_noise, time_sampler=fully_unmasked)

        batch = collate_fn([ds[0], ds[1]])
        assert "input_ids" in batch
        assert "target_ids" in batch
        assert batch["input_ids"].shape[0] == 2
        assert torch.equal(batch["input_ids"], batch["target_ids"])

    def test_collator_with_noise(self):
        ds = ProteinDataset(sequences=["ACDE", "ACDE"])
        tok = _make_mock_tokenizer()
        model = _make_mock_model(tok)
        noise_fn = uniform_mask_noise(tok)
        # t=0 → mask everything
        collate_fn = ds.collator(model, noise_fn=noise_fn, time_sampler=lambda: 0.0)

        batch = collate_fn([ds[0], ds[1]])
        # target_ids should be original
        # input_ids should have masked non-special positions
        assert not torch.equal(batch["input_ids"], batch["target_ids"])
        # CLS and EOS should match
        assert (batch["input_ids"][:, 0] == batch["target_ids"][:, 0]).all()
        assert (batch["input_ids"][:, -1] == batch["target_ids"][:, -1]).all()

    def test_collator_with_labels(self):
        labels = torch.tensor([1.0, 2.0])
        ds = ProteinDataset(sequences=["ACDE", "FGHI"], labels=labels)
        model = _make_mock_model()
        collate_fn = ds.collator(model, noise_fn=no_noise, time_sampler=fully_unmasked)

        batch = collate_fn([ds[0], ds[1]])
        assert batch["labels"] is not None
        assert torch.equal(batch["labels"], labels)

    def test_collator_no_labels_returns_none(self):
        ds = ProteinDataset(sequences=["ACDE", "FGHI"])
        model = _make_mock_model()
        collate_fn = ds.collator(model, noise_fn=no_noise, time_sampler=fully_unmasked)

        batch = collate_fn([ds[0], ds[1]])
        assert batch["labels"] is None

    def test_collator_with_observations(self):
        coords = [torch.randn(4, 3), torch.randn(4, 3)]
        ds = ProteinDataset(
            sequences=["ACDE", "FGHI"],
            observations={"coords": coords},
        )
        model = _make_mock_model()
        collate_fn = ds.collator(model, noise_fn=no_noise, time_sampler=fully_unmasked)

        batch = collate_fn([ds[0], ds[1]])
        assert "coords" in batch["observations"]
        assert batch["observations"]["coords"].shape == (2, 4, 3)

    def test_collator_rename_obs_keys(self):
        coords = [torch.randn(4, 3), torch.randn(4, 3)]
        ds = ProteinDataset(
            sequences=["ACDE", "FGHI"],
            observations={"coords": coords},
        )
        model = _make_mock_model()
        collate_fn = ds.collator(
            model,
            noise_fn=no_noise,
            time_sampler=fully_unmasked,
            rename_obs_keys={"coords_RAX": "coords"},  # model_kwarg: dataset_key
        )

        batch = collate_fn([ds[0], ds[1]])
        # Should be renamed to model's key
        assert "coords_RAX" in batch["observations"]
        assert "coords" not in batch["observations"]

    def test_collator_no_observations_empty_dict(self):
        ds = ProteinDataset(sequences=["ACDE"])
        model = _make_mock_model()
        collate_fn = ds.collator(model, noise_fn=no_noise, time_sampler=fully_unmasked)

        batch = collate_fn([ds[0]])
        assert batch["observations"] == {}


# ── FASTA utilities ──────────────────────────────────────────────────────────


class TestFasta:
    def test_read_fasta(self):
        content = ">seq1\nACDE\nFGHI\n>seq2\nKLMN\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(content)
            f.flush()
            entries = read_fasta(f.name)

        assert len(entries) == 2
        assert entries[0] == ("seq1", "ACDEFGHI")
        assert entries[1] == ("seq2", "KLMN")

    def test_aligned_sequences_to_raw(self):
        aligned = ["AC--DE", "A.CD.E", "ACDE--"]
        raw = aligned_sequences_to_raw(aligned)
        assert raw == ["ACDE", "ACDE", "ACDE"]
