"""Tests for guidance.data module."""

import torch
import numpy as np
import pytest

from proteingen import GuidanceDataset, unmasked_only, uniform_schedule


class TestNoiseSchedules:
    def test_unmasked_only_returns_one(self):
        for _ in range(100):
            assert unmasked_only() == 1.0

    def test_uniform_schedule_in_range(self):
        for _ in range(1000):
            t = uniform_schedule()
            assert 0.0 <= t <= 1.0


class TestGuidanceDataset:
    def test_dataset_length(self):
        sequences = ["MVLSPADKTN", "MVLSGEDKSN", "MVLSAADKTN"]
        labels = np.array([1.0, 2.0, 3.0])
        dataset = GuidanceDataset(
            sequences=sequences,
            labels=labels,
            tokenize=lambda x: {"input_ids": torch.tensor([1, 2, 3])},
            noise_schedule=unmasked_only,
            mask_token="<mask>",
        )
        assert len(dataset) == 3

    def test_dataset_getitem_structure(self):
        sequences = ["MVLSPADKTN", "MVLSGEDKSN"]
        labels = np.array([1.5, 2.5])
        dataset = GuidanceDataset(
            sequences=sequences,
            labels=labels,
            tokenize=lambda x: {"input_ids": torch.tensor([1, 2, 3])},
            noise_schedule=unmasked_only,
            mask_token="<mask>",
        )

        item = dataset[0]
        assert (
            "sequence" not in item
        )  # GuidanceDataset __getitem__ doesn't return 'sequence' anymore, check code
        assert "input" in item
        assert "labels" in item
        # assert item["sequence"] == "MVLSPADKTN" # Removed from __getitem__
        assert item["labels"] == 1.5

        item = dataset[1]
        assert item["labels"] == 2.5

    def test_dataset_mismatched_lengths_fails(self):
        sequences = ["MVLSPADKTN", "MVLSGEDKSN", "MVLSAADKTN"]
        labels = np.array([1.0, 2.0])  # Only 2 labels for 3 sequences

        with pytest.raises(ValueError, match="Length mismatch"):
            GuidanceDataset(
                sequences=sequences,
                labels=labels,
                tokenize=lambda x: {"input_ids": torch.tensor([1, 2, 3])},
                noise_schedule=unmasked_only,
                mask_token="<mask>",
            )
