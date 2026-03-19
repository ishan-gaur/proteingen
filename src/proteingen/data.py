"""
Base dataset class for guidance training.
"""

from typing import Any, Callable, Iterable
import random
import torch


# Type alias for noise schedule functions.
# Returns a timestep t in [0, 1] where t=0 is fully masked and t=1 is fully unmasked.
# During noising, a position is masked if random() > t.
NoiseSchedule = Callable[[], float]


def unmasked_only() -> float:
    """Noise schedule that always returns t=1 (no masking)."""
    return 1.0


def uniform_schedule() -> float:
    """Noise schedule that samples t uniformly from [0, 1]."""
    return random.random()


# Guidance dataset build on top of regular model training dataset, which is linked to its sampler
# Conditional training and guidance training can use the same dataset, they just treat different parts as inputs and output
class GuidanceDataset(torch.utils.data.Dataset):
    """
    Dataset for training noisy classifiers for guidance.

    Args:
        sequences: List of sequences (proteins, etc.)
        labels: Array of target values (n_samples,) or (n_samples, n_targets)
        tokenize: Callable that converts (sequences, structures, ...) -> model input.
        noise_schedule: Callable() -> float in [0,1], samples timestep.
        sequence_metadata: Optional data passed to tokenize() for each sequence (e.g. structures).
        mask_token: Token to use for masking. Required if noise_schedule is set.

    TODO:
        - Implement noising in __getitem__ (sample t, mask positions where rand < t)
        - Add train/dev/test split support with reproducible seeding
    """

    def __init__(
        self,
        sequences: list[str],
        labels: torch.Tensor,
        tokenize: Callable[..., dict[str, torch.Tensor]],
        noise_schedule: NoiseSchedule,
        mask_token: str,
        sequence_metadata: Any = None,
    ):
        self.sequences = sequences
        self.labels = labels
        self.tokenize = tokenize
        self.noise_schedule = noise_schedule
        self.sequence_metadata = sequence_metadata
        self.mask_token = mask_token

        if len(sequences) != len(self.labels):
            raise ValueError(
                f"Length mismatch: {len(sequences)} sequences, {len(self.labels)} labels"
            )

        if self.sequence_metadata is not None and not isinstance(self.sequence_metadata, Iterable) and not hasattr(
            self.sequence_metadata, "__getitem__"
        ):
            raise ValueError(
                "sequence_metadata must be an indexable iterable (e.g. list or tensor)"
            )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Returns dict with:
            - 'sequence': raw sequence string
            - 'labels': target values
            - If tokenize is set: keys from tokenize() output
            - If noise_schedule is set: 'timestep' and noised inputs

        """
        sequence = self.sequences[idx]
        t = self.noise_schedule()
        mask = torch.rand(len(sequence)) > t
        for i, to_mask in enumerate(mask):
            if to_mask:
                sequence[i] = self.mask_token
        if self.sequence_metadata is not None:
            metadata = self.sequence_metadata[idx]
            tokenized = self.tokenize(sequence, metadata)
        else:
            tokenized = self.tokenize(sequence)
        return {
            "input": tokenized,
            "labels": self.labels[idx],
            "timestep": t,
            "mask": mask,
        }
