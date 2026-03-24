"""Dataset and noise utilities for training generative and predictive protein models."""

from __future__ import annotations

import random
from typing import Any, Callable, Optional

import torch
from torch.utils.data import Dataset


# ── Type aliases ─────────────────────────────────────────────────────────────

# (input_ids_1D, t) -> noised_input_ids_1D
NoiseFn = Callable[[torch.LongTensor, float], torch.LongTensor]

# () -> float in [0, 1]
TimeSampler = Callable[[], float]


# ── Noise functions ──────────────────────────────────────────────────────────


def uniform_mask_noise(tokenizer: Any) -> NoiseFn:
    """Mask non-special positions independently with probability (1 - t).

    At t=1 nothing is masked; at t=0 everything maskable is masked.
    Uses ``tokenizer.all_special_ids`` to determine which positions to leave alone
    (CLS, EOS, PAD, MASK, etc.), so the logic is tokenizer-agnostic.
    """
    mask_id = tokenizer.vocab["<mask>"]
    special = torch.tensor(tokenizer.all_special_ids)

    def noise(input_ids: torch.LongTensor, t: float) -> torch.LongTensor:
        maskable = ~torch.isin(input_ids, special)
        to_mask = maskable & (torch.rand(input_ids.shape) > t)
        out = input_ids.clone()
        out[to_mask] = mask_id
        return out

    return noise


def no_noise(input_ids: torch.LongTensor, t: float) -> torch.LongTensor:
    """Identity noise function — returns input unchanged."""
    return input_ids


def fully_unmasked() -> float:
    """Time sampler that always returns t=1 (no masking)."""
    return 1.0


def uniform_time() -> float:
    """Time sampler that returns t ~ Uniform(0, 1)."""
    return random.random()


# ── Dataset ──────────────────────────────────────────────────────────────────


class ProteinDataset(Dataset):
    """Raw protein data: sequences, observations (conditioning variables), and labels.

    Stores raw data only — all model-specific transforms (tokenization,
    observation preprocessing, noising, padding) happen in the collator
    returned by :meth:`collator`.

    Args:
        sequences: Amino acid strings.
        observations: Dict mapping names to per-sample lists (e.g. structures,
            temperatures). Each value must be indexable with the same length
            as ``sequences``.
        labels: Per-sample targets. Shape ``(N,)`` or ``(N, n_targets)``.
    """

    def __init__(
        self,
        sequences: list[str],
        observations: Optional[dict[str, list[Any]]] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        self.sequences = sequences
        self.observations = observations or {}
        self.labels = labels

        n = len(sequences)
        for key, vals in self.observations.items():
            if len(vals) != n:
                raise ValueError(
                    f"Observation '{key}' has {len(vals)} items, expected {n}"
                )
        if self.labels is not None and len(self.labels) != n:
            raise ValueError(f"Labels has {len(self.labels)} items, expected {n}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item: dict[str, Any] = {"sequence": self.sequences[idx]}
        if self.observations:
            item["observations"] = {k: v[idx] for k, v in self.observations.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item

    def collator(
        self,
        model: Any,  # ProbabilityModel — avoids circular import
        noise_fn: NoiseFn,
        time_sampler: TimeSampler,
        rename_obs_keys: Optional[dict[str, str]] = None,
    ) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        """Build a collate_fn that tokenizes, noises, and preprocesses per batch.

        Args:
            model: Provides ``.tokenizer`` and ``.preprocess_observations`` (batched).
            noise_fn: ``(input_ids_1D, t) -> noised_input_ids_1D``. Use
                :func:`uniform_mask_noise` for MLM training or :func:`no_noise`
                for clean inputs.
            time_sampler: ``() -> float`` in ``[0, 1]``. Use :func:`uniform_time`
                or :func:`fully_unmasked`.
            rename_obs_keys: ``{model_kwarg: dataset_key}`` for renaming
                observation keys. If ``None``, dataset keys are passed through
                as model kwargs.

        Returns:
            A collate_fn producing dicts with:
                - ``input_ids``:    ``(B, L)`` — tokenized, optionally noised
                - ``target_ids``:   ``(B, L)`` — tokenized original sequences
                - ``observations``: dict ready for ``model.forward(**obs)``
                - ``labels``:       ``(B, ...)`` or ``None``
        """
        tokenizer = model.tokenizer

        def collate_fn(items: list[dict[str, Any]]) -> dict[str, Any]:
            sequences = [item["sequence"] for item in items]

            # Tokenize + pad to longest in batch
            encoded = tokenizer(sequences, padding=True, return_tensors="pt")
            target_ids = encoded["input_ids"].clone()
            input_ids = encoded["input_ids"]

            # Noise each sequence independently
            for i in range(input_ids.size(0)):
                t = time_sampler()
                input_ids[i] = noise_fn(input_ids[i], t)

            # Preprocess observations (batched)
            observations: dict[str, Any] = {}
            if "observations" in items[0]:
                if rename_obs_keys is not None:
                    # {model_kwarg: dataset_key} → {model_kwarg: [values...]}
                    batched = {
                        model_key: [item["observations"][dataset_key] for item in items]
                        for model_key, dataset_key in rename_obs_keys.items()
                    }
                else:
                    # Pass through dataset keys as-is
                    batched = {
                        key: [item["observations"][key] for item in items]
                        for key in items[0]["observations"]
                    }
                observations = model.preprocess_observations(batched)

            # Labels
            labels = None
            if "labels" in items[0]:
                labels = torch.stack([item["labels"] for item in items])

            return {
                "input_ids": input_ids,
                "target_ids": target_ids,
                "observations": observations,
                "labels": labels,
            }

        return collate_fn


# ── FASTA utilities ──────────────────────────────────────────────────────────


def read_fasta(path: str) -> list[tuple[str, str]]:
    """Read a FASTA file, returning (header, sequence) pairs.

    Concatenates multi-line sequences. Does not modify sequences
    (gaps, lowercase, etc. are preserved).
    """
    entries: list[tuple[str, str]] = []
    header = ""
    seq_parts: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header or seq_parts:
                    entries.append((header, "".join(seq_parts)))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if header or seq_parts:
        entries.append((header, "".join(seq_parts)))
    return entries


def aligned_sequences_to_raw(aligned_sequences: list[str]) -> list[str]:
    """Strip gap characters from aligned sequences to get raw AA strings.

    Removes ``-`` and ``.`` characters used in MSA formats.
    """
    return [seq.replace("-", "").replace(".", "") for seq in aligned_sequences]
