"""Base class for models that produce log probability distributions."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Optional, Any, Dict


class ProbabilityModel(nn.Module, ABC):
    """Base class for models that produce log probability distributions.

    Subclasses implement:
    1. ``preprocess_observations`` to turn conditioning variables input set
    through, e.g. set_condition into collated input tensors that match the
    forward function kwargs
    2. ``forward`` to return logits based on the conditioning information
    3. ``format_raw_to_logits``` to convert the forward output into something
    that can be safely softmaxxed in ``get_log_probs()``. ``format_raw_to_logits``
    typically includes output masking, coarse-graining classes, turning ensemble
    predictions into probabilities, etc.

    The default ``get_log_probs`` applies ``log_softmax`` along the last
    dimension, which is appropriate for class-valued models. Models with
    other output types (real-valued, ensemble, etc.) should override it.

    Checkpointing:
        Subclasses that support save/load implement ``_save_args()`` returning
        a JSON-serializable dict of constructor kwargs. The base class provides
        ``save(path)`` and ``from_checkpoint(path)`` that serialize/deserialize
        the constructor args and rebuild the object. Subclasses add their own
        state (weights, adapters, etc.) on top by overriding save/from_checkpoint
        and calling super().
    """

    def __init__(self):
        super().__init__()
        self.temp = 1.0
        self.observations: Optional[Dict] = None

    @property
    def device(self):
        return next(self.parameters()).device

    def preprocess_observations(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Transform raw observations into cached form.

        Called once when ``set_condition_()`` or ``conditioned_on()`` is
        invoked. Override for expensive operations (e.g. encoding structure)
        that should not be repeated every forward pass.

        Default: pass through.
        """
        return observations

    def collate_observations(
        self, x_B: torch.Tensor, observations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collate observations for input to the forward function.

        Default: tile each observation tensor to match batch size.
        Override when you need custom collation (e.g. only expanding
        certain keys, or handling non-tensor observations).
        """
        batch_size = x_B.size(0)
        return {
            k: v.unsqueeze(0).expand(batch_size, *v.shape)
            if isinstance(v, torch.Tensor) and v.dim() > 0
            else v
            for k, v in observations.items()
        }

    def set_condition_(self, observations: Dict[str, Any]):
        """Preprocess and cache observations in-place."""
        self.observations = self.preprocess_observations(observations)

    def set_condition(self, observations: Dict[str, Any]):
        """Preprocess and cache observations, returning self for chaining."""
        self.set_condition_(observations)
        return self

    @contextmanager
    def conditioned_on(self, observations: Dict[str, Any]):
        """Context manager that temporarily sets observations, reverting on exit."""
        pre_context_obs = self.observations
        try:
            yield self.set_condition(observations)
        finally:
            self.observations = pre_context_obs

    @abstractmethod
    def forward(self, x_B: torch.Tensor, **kwargs) -> Any:
        """Return logits from batched input."""

    @abstractmethod
    def format_raw_to_logits(
        self, raw_forward_output: Any, x_B: torch.Tensor, **kwargs
    ) -> torch.FloatTensor:
        """Convert raw forward output to logits suitable for log_softmax.

        Examples: output masking (transition models), coarse-graining classes,
        turning ensemble predictions into per-class logits, etc.
        """
        ...

    def set_temp_(self, temp: float):
        self.temp = temp

    def set_temp(self, temp: float):
        self.set_temp_(temp)
        return self

    @contextmanager
    def with_temp(self, temp: float):
        """Context manager to temporarily change the temperature."""
        pre_context_temp = self.temp
        try:
            yield self.set_temp(temp)
        finally:
            self.temp = pre_context_temp

    def get_log_probs(self, x_B: torch.Tensor) -> torch.FloatTensor:
        """Return temperature-scaled log probabilities.
        Input is some batched tensor with otherwise arbitrary dimensions.
        Default implementation: ``log_softmax(forward(x, **kwargs) / temp)``.
        """
        assert self.temp > 0, f"Temperature must be positive, got {self.temp}"
        if self.observations is not None:
            obs = self.collate_observations(x_B, self.observations)
            raw_output = self.forward(x_B, **obs)
            log_probs = self.format_raw_to_logits(raw_output, x_B, **obs)
        else:
            raw_output = self.forward(x_B)
            log_probs = self.format_raw_to_logits(raw_output, x_B)
        return F.log_softmax(log_probs / self.temp, dim=-1)

    # ── Checkpointing ────────────────────────────────────────────────────

    def _save_args(self) -> dict:
        """Return constructor kwargs as a JSON-serializable dict.

        Override in subclasses that support checkpointing. The dict must
        contain everything needed to reconstruct the object via ``cls(**args)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _save_args() for checkpointing"
        )

    def save(self, path: str | Path) -> None:
        """Save model to a directory. Writes config.json with constructor args.

        Subclasses override to add their own state (weights, adapters, etc.)
        and should call ``super().save(path)`` first.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "config.json", "w") as f:
            json.dump(self._save_args(), f, indent=2)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "ProbabilityModel":
        """Load model from a directory. Reads config.json and calls ``cls(**args)``.

        Subclasses override to load additional state (weights, adapters, etc.)
        and should call ``super().from_checkpoint(path)`` to get the base object.
        """
        path = Path(path)
        with open(path / "config.json") as f:
            config = json.load(f)
        return cls(**config)
