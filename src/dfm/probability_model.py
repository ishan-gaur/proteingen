"""Base class for models that produce log probability distributions."""

from __future__ import annotations

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
    """

    def __init__(self):
        super().__init__()
        self.temp = 1.0
        self.observations: Optional[Dict] = None

    @property
    def device(self):
        return next(self.parameters()).device

    @abstractmethod
    def preprocess_observations(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Transform raw observations into cached form.

        Called once when ``set_condition_()`` or ``conditioned_on()`` is
        invoked. Use this for expensive operations (e.g. encoding structure)
        that should not be repeated every forward pass.
        """
        ...

    @staticmethod
    @abstractmethod
    def collate_observations(
        x_B: torch.Tensor, observations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collate observations for input to the forward function."""
        ...

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
        if self.observations is not None:
            obs = self.collate_observations(x_B, self.observations)
            raw_output = self.forward(x_B, **obs)
            log_probs = self.format_raw_to_logits(raw_output, x_B, **obs)
        else:
            raw_output = self.forward(x_B)
            log_probs = self.format_raw_to_logits(raw_output, x_B)
        return F.log_softmax(log_probs / self.temp, dim=-1)
