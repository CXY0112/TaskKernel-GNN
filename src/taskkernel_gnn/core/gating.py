"""Temperature-softmax gates used by message and update kernel banks."""

from __future__ import annotations

import math

import torch
from torch import nn


def annealed_temperature(
    progress: float,
    start: float = 1.0,
    end: float = 0.5,
    hold_fraction: float = 0.30,
) -> float:
    """Piecewise-linear temperature schedule used in the frozen method."""
    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must lie in [0, 1]")
    if start <= 0 or end <= 0:
        raise ValueError("softmax temperatures must be positive")
    if progress <= hold_fraction:
        return float(start)
    fraction = min(1.0, (progress - hold_fraction) / max(1.0 - hold_fraction, 1e-12))
    return float(start + fraction * (end - start))


def temperature_softmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Normalize kernel logits to nonnegative weights that sum to one."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.softmax(logits / temperature, dim=-1)


def simplex_entropy(weights: torch.Tensor, *, normalized: bool = True, epsilon: float = 1e-8) -> torch.Tensor:
    """Return entropy along the candidate axis, optionally normalized to [0, 1]."""
    if weights.shape[-1] < 1:
        raise ValueError("at least one candidate weight is required")
    if epsilon <= 0:
        raise ValueError("entropy epsilon must be positive")
    entropy = -(weights * weights.clamp_min(epsilon).log()).sum(dim=-1)
    if normalized and weights.shape[-1] > 1:
        entropy = entropy / math.log(weights.shape[-1])
    return entropy


class TemperatureSoftmaxGate(nn.Module):
    """Learnable simplex gate with a frozen, explicit temperature schedule."""

    def __init__(
        self,
        kernels: int,
        *,
        start_temperature: float = 1.0,
        end_temperature: float = 0.5,
        hold_fraction: float = 0.30,
        initial_logits: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if kernels < 1:
            raise ValueError("at least one candidate kernel is required")
        logits = torch.zeros(kernels) if initial_logits is None else initial_logits.detach().float().clone()
        if logits.numel() != kernels:
            raise ValueError("initial_logits length must match kernels")
        self.logits = nn.Parameter(logits)
        self.start_temperature = float(start_temperature)
        self.end_temperature = float(end_temperature)
        self.hold_fraction = float(hold_fraction)

    def temperature(self, progress: float) -> float:
        return annealed_temperature(progress, self.start_temperature, self.end_temperature, self.hold_fraction)

    def forward(self, progress: float = 1.0) -> torch.Tensor:
        return temperature_softmax(self.logits, self.temperature(progress))
