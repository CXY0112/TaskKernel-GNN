"""Fixed response-scale calibration for heterogeneous candidate kernels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn


def rms_scales(candidates: Sequence[torch.Tensor], minimum: float = 1e-8) -> torch.Tensor:
    """Return one scalar RMS response for every candidate tensor."""
    if not candidates:
        raise ValueError("candidate sequence cannot be empty")
    return torch.stack([candidate.detach().square().mean().sqrt().clamp_min(minimum) for candidate in candidates])


class FixedRMSCalibrator(nn.Module):
    """Fit fixed multipliers that map candidate RMS to one reference RMS."""

    def __init__(
        self,
        names: Sequence[str],
        scales: Sequence[float] | torch.Tensor | None = None,
        *,
        minimum: float = 1e-8,
        minimum_scale: float = 0.1,
        maximum_scale: float = 10.0,
        reference_index: int = 0,
    ) -> None:
        super().__init__()
        self.names = tuple(names)
        if len(set(self.names)) != len(self.names) or not self.names:
            raise ValueError("candidate names must be nonempty and unique")
        if not 0 <= reference_index < len(self.names):
            raise ValueError("reference_index is outside the candidate bank")
        if minimum <= 0 or minimum_scale <= 0 or maximum_scale < minimum_scale:
            raise ValueError("invalid RMS calibration limits")
        self.minimum = float(minimum)
        self.minimum_scale = float(minimum_scale)
        self.maximum_scale = float(maximum_scale)
        self.reference_index = int(reference_index)
        initial = torch.ones(len(self.names)) if scales is None else torch.as_tensor(scales, dtype=torch.float32)
        if initial.numel() != len(self.names) or torch.any(initial <= 0):
            raise ValueError("scales must be positive and match candidate names")
        self.register_buffer("scales", initial.clone())
        self.register_buffer("response_rms", torch.ones(len(self.names)))

    @torch.no_grad()
    def fit(self, candidates: Mapping[str, torch.Tensor]) -> None:
        missing = set(self.names) - set(candidates)
        if missing:
            raise KeyError(f"missing calibration responses: {sorted(missing)}")
        rms = rms_scales([candidates[name] for name in self.names], self.minimum).to(self.scales)
        multipliers = (rms[self.reference_index] / rms).clamp(self.minimum_scale, self.maximum_scale)
        self.response_rms.copy_(rms)
        self.scales.copy_(multipliers)

    def forward(self, candidates: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(candidates) != len(self.names):
            raise ValueError("candidate count does not match calibrator")
        return [candidate * self.scales[index] for index, candidate in enumerate(candidates)]

    def as_dict(self) -> dict[str, float]:
        return {name: float(self.scales[index]) for index, name in enumerate(self.names)}
