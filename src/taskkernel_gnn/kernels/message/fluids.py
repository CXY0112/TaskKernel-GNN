"""Local, particle-based fluid interaction candidates.

These are generic density/force approximations; they do not reproduce an MPM
solver.  All message outputs have acceleration units for unit-mass particles.
"""

from __future__ import annotations

import torch
from torch import nn

from ..base import positive, raw_positive, require_context


def compact_density(position: torch.Tensor, edge_index: torch.Tensor, radius: float) -> torch.Tensor:
    """One scalar density per particle, retaining the self contribution."""
    source, target = edge_index
    distance = (position[source] - position[target]).norm(dim=-1)
    weight = (1.0 - distance / radius).clamp_min(0).square()
    density = torch.ones(len(position), device=position.device, dtype=position.dtype)
    return density.index_add(0, target, weight)


class DensityPressureKernel(nn.Module):
    """Symmetric outward force driven by locally estimated density."""

    def __init__(self, radius: float = 0.015, strength: float = 100.0,
                 reference_density: float = 1.45) -> None:
        super().__init__()
        if radius <= 0:
            raise ValueError("pressure radius must be positive")
        self.radius = float(radius)
        self.raw_strength = nn.Parameter(raw_positive(strength))
        self.raw_reference_density = nn.Parameter(raw_positive(reference_density))

    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        source, target = edge_index
        position = require_context(context, "position")
        density = require_context(context, "density")
        delta = position[target] - position[source]
        distance = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        support = (1.0 - distance / self.radius).clamp_min(0)
        excess = (density - positive(self.raw_reference_density)).clamp_min(0)
        pressure = 0.5 * (excess[source] + excess[target]).unsqueeze(-1)
        return positive(self.raw_strength) * pressure * support * delta / distance


class FluidViscosityKernel(nn.Module):
    """Damp relative velocity inside a finite interaction radius."""

    def __init__(self, radius: float = 0.015, damping: float = 1.0) -> None:
        super().__init__()
        if radius <= 0:
            raise ValueError("viscosity radius must be positive")
        self.radius = float(radius)
        self.raw_damping = nn.Parameter(raw_positive(damping))

    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        source, target = edge_index
        position = require_context(context, "position")
        velocity = require_context(context, "velocity")
        distance = (position[source] - position[target]).norm(dim=-1, keepdim=True)
        support = (1.0 - distance / self.radius).clamp_min(0)
        return positive(self.raw_damping) * support * (velocity[source] - velocity[target])


class FluidResidualMessageKernel(nn.Module):
    """Bounded pair correction informed by density and distance to walls."""

    def __init__(self, hidden: int = 64, eta: float = 0.1,
                 acceleration_scale: float = 100.0) -> None:
        super().__init__()
        self.eta = float(eta)
        self.acceleration_scale = float(acceleration_scale)
        # Two states (position, velocity), relative displacement/distance,
        # two densities, and the receiver's four signed wall distances.
        self.network = nn.Sequential(
            nn.Linear(4 + 4 + 3 + 2 + 4, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        source, target = edge_index
        density = require_context(context, "density")
        wall_features = require_context(context, "wall_features")
        features = torch.cat((
            node_state[source], node_state[target], edge_attr,
            density[source, None], density[target, None], wall_features[target],
        ), dim=-1)
        return self.eta * self.acceleration_scale * torch.tanh(self.network(features))
