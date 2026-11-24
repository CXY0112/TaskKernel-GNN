"""Common body/wall accelerations and particle update candidates."""

from __future__ import annotations

import torch
from torch import nn

from ..base import positive, raw_positive, require_context


class UniformGravityAcceleration(nn.Module):
    """A learnable uniform body acceleration, shared by all node updates."""

    def __init__(self, dimensions: int = 2) -> None:
        super().__init__()
        self.acceleration = nn.Parameter(torch.zeros(dimensions))

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        return self.acceleration.expand_as(position)


class SoftBoxWallAcceleration(nn.Module):
    """Smooth normal repulsion plus approach-only damping at box walls."""

    def __init__(self, dimensions: int = 2, contact_radius: float = 0.015,
                 strength: float = 200.0, damping: float = 20.0,
                 maximum_offset: float = 0.02,
                 initial_lower_offset: float = -0.005,
                 initial_upper_offset: float = 0.005) -> None:
        super().__init__()
        if contact_radius <= 0 or maximum_offset <= 0:
            raise ValueError("contact radius and maximum offset must be positive")
        self.contact_radius = float(contact_radius)
        self.maximum_offset = float(maximum_offset)
        self.raw_strength = nn.Parameter(raw_positive(strength))
        self.raw_damping = nn.Parameter(raw_positive(damping))
        lower_ratio = torch.full((dimensions,), initial_lower_offset / maximum_offset)
        upper_ratio = torch.full((dimensions,), initial_upper_offset / maximum_offset)
        if lower_ratio.abs().max() >= 1 or upper_ratio.abs().max() >= 1:
            raise ValueError("initial wall offset exceeds maximum offset")
        self.raw_lower_offset = nn.Parameter(torch.atanh(lower_ratio))
        self.raw_upper_offset = nn.Parameter(torch.atanh(upper_ratio))

    def wall_distances(self, position: torch.Tensor, bounds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lower = bounds[:, 0] + self.maximum_offset * self.raw_lower_offset.tanh()
        upper = bounds[:, 1] + self.maximum_offset * self.raw_upper_offset.tanh()
        return position - lower, upper - position

    def forward(self, position: torch.Tensor, velocity: torch.Tensor,
                bounds: torch.Tensor) -> torch.Tensor:
        lower_distance, upper_distance = self.wall_distances(position, bounds)
        lower_contact = (self.contact_radius - lower_distance).clamp_min(0) / self.contact_radius
        upper_contact = (self.contact_radius - upper_distance).clamp_min(0) / self.contact_radius
        lower_approach = (-velocity).clamp_min(0)
        upper_approach = velocity.clamp_min(0)
        strength = positive(self.raw_strength)
        damping = positive(self.raw_damping)
        return (strength * (lower_contact - upper_contact)
                + damping * (lower_approach * lower_contact - upper_approach * upper_contact))


class FluidSemiImplicitUpdate(nn.Module):
    """Advance position and velocity with the common body/wall acceleration."""

    def __init__(self, dt: float = 0.0025, spatial_dims: int = 2) -> None:
        super().__init__()
        self.dt = float(dt)
        self.spatial_dims = int(spatial_dims)

    def _acceleration(self, aggregate: torch.Tensor, context: dict) -> torch.Tensor:
        return (aggregate + require_context(context, "gravity_acceleration")
                + require_context(context, "wall_acceleration"))

    def _integrate(self, node_state: torch.Tensor, acceleration: torch.Tensor) -> torch.Tensor:
        dims = self.spatial_dims
        velocity = node_state[..., dims:2 * dims]
        next_velocity = velocity + self.dt * acceleration
        return torch.cat((self.dt * next_velocity, self.dt * acceleration), dim=-1)

    def forward(self, node_state, aggregate, context=None):
        return self._integrate(node_state, self._acceleration(aggregate, context))


class FluidResidualAccelerationUpdate(FluidSemiImplicitUpdate):
    """The same physical integrator plus a bounded learned acceleration."""

    def __init__(self, dt: float = 0.0025, spatial_dims: int = 2,
                 hidden: int = 64, eta: float = 0.1,
                 acceleration_scale: float = 100.0) -> None:
        super().__init__(dt, spatial_dims)
        self.eta = float(eta)
        self.acceleration_scale = float(acceleration_scale)
        self.network = nn.Sequential(
            nn.Linear(4 + 2 + 1 + 4, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, node_state, aggregate, context=None):
        features = torch.cat((
            node_state, aggregate, require_context(context, "density")[:, None],
            require_context(context, "wall_features"),
        ), dim=-1)
        correction = self.eta * self.acceleration_scale * torch.tanh(self.network(features))
        return self._integrate(node_state, self._acceleration(aggregate, context) + correction)
