"""Local reaction and first-order damping update kernels."""

from __future__ import annotations
from collections.abc import Sequence
from itertools import combinations_with_replacement
import torch
from torch import nn
import torch.nn.functional as F


class LinearReactionUpdate(nn.Module):
    def __init__(self,rate=0.0,dt=1.0,learnable=True):
        super().__init__(); value=torch.tensor(float(rate)); self.dt=float(dt)
        if learnable:self.rate=nn.Parameter(value)
        else:self.register_buffer("rate",value)
    def forward(self,node_state,aggregate,context=None): return self.dt*(aggregate+self.rate*node_state)


class PolynomialReactionUpdate(nn.Module):
    def __init__(self,coefficients:Sequence[float],dt=1.0):
        super().__init__();
        if not coefficients: raise ValueError("at least one polynomial coefficient is required")
        self.coefficients=nn.Parameter(torch.tensor(tuple(coefficients),dtype=torch.float32)); self.dt=float(dt)
    def forward(self,node_state,aggregate,context=None):
        reaction=torch.zeros_like(node_state)
        for power,coefficient in enumerate(self.coefficients,start=1): reaction=reaction+coefficient*node_state.pow(power)
        return self.dt*(aggregate+reaction)


class CoupledPolynomialReactionUpdate(nn.Module):
    """Full multivariate polynomial reaction basis up to a chosen degree.

    Unlike :class:`PolynomialReactionUpdate`, this kernel includes a constant
    and every cross-channel monomial, so it remains generic while representing
    coupled multi-species reaction systems.
    """

    def __init__(self, state_features: int, output_features: int | None = None,
                 degree: int = 3, dt: float = 1.0, initial_scale: float = 0.02):
        super().__init__()
        if state_features < 1 or degree < 1 or initial_scale < 0:
            raise ValueError("state_features and degree must be positive; initial_scale must be nonnegative")
        self.state_features = int(state_features)
        self.output_features = int(output_features or state_features)
        self.degree = int(degree)
        self.dt = float(dt)
        self.exponents = tuple(
            combination
            for order in range(1, self.degree + 1)
            for combination in combinations_with_replacement(range(self.state_features), order)
        )
        self.coefficients = nn.Parameter(torch.empty(self.output_features, 1 + len(self.exponents)))
        nn.init.normal_(self.coefficients, mean=0.0, std=float(initial_scale))

    def basis(self, node_state: torch.Tensor) -> torch.Tensor:
        terms = [torch.ones_like(node_state[..., :1])]
        for indices in self.exponents:
            term = torch.ones_like(node_state[..., :1])
            for index in indices:
                term = term * node_state[..., index:index + 1]
            terms.append(term)
        return torch.cat(terms, dim=-1)

    def forward(self, node_state, aggregate, context=None):
        reaction = self.basis(node_state) @ self.coefficients.transpose(0, 1)
        return self.dt * (aggregate + reaction)


class SaturatingReactionUpdate(nn.Module):
    """Generic coupled linear-plus-saturating local reaction kernel.

    Each state channel contributes both directly and through the bounded feature
    ``x / (K + |x|)`` with a learned positive half-saturation scale ``K``.
    """

    def __init__(self, state_features: int, output_features: int | None = None,
                 half_saturation: float = 1.0, dt: float = 1.0,
                 initial_scale: float = 0.02):
        super().__init__()
        if state_features < 1 or half_saturation <= 0 or initial_scale < 0:
            raise ValueError("state_features and half_saturation must be positive; initial_scale must be nonnegative")
        self.state_features = int(state_features)
        self.output_features = int(output_features or state_features)
        self.dt = float(dt)
        raw = torch.log(torch.expm1(torch.tensor(float(half_saturation))))
        self.raw_half_saturation = nn.Parameter(raw.repeat(self.state_features))
        self.coefficients = nn.Parameter(torch.empty(self.output_features, 1 + 2 * self.state_features))
        nn.init.normal_(self.coefficients, mean=0.0, std=float(initial_scale))

    def basis(self, node_state: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.raw_half_saturation).clamp_min(1e-8)
        saturated = node_state / (scale + node_state.abs())
        return torch.cat((torch.ones_like(node_state[..., :1]), node_state, saturated), dim=-1)

    def forward(self, node_state, aggregate, context=None):
        reaction = self.basis(node_state) @ self.coefficients.transpose(0, 1)
        return self.dt * (aggregate + reaction)


class LogisticReactionUpdate(nn.Module):
    def __init__(self,rate=1.0,capacity=1.0,dt=1.0):
        super().__init__(); self.raw_rate=nn.Parameter(torch.tensor(float(rate)).clamp_min(1e-8).log()); self.raw_capacity=nn.Parameter(torch.tensor(float(capacity)).clamp_min(1e-8).log()); self.dt=float(dt)
    def forward(self,node_state,aggregate,context=None):
        rate,capacity=self.raw_rate.exp(),self.raw_capacity.exp(); return self.dt*(aggregate+rate*node_state*(1-node_state/capacity))


class DampedUpdate(nn.Module):
    def __init__(self,damping=0.1,dt=1.0): super().__init__(); self.raw_damping=nn.Parameter(torch.tensor(float(damping)).clamp_min(1e-8).log()); self.dt=float(dt)
    def forward(self,node_state,aggregate,context=None): return self.dt*(aggregate-self.raw_damping.exp()*node_state)
