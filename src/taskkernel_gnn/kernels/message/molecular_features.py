"""Invariant molecular feature messages, separate from scalar energy kernels."""

import torch
from torch import nn

from ..base import pair_geometry


class ScalarEnergyFeatureMessage(nn.Module):
    """Lift an existing [E,1] energy descriptor into [E,features]."""

    def __init__(self, energy_kernel: nn.Module, features: int = 32):
        super().__init__()
        if features < 1:
            raise ValueError("features must be positive")
        self.energy_kernel = energy_kernel
        self.projection = nn.Linear(1, features, bias=False)

    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        energy = self.energy_kernel(node_state, edge_index, edge_attr, context)
        return self.projection(energy)


class MolecularContinuousFilterMessage(nn.Module):
    """Distance RBF filters multiplied by source atom latent features."""

    def __init__(self, features: int = 32, radial_basis: int = 32,
                 hidden: int = 32, cutoff: float = 8.0):
        super().__init__()
        if min(features, hidden) < 1 or radial_basis < 2 or cutoff <= 0:
            raise ValueError("positive feature/hidden/cutoff and at least two radial bases required")
        self.cutoff = float(cutoff)
        self.width = self.cutoff / (radial_basis - 1)
        self.register_buffer("centers", torch.linspace(0, cutoff, radial_basis))
        self.source = nn.Linear(features, features, bias=False)
        self.filter = nn.Sequential(nn.Linear(radial_basis, hidden), nn.SiLU(),
                                    nn.Linear(hidden, features))

    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        source, _, _, distance, _ = pair_geometry(context, edge_index)
        d = distance.squeeze(-1)
        radial = torch.exp(-0.5 * ((d[:, None] - self.centers) / self.width).square())
        # Quintic cutoff has zero first and second derivatives at the boundary.
        x = (d / self.cutoff).clamp(0, 1)
        envelope = 1 - 10 * x.pow(3) + 15 * x.pow(4) - 6 * x.pow(5)
        return envelope[:, None] * self.filter(radial) * self.source(node_state[source])
