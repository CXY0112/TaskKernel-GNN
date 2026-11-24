"""Rotation-invariant molecular energy messages.

Every directed-edge contribution sums to a molecular energy once. Forces must
be obtained by differentiating the final scalar energy with respect to atomic
positions, not by treating these messages as forces.
"""

from __future__ import annotations

import torch
from torch import nn

from ..base import pair_geometry, positive, raw_positive, require_context


class MolecularBondEnergyKernel(nn.Module):
    """Harmonic bond potential shared by unordered element-pair type."""

    def __init__(self, pair_types: int = 6):
        super().__init__()
        if pair_types != 6:
            raise ValueError("the current element-pair table is defined for H/C/O")
        self.raw_stiffness = nn.Parameter(raw_positive([8.0] * pair_types))
        self.rest_length = nn.Parameter(torch.tensor([0.74, 1.09, 0.97, 1.52, 1.43, 1.48]))

    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        _, _, _, distance, _ = pair_geometry(context, edge_index)
        kind = require_context(context, "pair_type")
        bonded = require_context(context, "bond_edge_mask")
        stretch = distance.squeeze(-1) - self.rest_length[kind]
        # Two directed copies of each physical bond; each carries half its energy.
        return (0.25 * positive(self.raw_stiffness[kind]) * stretch.square() * bonded)[:, None]


class MolecularAngleEnergyKernel(nn.Module):
    """Harmonic cosine-angle potential distributed over four directed edges."""

    def __init__(self):
        super().__init__()
        self.raw_stiffness = nn.Parameter(raw_positive([2.0, 3.0, 3.0]))
        self.raw_rest_cosine = nn.Parameter(torch.atanh(torch.tensor([-0.33, -0.33, -0.33])))

    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        position = require_context(context, "position")
        triplets = require_context(context, "angle_triplets")
        edges = require_context(context, "angle_edge_indices")
        center_type = require_context(context, "angle_center_type")
        message = position.new_zeros((edge_index.shape[1], 1))
        if triplets.numel() == 0:
            return message
        i, j, k = triplets.unbind(-1)
        u, v = position[i] - position[j], position[k] - position[j]
        cosine = (u * v).sum(-1) / (u.norm(dim=-1).clamp_min(1e-8) * v.norm(dim=-1).clamp_min(1e-8))
        rest = self.raw_rest_cosine[center_type].tanh()
        total = 0.5 * positive(self.raw_stiffness[center_type]) * (cosine - rest).square()
        for column in range(4):
            message.index_add_(0, edges[:, column], (total / 4)[:, None])
        return message


class MolecularNonbondedEnergyKernel(nn.Module):
    """Smooth radial short-range repulsion and attraction energy."""

    def __init__(self):
        super().__init__()
        self.raw_repulsion = nn.Parameter(raw_positive([0.5] * 6))
        self.raw_attraction = nn.Parameter(raw_positive([0.2] * 6))
        self.register_buffer("radius", torch.tensor([2.0, 2.4, 2.2, 2.9, 2.7, 2.5]))

    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        _, _, _, distance, _ = pair_geometry(context, edge_index)
        kind = require_context(context, "pair_type")
        mask = require_context(context, "nonbonded_edge_mask")
        d = distance.squeeze(-1)
        radius = self.radius[kind]
        compression = ((radius - d) / radius).clamp_min(0)
        potential = (0.5 * positive(self.raw_repulsion[kind]) * compression.square()
                     - positive(self.raw_attraction[kind]) * torch.exp(-((d - radius) / 0.8).square()))
        return (0.5 * potential * mask)[:, None]


class MolecularPairMLPEnergyKernel(nn.Module):
    """Symmetric learned pair energy using radial and local-density features."""

    def __init__(self, hidden: int = 64, amplitude: float = 0.2):
        super().__init__()
        if hidden < 1 or amplitude <= 0:
            raise ValueError("hidden and amplitude must be positive")
        self.amplitude = float(amplitude)
        self.net = nn.Sequential(nn.Linear(17, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("centers", torch.tensor([0.7, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0, 5.0]))

    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        source, target, _, distance, _ = pair_geometry(context, edge_index)
        d = distance.squeeze(-1)
        kind = require_context(context, "pair_type")
        bonded = require_context(context, "bond_edge_mask")
        radial = torch.exp(-((d[:, None] - self.centers) / 0.6).square())
        species = torch.nn.functional.one_hot(kind, num_classes=6).to(radial.dtype)
        density = d.new_zeros((len(node_state),))
        density.index_add_(0, target, torch.exp(-d.square() / 4.0))
        left, right = density[source] / 18, density[target] / 18
        features = torch.cat([radial, species, bonded[:, None],
                              ((left + right) / 2)[:, None], (left - right).abs()[:, None]], dim=-1)
        return 0.5 * self.amplitude * torch.tanh(self.net(features))


class MolecularTripletMLPEnergyKernel(nn.Module):
    """Permutation-invariant three-body energy for an atom-centered triplet."""

    def __init__(self, hidden: int = 32, amplitude: float = 0.2):
        super().__init__()
        if hidden < 1 or amplitude <= 0:
            raise ValueError("hidden and amplitude must be positive")
        self.amplitude = float(amplitude)
        self.net = nn.Sequential(nn.Linear(18, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        position = require_context(context, "position")
        triplets = require_context(context, "angle_triplets")
        edges = require_context(context, "angle_edge_indices")
        center_type = require_context(context, "angle_center_type")
        pair_type = require_context(context, "pair_type")
        message = position.new_zeros((edge_index.shape[1], 1))
        if triplets.numel() == 0:
            return message
        i, j, k = triplets.unbind(-1)
        u, v = position[i] - position[j], position[k] - position[j]
        ru, rv = u.norm(dim=-1).clamp_min(1e-8), v.norm(dim=-1).clamp_min(1e-8)
        cosine = (u * v).sum(-1) / (ru * rv)
        first = torch.nn.functional.one_hot(pair_type[edges[:, 0]], num_classes=6).to(position.dtype)
        second = torch.nn.functional.one_hot(pair_type[edges[:, 1]], num_classes=6).to(position.dtype)
        center = torch.nn.functional.one_hot(center_type, num_classes=3).to(position.dtype)
        features = torch.cat([
            first + second, (first - second).abs(), center,
            ((ru + rv) / 2)[:, None], (ru - rv).abs()[:, None], cosine[:, None],
        ], dim=-1)
        total = self.amplitude * torch.tanh(self.net(features)).squeeze(-1)
        for column in range(4):
            message.index_add_(0, edges[:, column], (total / 4)[:, None])
        return message
