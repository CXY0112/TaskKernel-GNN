"""Generic equivariant molecular force message kernels.

All outputs are forces on target atoms in eV/Å.  These are structural priors,
not a decomposition of a quantum-chemical force label.
"""

from __future__ import annotations

import torch
from torch import nn

from ..base import pair_geometry, positive, raw_positive, require_context


class MolecularBondStretchKernel(nn.Module):
    """Learnable harmonic bond stretch by unordered element-pair type."""

    def __init__(self, pair_types: int = 6):
        super().__init__()
        self.raw_stiffness = nn.Parameter(raw_positive([8.0] * pair_types))
        self.rest_length = nn.Parameter(torch.tensor([0.74, 1.09, 0.97, 1.52, 1.43, 1.48]))
        if pair_types != 6:
            raise ValueError("the current element-pair table is defined for H/C/O")

    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        _, _, _, distance, direction = pair_geometry(context, edge_index)
        kind = require_context(context, "pair_type")
        bonded = require_context(context, "bond_edge_mask")
        stretch = distance.squeeze(-1) - self.rest_length[kind]
        scalar = positive(self.raw_stiffness[kind]) * stretch * bonded
        return scalar.unsqueeze(-1) * direction


class MolecularAngleBendKernel(nn.Module):
    """Three-body harmonic cosine-angle force distributed over directed bonds.

    For each triplet (i, j, k), j is the central atom.  The triplet's force
    on i and k is sent from j; its reaction force on j is sent from i and k.
    Thus add aggregation recovers the full three-body force and its net force
    is zero.
    """

    def __init__(self):
        super().__init__()
        self.raw_stiffness = nn.Parameter(raw_positive([2.0, 3.0, 3.0]))
        self.raw_rest_cosine = nn.Parameter(torch.atanh(torch.tensor([-0.33, -0.33, -0.33])))

    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        position = require_context(context, "position")
        triplets = require_context(context, "angle_triplets")
        edges = require_context(context, "angle_edge_indices")
        center_type = require_context(context, "angle_center_type")
        message = position.new_zeros((edge_index.shape[1], 3))
        if triplets.numel() == 0:
            return message
        i, j, k = triplets.unbind(-1)
        u = position[i] - position[j]
        v = position[k] - position[j]
        ru = u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        rv = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cosine = ((u * v).sum(-1, keepdim=True) / (ru * rv)).clamp(-1.0, 1.0)
        rest = self.raw_rest_cosine[center_type].tanh().unsqueeze(-1)
        stiffness = positive(self.raw_stiffness[center_type]).unsqueeze(-1)
        coefficient = -stiffness * (cosine - rest)
        force_i = coefficient * (v / (ru * rv) - cosine * u / ru.square())
        force_k = coefficient * (u / (ru * rv) - cosine * v / rv.square())
        message.index_add_(0, edges[:, 0], force_i)
        message.index_add_(0, edges[:, 1], force_k)
        message.index_add_(0, edges[:, 2], -force_i)
        message.index_add_(0, edges[:, 3], -force_k)
        return message


class MolecularNonbondedKernel(nn.Module):
    """Short-range repulsion plus smooth attraction on nonbonded pairs."""

    def __init__(self):
        super().__init__()
        self.raw_repulsion = nn.Parameter(raw_positive([0.5] * 6))
        self.raw_attraction = nn.Parameter(raw_positive([0.2] * 6))
        self.register_buffer("radius", torch.tensor([2.0, 2.4, 2.2, 2.9, 2.7, 2.5]))

    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        _, _, _, distance, direction = pair_geometry(context, edge_index)
        kind = require_context(context, "pair_type")
        nonbonded = require_context(context, "nonbonded_edge_mask")
        radius = self.radius[kind]
        d = distance.squeeze(-1)
        repulsion = positive(self.raw_repulsion[kind]) * ((radius - d) / radius).clamp_min(0)
        attraction = positive(self.raw_attraction[kind]) * torch.exp(-((d - radius) / 0.8).square())
        scalar = (attraction - repulsion) * nonbonded
        return scalar.unsqueeze(-1) * direction


class MolecularEquivariantMLPKernel(nn.Module):
    """Learn a bounded scalar pair correction with local environment features."""

    def __init__(self, hidden: int = 64, eta: float = 0.1, amplitude: float = 2.0):
        super().__init__()
        if eta < 0 or amplitude <= 0:
            raise ValueError("eta must be nonnegative and amplitude positive")
        self.eta = float(eta)
        self.amplitude = float(amplitude)
        self.net = nn.Sequential(nn.Linear(17, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("centers", torch.tensor([0.7, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0, 5.0]))

    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        source, target, _, distance, direction = pair_geometry(context, edge_index)
        d = distance.squeeze(-1)
        kind = require_context(context, "pair_type")
        bonded = require_context(context, "bond_edge_mask")
        radial = torch.exp(-((d[:, None] - self.centers) / 0.6).square())
        species = torch.nn.functional.one_hot(kind, num_classes=6).to(radial.dtype)
        local_density = d.new_zeros((len(node_state),))
        local_density.index_add_(0, target, torch.exp(-d.square() / 4.0))
        source_density = local_density[source] / 18
        target_density = local_density[target] / 18
        features = torch.cat(
            [radial, species, bonded[:, None],
             (source_density + target_density)[:, None],
             (source_density - target_density).abs()[:, None]], dim=-1,
        )
        scalar = self.eta * self.amplitude * torch.tanh(self.net(features).squeeze(-1))
        return scalar.unsqueeze(-1) * direction
