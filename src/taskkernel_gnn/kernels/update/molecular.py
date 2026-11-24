"""Equivariant force-space node updates for molecular graphs."""

from __future__ import annotations

import torch
from torch import nn

from ..base import positive, raw_positive


class MolecularSaturatingForceUpdate(nn.Module):
    """Smoothly scale an entire molecule's forces by one shared factor.

    A shared factor preserves zero net internal force when message kernels are
    antisymmetric on directed atom pairs.
    """

    def __init__(self, scale: float = 5.0):
        super().__init__()
        self.raw_scale = nn.Parameter(raw_positive(scale))

    def forward(self, node_state, aggregate, context=None):
        scale = positive(self.raw_scale)
        molecule = context["molecule_index"]
        groups = int(molecule.max()) + 1
        squared = aggregate.square().sum(-1)
        total = squared.new_zeros(groups).index_add_(0, molecule, squared)
        count = squared.new_zeros(groups).index_add_(0, molecule, torch.ones_like(squared))
        # Keep the derivative finite when an MLP-only message bank starts at
        # exactly zero (the MLP output layer is intentionally zero-initialized).
        rms = (total / count.clamp_min(1) + 1e-12).sqrt()
        multiplier = scale * torch.tanh(rms / scale) / rms
        return aggregate * multiplier[molecule, None]
