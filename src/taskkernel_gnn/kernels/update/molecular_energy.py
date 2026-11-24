"""Scalar node-energy updates for conservative molecular models."""

from __future__ import annotations

import torch
from torch import nn

from ..base import positive, raw_positive


class MolecularSaturatingEnergyUpdate(nn.Module):
    """Bound each atom's aggregated energy contribution with a smooth tanh."""

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.raw_scale = nn.Parameter(raw_positive(scale))

    def forward(self, node_state, aggregate, context=None):
        scale = positive(self.raw_scale)
        return scale * torch.tanh(aggregate / scale)
