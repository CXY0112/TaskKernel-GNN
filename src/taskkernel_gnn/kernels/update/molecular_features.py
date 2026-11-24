"""Smooth residual updates for molecular latent feature states."""

import torch
from torch import nn


class MolecularFeatureLinearUpdate(nn.Module):
    def __init__(self, features: int = 32):
        super().__init__()
        if features < 1:
            raise ValueError("features must be positive")
        self.projection = nn.Linear(features, features, bias=False)

    def forward(self, node_state, aggregate, context=None):
        return self.projection(aggregate)


class MolecularFeatureMLPUpdate(nn.Module):
    def __init__(self, features: int = 32, hidden: int = 64):
        super().__init__()
        if min(features, hidden) < 1:
            raise ValueError("features and hidden must be positive")
        self.network = nn.Sequential(nn.Linear(2 * features, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(),
                                     nn.Linear(hidden, features))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, node_state, aggregate, context=None):
        return self.network(torch.cat((node_state, aggregate), dim=-1))
