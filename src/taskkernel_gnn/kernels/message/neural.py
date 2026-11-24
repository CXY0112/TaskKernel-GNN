"""Neural message completion kernels."""

from __future__ import annotations
import torch
from torch import nn


class EdgeGatedMessageKernel(nn.Module):
    def __init__(self, node_features, edge_features, output_features, hidden=32):
        super().__init__(); self.projection = nn.Linear(node_features, output_features, bias=False)
        self.gate = nn.Sequential(nn.Linear(edge_features, hidden), nn.SiLU(), nn.Linear(hidden, output_features))
    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        if edge_attr is None: raise ValueError("EdgeGatedMessageKernel requires edge_attr")
        source, _ = edge_index
        return torch.sigmoid(self.gate(edge_attr)) * self.projection(node_state[source])


class MLPMessageKernel(nn.Module):
    def __init__(self, node_features, edge_features, output_features, hidden=64, eta=0.1):
        super().__init__(); self.eta = float(eta)
        self.network = nn.Sequential(nn.Linear(2*node_features+edge_features, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, output_features))
        nn.init.zeros_(self.network[-1].weight); nn.init.zeros_(self.network[-1].bias)
    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        source, target = edge_index
        if edge_attr is None: edge_attr = node_state.new_zeros((source.numel(), 0))
        return self.eta * torch.tanh(self.network(torch.cat([node_state[source], node_state[target], edge_attr], -1)))
