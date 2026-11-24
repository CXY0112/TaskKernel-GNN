"""Wave and directed-transport message kernels."""

from __future__ import annotations
import torch
from torch import nn
from ..base import pair_geometry, positive, raw_positive, require_context


class WavePropagationKernel(nn.Module):
    def __init__(self, amplitude=1.0, wavenumber=1.0, frequency=1.0, phase=0.0):
        super().__init__(); self.raw_amplitude = nn.Parameter(raw_positive(amplitude))
        self.raw_wavenumber = nn.Parameter(raw_positive(wavenumber)); self.raw_frequency = nn.Parameter(raw_positive(frequency))
        self.phase = nn.Parameter(torch.tensor(float(phase)))
    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        source, _, _, distance, _ = pair_geometry(context, edge_index)
        time = torch.as_tensor(require_context(context, "time"), dtype=node_state.dtype, device=node_state.device)
        angle = positive(self.raw_wavenumber) * distance - positive(self.raw_frequency) * time + self.phase
        return positive(self.raw_amplitude) * torch.cos(angle) * node_state[source]


class DirectedTransportKernel(nn.Module):
    def __init__(self, flow_key="directed_flow"): super().__init__(); self.flow_key = flow_key
    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        source, target = edge_index; flow = require_context(context, self.flow_key)
        if flow.ndim == 1: flow = flow.unsqueeze(-1)
        return flow * (node_state[source] - node_state[target])
