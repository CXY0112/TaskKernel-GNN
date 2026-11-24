"""Cardiac FEM-adapter message kernels."""

from __future__ import annotations
import torch
from torch import nn


class AdapterDiffusionKernel(nn.Module):
    def __init__(self, adapter_key, scale=1.0, state_channel=0, learnable=True):
        super().__init__(); self.adapter_key = adapter_key; self.state_channel = int(state_channel)
        raw = torch.tensor(float(scale)).clamp_min(1e-8).log()
        if learnable: self.log_scale = nn.Parameter(raw)
        else: self.register_buffer("log_scale", raw)
    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        source, target = edge_index; adapter = context[self.adapter_key]
        jump = node_state[source, self.state_channel] - node_state[target, self.state_channel]
        return (self.log_scale.exp() * adapter * jump).unsqueeze(-1)
