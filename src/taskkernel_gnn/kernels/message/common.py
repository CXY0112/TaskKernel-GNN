"""Common difference and distance-weighted diffusion message kernels."""

from __future__ import annotations
import torch
from torch import nn
from ..base import pair_geometry, positive, raw_positive, require_context


class DifferenceMessageKernel(nn.Module):
    def __init__(self, scale: float = 1.0, learnable: bool = False):
        super().__init__(); value = torch.tensor(float(scale))
        if learnable: self.scale = nn.Parameter(value)
        else: self.register_buffer("scale", value)
    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        source, target = edge_index
        return self.scale * (node_state[source] - node_state[target])


class GraphDiffusionKernel(nn.Module):
    def __init__(self, diffusion: float = 1.0, weight_key: str = "edge_weight", learnable: bool = True):
        super().__init__(); self.weight_key = weight_key; raw = raw_positive(diffusion)
        if learnable: self.raw_diffusion = nn.Parameter(raw)
        else: self.register_buffer("raw_diffusion", raw)
    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        source, target = edge_index; weight = require_context(context, self.weight_key)
        if weight.ndim == 1: weight = weight.unsqueeze(-1)
        if self.raw_diffusion.ndim and self.raw_diffusion.numel() != node_state.shape[-1]:
            raise ValueError("diffusion channel count must match node features")
        return positive(self.raw_diffusion) * weight * (node_state[source] - node_state[target])


class GaussianDiffusionKernel(nn.Module):
    def __init__(self, diffusion=1.0, beta=1.0, position_key="position"):
        super().__init__(); self.position_key = position_key
        self.raw_diffusion = nn.Parameter(raw_positive(diffusion)); self.raw_beta = nn.Parameter(raw_positive(beta))
    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        source, target, delta, _, _ = pair_geometry(context, edge_index, self.position_key)
        if self.raw_diffusion.ndim and self.raw_diffusion.numel() != node_state.shape[-1]:
            raise ValueError("diffusion channel count must match node features")
        weight = positive(self.raw_diffusion) * torch.exp(-positive(self.raw_beta) * delta.square().sum(-1, keepdim=True))
        return weight * (node_state[source] - node_state[target])


class GaussianDistanceKernel(GaussianDiffusionKernel):
    def __init__(self, sigma=1.0, diffusion=1.0, position_key="position"):
        super().__init__(diffusion, 1.0 / (2.0 * sigma**2), position_key)


class ExponentialDiffusionKernel(nn.Module):
    def __init__(self, amplitude=1.0, beta=1.0, position_key="position"):
        super().__init__(); self.position_key = position_key
        self.raw_amplitude = nn.Parameter(raw_positive(amplitude)); self.raw_beta = nn.Parameter(raw_positive(beta))
    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        source, target, _, distance, _ = pair_geometry(context, edge_index, self.position_key)
        weight = positive(self.raw_amplitude) * torch.exp(-positive(self.raw_beta) * distance)
        return weight * (node_state[source] - node_state[target])


class InversePowerDistanceKernel(nn.Module):
    def __init__(self, amplitude=1.0, power=1.0, epsilon=0.1, position_key="position"):
        super().__init__(); self.position_key = position_key; self.epsilon = float(epsilon)
        self.raw_amplitude = nn.Parameter(raw_positive(amplitude)); self.raw_power = nn.Parameter(raw_positive(power))
    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        source, _, _, distance, _ = pair_geometry(context, edge_index, self.position_key)
        return positive(self.raw_amplitude) * (distance + self.epsilon).pow(-positive(self.raw_power)) * node_state[source]
