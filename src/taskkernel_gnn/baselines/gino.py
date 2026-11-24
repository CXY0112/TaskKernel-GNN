"""Geometry-informed neural operator baseline for irregular meshes."""

from __future__ import annotations

import math

import torch
from torch import nn

from .graph_pde import mlp, scatter_add, graph_indices


class GeometryInformedNeuralOperator(nn.Module):
    def __init__(self, *, coordinate_dim: int = 3, state_features: int = 1,
                 hidden_features: int = 64, layers: int = 4, fourier_bands: int = 6,
                 dt: float = 1.0) -> None:
        super().__init__()
        self.coordinate_dim = coordinate_dim
        self.state_features = state_features
        self.fourier_bands = fourier_bands
        self.dt = float(dt)
        encoded_coordinates = coordinate_dim * (1 + 2 * fourier_bands)
        self.lift = mlp(state_features + encoded_coordinates, hidden_features, hidden_features)
        self.kernels = nn.ModuleList(
            mlp(2 * coordinate_dim + 1, hidden_features, hidden_features) for _ in range(layers)
        )
        self.values = nn.ModuleList(nn.Linear(hidden_features, hidden_features, bias=False) for _ in range(layers))
        self.locals = nn.ModuleList(nn.Linear(hidden_features, hidden_features) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_features) for _ in range(layers))
        self.project = mlp(hidden_features, hidden_features, state_features)

    def encode_coordinates(self, coordinates: torch.Tensor) -> torch.Tensor:
        values = [coordinates]
        for band in range(self.fourier_bands):
            frequency = (2.0**band) * math.pi
            values.extend((torch.sin(frequency * coordinates), torch.cos(frequency * coordinates)))
        return torch.cat(values, -1)

    def step(self, state: torch.Tensor, sample, time_index: int | None = None) -> torch.Tensor:
        source, target = graph_indices(sample)
        coordinates = sample.coordinates[:, : self.coordinate_dim]
        state_matrix = state[:, None] if state.ndim == 1 else state
        nodes = self.lift(torch.cat((state_matrix, self.encode_coordinates(coordinates)), -1))
        delta = coordinates[source] - coordinates[target]
        distance = delta.square().sum(-1, keepdim=True).sqrt()
        edge_input = torch.cat((coordinates[source], coordinates[target], distance), -1)
        for kernel, value, local, norm in zip(self.kernels, self.values, self.locals, self.norms):
            weights = torch.sigmoid(kernel(edge_input))
            message = weights * value(nodes[source])
            aggregate = scatter_add(message, target, sample.nodes) / sample.degree.clamp_min(1).unsqueeze(-1)
            nodes = norm(nodes + torch.tanh(local(nodes) + aggregate))
        result = state_matrix + self.dt * torch.tanh(self.project(nodes))
        return result.squeeze(-1) if state.ndim == 1 else result
