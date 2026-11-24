"""GNS, EGNN, and MeshGraphNet-style particle rollout baselines."""

from __future__ import annotations

import torch
from torch import nn

from .graph_pde import mlp, scatter_add


def _particle_inputs(history: torch.Tensor, particle_type: torch.Tensor, type_embedding: nn.Embedding,
                     velocity_mean: torch.Tensor, velocity_std: torch.Tensor,
                     acceleration_mean: torch.Tensor, acceleration_std: torch.Tensor):
    velocity = history[1:] - history[:-1]
    latest_velocity = velocity[-1]
    previous_velocity = velocity[-2] if len(velocity) > 1 else latest_velocity
    acceleration = latest_velocity - previous_velocity
    types = type_embedding(particle_type.clamp(0, type_embedding.num_embeddings - 1))
    normalized_velocity = (latest_velocity - velocity_mean) / velocity_std
    normalized_acceleration = (acceleration - acceleration_mean) / acceleration_std
    return torch.cat((normalized_velocity, normalized_acceleration, types), -1), latest_velocity


def _gns_particle_inputs(history: torch.Tensor, particle_type: torch.Tensor,
                         type_embedding: nn.Embedding, normalization: "_ParticleNormalization"):
    velocities = history[1:] - history[:-1]
    normalized = (velocities - normalization.velocity_mean) / normalization.velocity_std
    lower = history[-1] - normalization.bounds[:, 0]
    upper = normalization.bounds[:, 1] - history[-1]
    boundary = torch.cat((lower, upper), -1) / normalization.connectivity_radius
    boundary = boundary.clamp(-1.0, 1.0)
    types = type_embedding(particle_type.clamp(0, type_embedding.num_embeddings - 1))
    return torch.cat((normalized.movedim(0, 1).flatten(1), boundary, types), -1), velocities[-1]


class _ParticleNormalization(nn.Module):
    """Official Learning-to-Simulate velocity/acceleration normalization."""

    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.register_buffer("velocity_mean", torch.zeros(dimensions))
        self.register_buffer("velocity_std", torch.ones(dimensions))
        self.register_buffer("acceleration_mean", torch.zeros(dimensions))
        self.register_buffer("acceleration_std", torch.ones(dimensions))
        self.register_buffer("bounds", torch.stack((torch.zeros(dimensions), torch.ones(dimensions)), -1))
        self.register_buffer("connectivity_radius", torch.tensor(1.0))

    def configure_normalization(self, metadata: dict) -> None:
        """Load the official statistics distributed with a GNS dataset."""
        for name, key in (
            ("velocity_mean", "vel_mean"), ("velocity_std", "vel_std"),
            ("acceleration_mean", "acc_mean"), ("acceleration_std", "acc_std"),
        ):
            value = torch.as_tensor(metadata[key], dtype=getattr(self, name).dtype,
                                    device=getattr(self, name).device)
            if value.shape != getattr(self, name).shape:
                raise ValueError(f"metadata {key} has shape {tuple(value.shape)}")
            getattr(self, name).copy_(value)
        bounds = torch.as_tensor(metadata["bounds"], dtype=self.bounds.dtype, device=self.bounds.device)
        if bounds.shape != self.bounds.shape:
            raise ValueError(f"metadata bounds has shape {tuple(bounds.shape)}")
        self.bounds.copy_(bounds)
        self.connectivity_radius.copy_(torch.as_tensor(
            metadata["default_connectivity_radius"], dtype=self.connectivity_radius.dtype,
            device=self.connectivity_radius.device,
        ))

    def _decode_acceleration(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized * self.acceleration_std + self.acceleration_mean


class _GNSBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.edge = mlp(3 * hidden, hidden, hidden)
        self.node = mlp(2 * hidden, hidden, hidden)
        self.edge_norm = nn.LayerNorm(hidden)
        self.node_norm = nn.LayerNorm(hidden)

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor, source: torch.Tensor,
                target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        edges = self.edge_norm(edges + self.edge(torch.cat((nodes[source], nodes[target], edges), -1)))
        aggregate = scatter_add(edges, target, len(nodes))
        nodes = self.node_norm(nodes + self.node(torch.cat((nodes, aggregate), -1)))
        return nodes, edges


class GraphNetworkSimulator(_ParticleNormalization):
    """Encode-process-decode GNS with learned acceleration output."""

    def __init__(self, *, dimensions: int = 2, hidden: int = 128, message_steps: int = 10,
                 particle_types: int = 16, type_features: int = 16, history_steps: int = 6) -> None:
        super().__init__(dimensions)
        self.dimensions = dimensions
        self.history_steps = history_steps
        self.types = nn.Embedding(particle_types, type_features)
        self.node_encoder = mlp((history_steps - 1) * dimensions + 2 * dimensions + type_features, hidden, hidden)
        self.edge_encoder = mlp(dimensions + 1, hidden, hidden)
        self.processor = nn.ModuleList(_GNSBlock(hidden) for _ in range(message_steps))
        self.decoder = mlp(hidden, hidden, dimensions)

    def predict_next(self, history: torch.Tensor, particle_type: torch.Tensor, graph) -> torch.Tensor:
        if len(history) != self.history_steps:
            raise ValueError(f"GNS expects {self.history_steps} position frames, received {len(history)}")
        features, velocity = _gns_particle_inputs(history, particle_type, self.types, self)
        source, target = graph.source, graph.target
        relative = (history[-1, source] - history[-1, target]) / self.connectivity_radius
        distance = relative.square().sum(-1, keepdim=True).sqrt()
        nodes = self.node_encoder(features)
        edges = self.edge_encoder(torch.cat((relative, distance), -1))
        for block in self.processor:
            nodes, edges = block(nodes, edges, source, target)
        acceleration = self._decode_acceleration(self.decoder(nodes))
        return history[-1] + velocity + acceleration


class EGNNParticleSimulator(_ParticleNormalization):
    """E(n)-equivariant coordinate-update particle baseline."""

    def __init__(self, *, dimensions: int = 2, hidden: int = 128, layers: int = 4,
                 particle_types: int = 16, type_features: int = 16) -> None:
        super().__init__(dimensions)
        self.types = nn.Embedding(particle_types, type_features)
        self.node_encoder = mlp(2 * dimensions + type_features, hidden, hidden)
        self.edge_mlps = nn.ModuleList(mlp(2 * hidden + 1, hidden, hidden) for _ in range(layers))
        self.node_mlps = nn.ModuleList(mlp(2 * hidden, hidden, hidden) for _ in range(layers))
        self.coordinate_mlps = nn.ModuleList(mlp(hidden, hidden, 1) for _ in range(layers))
        self.acceleration = mlp(hidden, hidden, dimensions)

    def predict_next(self, history: torch.Tensor, particle_type: torch.Tensor, graph) -> torch.Tensor:
        features, velocity = _particle_inputs(
            history, particle_type, self.types, self.velocity_mean, self.velocity_std,
            self.acceleration_mean, self.acceleration_std,
        )
        nodes = self.node_encoder(features)
        coordinates = history[-1]
        source, target = graph.source, graph.target
        for edge_mlp, node_mlp, coordinate_mlp in zip(self.edge_mlps, self.node_mlps, self.coordinate_mlps):
            relative = coordinates[source] - coordinates[target]
            squared = relative.square().sum(-1, keepdim=True)
            edge = edge_mlp(torch.cat((nodes[source], nodes[target], squared), -1))
            aggregate = scatter_add(edge, target, len(nodes))
            nodes = nodes + node_mlp(torch.cat((nodes, aggregate), -1))
            scale = 0.01 * torch.tanh(coordinate_mlp(edge))
            coordinates = coordinates + scatter_add(scale * relative, target, len(nodes))
        acceleration = self._decode_acceleration(self.acceleration(nodes))
        return coordinates + velocity + acceleration


class ParticleMeshGraphNet(GraphNetworkSimulator):
    """MeshGraphNet encode-process-decode applied to dynamically built particle graphs."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("message_steps", 4)
        super().__init__(**kwargs)
