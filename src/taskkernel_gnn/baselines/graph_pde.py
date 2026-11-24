"""Graph baselines for time-dependent PDE experiments.

The implementations preserve the defining computation of each published
architecture while exposing the same ``step(state, sample)`` interface used by
the TaskKernel experiments.  They do not use analytic TaskKernel candidates.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def scatter_add(values: torch.Tensor, destination: torch.Tensor, nodes: int) -> torch.Tensor:
    shape = (nodes,) + tuple(values.shape[1:])
    result = values.new_zeros(shape)
    result.index_add_(0, destination, values)
    return result


def mlp(input_features: int, hidden_features: int, output_features: int, *, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_features
    for _ in range(max(depth - 1, 1)):
        layers.extend([nn.Linear(current, hidden_features), nn.SiLU()])
        current = hidden_features
    layers.append(nn.Linear(current, output_features))
    network = nn.Sequential(*layers)
    for layer in network.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=0.35)
            nn.init.zeros_(layer.bias)
    return network


def graph_indices(sample) -> tuple[torch.Tensor, torch.Tensor]:
    """Return directed edge indices for either the generic or cardiac sample API."""
    source = sample.src if hasattr(sample, "src") else sample.source
    target = sample.dst if hasattr(sample, "dst") else sample.target
    return source, target


def edge_geometry(sample, coordinate_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = sample.coordinates[:, :coordinate_dim]
    source, target = graph_indices(sample)
    relative = coordinates[source] - coordinates[target]
    if hasattr(sample, "distance"):
        distance = sample.distance
    else:
        distance = sample.inverse_length2.clamp_min(1e-12).rsqrt()
    if hasattr(sample, "weight"):
        edge_attribute = sample.weight
    elif hasattr(sample, "alignment"):
        edge_attribute = sample.alignment
    else:
        edge_attribute = torch.ones_like(distance)
    degree_scale = sample.degree.clamp_min(1).log1p()
    edge = torch.cat([
        relative,
        distance[:, None],
        edge_attribute[:, None],
        degree_scale[source, None],
        degree_scale[target, None],
    ], dim=-1)
    node = torch.cat([coordinates, degree_scale[:, None]], dim=-1)
    return edge, node


class MeshGraphBlock(nn.Module):
    def __init__(self, hidden_features: int) -> None:
        super().__init__()
        self.edge_update = mlp(3 * hidden_features, hidden_features, hidden_features)
        self.node_update = mlp(2 * hidden_features, hidden_features, hidden_features)
        self.edge_norm = nn.LayerNorm(hidden_features)
        self.node_norm = nn.LayerNorm(hidden_features)

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor, sample) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = graph_indices(sample)
        edge_delta = self.edge_update(torch.cat([nodes[source], nodes[target], edges], dim=-1))
        edges = self.edge_norm(edges + edge_delta)
        aggregate = scatter_add(edges, target, sample.nodes)
        node_delta = self.node_update(torch.cat([nodes, aggregate], dim=-1))
        nodes = self.node_norm(nodes + node_delta)
        return nodes, edges


class MeshGraphNet(nn.Module):
    """Encoder-processor-decoder MeshGraphNet baseline."""

    def __init__(self, *, hidden_features: int = 4, processor_steps: int = 1,
                 coordinate_dim: int = 2, state_features: int = 1, dt: float = 0.02) -> None:
        super().__init__()
        self.coordinate_dim = coordinate_dim
        self.dt = dt
        self.state_features = state_features
        self.node_encoder = mlp(state_features + coordinate_dim + 1, hidden_features, hidden_features)
        self.edge_encoder = mlp(coordinate_dim + 4, hidden_features, hidden_features)
        self.processor = nn.ModuleList(MeshGraphBlock(hidden_features) for _ in range(processor_steps))
        self.decoder = mlp(hidden_features, hidden_features, state_features)

    def step(self, state: torch.Tensor, sample, time_index: int | None = None) -> torch.Tensor:
        edge_features, node_geometry = edge_geometry(sample, self.coordinate_dim)
        state_matrix = state[:, None] if state.ndim == 1 else state
        nodes = self.node_encoder(torch.cat([state_matrix, node_geometry], dim=-1))
        edges = self.edge_encoder(edge_features)
        for block in self.processor:
            nodes, edges = block(nodes, edges, sample)
        rate = 2.0 * torch.tanh(self.decoder(nodes))
        result = state_matrix + self.dt * rate
        return result.squeeze(-1) if state.ndim == 1 else result


class GraphOperatorLayer(nn.Module):
    """Factorized graph integral-kernel layer used by the GNO baseline."""

    def __init__(self, hidden_features: int, edge_features: int) -> None:
        super().__init__()
        self.kernel = mlp(edge_features, hidden_features, hidden_features)
        self.value = nn.Linear(hidden_features, hidden_features, bias=False)
        self.local = nn.Linear(hidden_features, hidden_features)
        self.norm = nn.LayerNorm(hidden_features)

    def forward(self, nodes: torch.Tensor, edge_input: torch.Tensor, sample) -> torch.Tensor:
        source, target = graph_indices(sample)
        kernel = torch.tanh(self.kernel(edge_input))
        difference = self.value(nodes[source] - nodes[target])
        integral = scatter_add(kernel * difference, target, sample.nodes)
        return self.norm(nodes + torch.tanh(self.local(nodes) + integral))


class GraphNeuralOperator(nn.Module):
    """Graph neural operator with learned edge-conditioned integral kernels."""

    def __init__(self, *, hidden_features: int = 4, operator_layers: int = 1,
                 coordinate_dim: int = 2, state_features: int = 1, dt: float = 0.02) -> None:
        super().__init__()
        self.coordinate_dim = coordinate_dim
        self.dt = dt
        self.state_features = state_features
        node_features = state_features + coordinate_dim + 1
        edge_features = coordinate_dim + 4 + 2 * state_features
        self.lift = mlp(node_features, hidden_features, hidden_features)
        self.layers = nn.ModuleList(
            GraphOperatorLayer(hidden_features, edge_features) for _ in range(operator_layers)
        )
        self.projection = mlp(hidden_features, hidden_features, state_features)

    def step(self, state: torch.Tensor, sample, time_index: int | None = None) -> torch.Tensor:
        source, target = graph_indices(sample)
        geometry, node_geometry = edge_geometry(sample, self.coordinate_dim)
        state_matrix = state[:, None] if state.ndim == 1 else state
        state_edge = torch.cat([state_matrix[source], state_matrix[target]], dim=-1)
        edge_input = torch.cat([geometry, state_edge], dim=-1)
        nodes = self.lift(torch.cat([state_matrix, node_geometry], dim=-1))
        for layer in self.layers:
            nodes = layer(nodes, edge_input, sample)
        rate = 2.0 * torch.tanh(self.projection(nodes))
        result = state_matrix + self.dt * rate
        return result.squeeze(-1) if state.ndim == 1 else result


class PDEMessageBlock(nn.Module):
    def __init__(self, hidden_features: int, geometry_features: int, state_features: int) -> None:
        super().__init__()
        self.message = mlp(2 * hidden_features + geometry_features + state_features, hidden_features, hidden_features)
        self.update = mlp(2 * hidden_features, hidden_features, hidden_features)
        self.norm = nn.LayerNorm(hidden_features)

    def forward(self, nodes: torch.Tensor, state: torch.Tensor, geometry: torch.Tensor, sample) -> torch.Tensor:
        source, target = graph_indices(sample)
        difference = state[source] - state[target]
        message = self.message(torch.cat([
            nodes[source], nodes[target], geometry, difference
        ], dim=-1))
        aggregate = scatter_add(message, target, sample.nodes)
        return self.norm(nodes + self.update(torch.cat([nodes, aggregate], dim=-1)))


class MessagePassingPDESolver(nn.Module):
    """Autoregressive message-passing neural PDE solver baseline."""

    def __init__(self, *, hidden_features: int = 4, message_steps: int = 2,
                 coordinate_dim: int = 2, state_features: int = 1, dt: float = 0.02) -> None:
        super().__init__()
        self.coordinate_dim = coordinate_dim
        self.dt = dt
        geometry_features = coordinate_dim + 4
        self.state_features = state_features
        self.encoder = mlp(state_features + coordinate_dim + 1, hidden_features, hidden_features)
        self.blocks = nn.ModuleList(
            PDEMessageBlock(hidden_features, geometry_features, state_features) for _ in range(message_steps)
        )
        self.decoder = mlp(hidden_features, hidden_features, state_features)

    def step(self, state: torch.Tensor, sample, time_index: int | None = None) -> torch.Tensor:
        geometry, node_geometry = edge_geometry(sample, self.coordinate_dim)
        state_matrix = state[:, None] if state.ndim == 1 else state
        nodes = self.encoder(torch.cat([state_matrix, node_geometry], dim=-1))
        for block in self.blocks:
            nodes = block(nodes, state_matrix, geometry, sample)
        rate = 2.0 * torch.tanh(self.decoder(nodes))
        result = state_matrix + self.dt * rate
        return result.squeeze(-1) if state.ndim == 1 else result


@dataclass(frozen=True)
class BaselineConfiguration:
    hidden_features: int
    processor_steps: int


CONFIGURATIONS = {
    "matched": {
        "meshgraphnet": BaselineConfiguration(4, 1),
        "gno": BaselineConfiguration(4, 1),
        "mp_pde": BaselineConfiguration(4, 1),
    },
    "canonical": {
        "meshgraphnet": BaselineConfiguration(32, 4),
        "gno": BaselineConfiguration(32, 4),
        "mp_pde": BaselineConfiguration(32, 6),
    },
}
CONFIGURATIONS["canonical_extended"] = CONFIGURATIONS["canonical"]


def build_graph_pde_baseline(name: str, track: str, *, coordinate_dim: int = 2,
                             state_features: int = 1, dt: float = 0.02) -> nn.Module:
    if track not in CONFIGURATIONS or name not in CONFIGURATIONS[track]:
        raise ValueError(f"unknown baseline configuration: {name}/{track}")
    configuration = CONFIGURATIONS[track][name]
    common = dict(hidden_features=configuration.hidden_features, coordinate_dim=coordinate_dim,
                  state_features=state_features, dt=dt)
    if name == "meshgraphnet":
        return MeshGraphNet(processor_steps=configuration.processor_steps, **common)
    if name == "gno":
        return GraphNeuralOperator(operator_layers=configuration.processor_steps, **common)
    if name == "mp_pde":
        return MessagePassingPDESolver(message_steps=configuration.processor_steps, **common)
    raise ValueError(name)
