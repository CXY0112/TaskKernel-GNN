"""Generate the controlled D2 reaction–diffusion graphs and trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


DT = 0.02
D = 0.10
RHO1 = 0.50
RHO2 = 0.65
BETA = 6.0


@dataclass
class Graph:
    kind: str
    coordinates: torch.Tensor
    mask: torch.Tensor
    distance: torch.Tensor
    weight: torch.Tensor


def factor_grid(n: int) -> tuple[int, int]:
    rows = int(math.sqrt(n))
    while n % rows:
        rows -= 1
    return rows, n // rows


def connect_components(mask: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    n = mask.shape[0]
    while True:
        seen = {0}
        frontier = [0]
        while frontier:
            node = frontier.pop()
            neighbors = torch.where(mask[node])[0].tolist()
            for neighbor in neighbors:
                if neighbor not in seen:
                    seen.add(neighbor)
                    frontier.append(neighbor)
        if len(seen) == n:
            return mask
        left = torch.tensor(sorted(seen), dtype=torch.long)
        right = torch.tensor(sorted(set(range(n)) - seen), dtype=torch.long)
        cross = distance[left][:, right]
        flat = int(cross.argmin())
        i = int(left[flat // len(right)])
        j = int(right[flat % len(right)])
        mask[i, j] = True
        mask[j, i] = True


def make_graph(kind: str, n: int, seed: int) -> Graph:
    generator = torch.Generator().manual_seed(seed)
    if kind == "ring":
        theta = 2.0 * torch.pi * torch.arange(n) / n
        coordinates = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        mask = torch.zeros(n, n, dtype=torch.bool)
        indices = torch.arange(n)
        mask[indices, (indices + 1) % n] = True
        mask[indices, (indices - 1) % n] = True
    elif kind == "grid":
        rows, columns = factor_grid(n)
        yy, xx = torch.meshgrid(torch.linspace(0, 1, rows), torch.linspace(0, 1, columns), indexing="ij")
        coordinates = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        mask = torch.zeros(n, n, dtype=torch.bool)
        for row in range(rows):
            for column in range(columns):
                index = row * columns + column
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = row + dr, column + dc
                    if 0 <= rr < rows and 0 <= cc < columns:
                        mask[index, rr * columns + cc] = True
    elif kind == "rgg":
        coordinates = torch.rand(n, 2, generator=generator)
        distance = torch.cdist(coordinates, coordinates)
        radius = 1.55 * math.sqrt(6.0 / (math.pi * n))
        mask = (distance <= radius) & ~torch.eye(n, dtype=torch.bool)
        mask = connect_components(mask, distance)
    else:
        raise ValueError(kind)

    distance = torch.cdist(coordinates, coordinates)
    weight = torch.exp(-BETA * distance.square()) * mask
    return Graph(kind=kind, coordinates=coordinates, mask=mask, distance=distance, weight=weight)


def initial_state(graph: Graph, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    coordinates = graph.coordinates
    edge_scale = graph.distance[graph.mask].median()
    field = torch.zeros(len(coordinates))
    for _ in range(3):
        center = coordinates[torch.randint(len(coordinates), (), generator=generator)]
        amplitude = 0.35 * torch.randn((), generator=generator)
        width = edge_scale * (1.5 + 3.0 * torch.rand((), generator=generator))
        field += amplitude * torch.exp(-((coordinates - center).square().sum(-1)) / (2.0 * width.square()))
    pulse_center = coordinates[torch.randint(len(coordinates), (), generator=generator)]
    pulse_width = edge_scale * (0.4 + 0.6 * torch.rand((), generator=generator))
    pulse = (0.4 + 0.5 * torch.rand((), generator=generator)) * torch.exp(
        -((coordinates - pulse_center).square().sum(-1)) / (2.0 * pulse_width.square())
    )
    return torch.clamp(field + pulse, -1.2, 1.2)


def rhs(state: torch.Tensor, graph: Graph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    difference = state.unsqueeze(0) - state.unsqueeze(1)
    diffusion = D * (graph.weight * difference).sum(dim=1)
    reaction = RHO1 * state - RHO2 * state.pow(3)
    return diffusion + reaction, diffusion, reaction


def rollout(graph: Graph, seed: int, steps: int = 200) -> tuple[torch.Tensor, dict[str, float]]:
    states = [initial_state(graph, seed)]
    conservation_residual = 0.0
    for _ in range(steps):
        derivative, diffusion, _ = rhs(states[-1], graph)
        conservation_residual = max(conservation_residual, abs(float(diffusion.sum())) / len(diffusion))
        states.append(states[-1] + DT * derivative)
    trajectory = torch.stack(states)
    degree = graph.mask.sum(dim=1).float()
    metrics = {
        "nodes": float(len(degree)),
        "edges": float(graph.mask.sum() // 2),
        "mean_degree": float(degree.mean()),
        "max_degree": float(degree.max()),
        "connected": float(is_connected(graph.mask)),
        "finite": float(torch.isfinite(trajectory).all()),
        "max_abs_state": float(trajectory.abs().max()),
        "diffusion_conservation_residual": conservation_residual,
        "initial_rms": float(trajectory[0].square().mean().sqrt()),
        "final_rms": float(trajectory[-1].square().mean().sqrt()),
    }
    return trajectory, metrics


def is_connected(mask: torch.Tensor) -> bool:
    seen = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbor in torch.where(mask[node])[0].tolist():
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return len(seen) == mask.shape[0]
