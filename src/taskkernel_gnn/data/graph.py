"""Small dependency-free graph containers and constructors for external data."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class GraphSample:
    coordinates: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    distance: torch.Tensor
    weight: torch.Tensor
    degree: torch.Tensor
    trajectory: torch.Tensor | None = None
    context: dict[str, torch.Tensor | float | int] = field(default_factory=dict)

    @property
    def nodes(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def edge_index(self) -> torch.Tensor:
        return torch.stack((self.source, self.target))


def _finish_graph(coordinates: torch.Tensor, source: torch.Tensor, target: torch.Tensor,
                  weight: torch.Tensor | None = None) -> GraphSample:
    delta = coordinates[source] - coordinates[target]
    distance = delta.square().sum(-1).clamp_min(1e-24).sqrt()
    if weight is None:
        weight = torch.ones_like(distance)
    degree = torch.bincount(target, minlength=len(coordinates)).to(coordinates.dtype)
    return GraphSample(coordinates, source, target, distance, weight, degree)


def grid_graph(x: torch.Tensor, y: torch.Tensor) -> GraphSample:
    """Four-neighbour directed graph for a Cartesian ``x``/``y`` grid."""

    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coordinates = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    height, width = len(y), len(x)
    ids = torch.arange(height * width).reshape(height, width)
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    if width > 1:
        pairs.extend(((ids[:, :-1].reshape(-1), ids[:, 1:].reshape(-1)),
                      (ids[:, 1:].reshape(-1), ids[:, :-1].reshape(-1))))
    if height > 1:
        pairs.extend(((ids[:-1, :].reshape(-1), ids[1:, :].reshape(-1)),
                      (ids[1:, :].reshape(-1), ids[:-1, :].reshape(-1))))
    source = torch.cat([pair[0] for pair in pairs]).long()
    target = torch.cat([pair[1] for pair in pairs]).long()
    spacing2 = (coordinates[source] - coordinates[target]).square().sum(-1).clamp_min(1e-24)
    return _finish_graph(coordinates, source, target, spacing2.reciprocal())


def radius_graph(coordinates: torch.Tensor, radius: float, *, max_neighbors: int | None = None) -> GraphSample:
    """Directed radius graph using PyTorch only.

    The implementation is intended for the 1k-particle WaterDrop benchmark and
    avoids an optional torch-cluster dependency.  Neighbor truncation is
    deterministic and selects the closest particles.
    """

    if radius <= 0:
        raise ValueError("radius must be positive")
    if coordinates.device.type == "cpu":
        try:
            import numpy as np
            from scipy.spatial import cKDTree
        except ImportError:
            pass
        else:
            points = coordinates.detach().numpy()
            pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
            sources = np.concatenate((pairs[:, 0], pairs[:, 1]))
            targets = np.concatenate((pairs[:, 1], pairs[:, 0]))
            if max_neighbors is not None and len(targets):
                counts = np.bincount(targets, minlength=len(points))
                if counts.max() > max_neighbors:
                    distances = ((points[sources] - points[targets]) ** 2).sum(axis=1)
                    order = np.lexsort((sources, distances, targets))
                    sorted_targets = targets[order]
                    _, first, inverse = np.unique(
                        sorted_targets, return_index=True, return_inverse=True,
                    )
                    rank = np.arange(len(order)) - first[inverse]
                    kept = order[rank < max_neighbors]
                    sources, targets = sources[kept], targets[kept]
            order = np.lexsort((sources, targets))
            return _finish_graph(
                coordinates,
                torch.from_numpy(sources[order].copy()).long(),
                torch.from_numpy(targets[order].copy()).long(),
            )
    distances = torch.cdist(coordinates, coordinates)
    mask = (distances <= radius) & ~torch.eye(len(coordinates), dtype=torch.bool, device=coordinates.device)
    targets: list[torch.Tensor] = []
    sources: list[torch.Tensor] = []
    for target in range(len(coordinates)):
        source = torch.nonzero(mask[target], as_tuple=False).flatten()
        if max_neighbors is not None and len(source) > max_neighbors:
            order = torch.argsort(distances[target, source])[:max_neighbors]
            source = source[order]
        sources.append(source)
        targets.append(torch.full_like(source, target))
    source = torch.cat(sources) if sources else torch.empty(0, dtype=torch.long)
    target = torch.cat(targets) if targets else torch.empty(0, dtype=torch.long)
    return _finish_graph(coordinates, source.long(), target.long())


def tetrahedral_edges(points: torch.Tensor, tetrahedra: torch.Tensor) -> GraphSample:
    """Unique bidirectional graph edges induced by tetrahedral elements."""

    local = torch.tensor(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=torch.long)
    edges = tetrahedra[:, local].reshape(-1, 2)
    edges = torch.sort(edges, dim=1).values.unique(dim=0)
    source = torch.cat((edges[:, 0], edges[:, 1]))
    target = torch.cat((edges[:, 1], edges[:, 0]))
    return _finish_graph(points, source, target)
