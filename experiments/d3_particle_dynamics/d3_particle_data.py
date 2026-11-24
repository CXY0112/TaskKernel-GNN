"""Generate controlled D3 particle systems and reference forces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist



G_DEFAULT = 0.002
K_DEFAULT = 0.80
GAMMA_DEFAULT = 0.18
DT_DEFAULT = 0.01
EPS_DEFAULT = 0.10
HORIZON_DEFAULT = 4.0
MIN_INITIAL_SEPARATION = 0.34
COLLISION_DISTANCE = 0.075
EXPLOSION_POSITION = 100.0
EXPLOSION_SPEED = 100.0


@dataclass
class System:
    position: np.ndarray
    velocity: np.ndarray
    mass: np.ndarray
    edges: np.ndarray
    rest_length: np.ndarray
    spring_k: np.ndarray
    dashpot_gamma: np.ndarray
    gravity_g: float


def sample_positions(rng: np.random.Generator, n: int) -> np.ndarray:
    """Poisson-like sampling at approximately fixed density."""
    extent = 1.15 * np.sqrt(n / 8.0)
    points: list[np.ndarray] = []
    attempts = 0
    while len(points) < n and attempts < 100_000:
        candidate = rng.uniform(-extent, extent, size=2)
        if not points or np.min(np.linalg.norm(np.asarray(points) - candidate, axis=1)) >= MIN_INITIAL_SEPARATION:
            points.append(candidate)
        attempts += 1
    if len(points) != n:
        raise RuntimeError(f"could not sample {n} separated particles")
    return np.asarray(points, dtype=np.float64)


def random_connected_edges(position: np.ndarray, rng: np.random.Generator, target_degree: float = 2.8) -> np.ndarray:
    n = len(position)
    distance = cdist(position, position)
    np.fill_diagonal(distance, np.inf)
    mst = minimum_spanning_tree(distance).tocoo()
    edges = {tuple(sorted((int(i), int(j)))) for i, j in zip(mst.row, mst.col)}
    candidates = [(distance[i, j], i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in edges]
    rng.shuffle(candidates)
    candidates.sort(key=lambda item: item[0] * rng.uniform(0.85, 1.15))
    target = max(n - 1, int(round(target_degree * n / 2)))
    for _, i, j in candidates:
        if len(edges) >= target:
            break
        edges.add((i, j))
    return np.asarray(sorted(edges), dtype=np.int64)


def topology_edges(position: np.ndarray, rng: np.random.Generator, topology: str) -> np.ndarray:
    n = len(position)
    if topology == "random":
        return random_connected_edges(position, rng)
    if topology == "chain":
        remaining = set(range(n))
        current = int(np.argmin(position[:, 0]))
        order = [current]
        remaining.remove(current)
        while remaining:
            current = min(remaining, key=lambda i: np.linalg.norm(position[i] - position[current]))
            order.append(current)
            remaining.remove(current)
        order = np.asarray(order)
        pairs = [(int(order[i]), int(order[i + 1])) for i in range(n - 1)]
    elif topology == "ring":
        angle = np.arctan2(position[:, 1], position[:, 0])
        order = np.argsort(angle)
        pairs = [(int(order[i]), int(order[(i + 1) % n])) for i in range(n)]
    elif topology == "branched":
        center = int(np.argmin(np.linalg.norm(position, axis=1)))
        others = [int(i) for i in np.argsort(np.linalg.norm(position - position[center], axis=1)) if i != center]
        pairs = [(center, i) for i in others[: min(4, len(others))]]
        attached = {center, *(i for pair in pairs for i in pair)}
        for node in others:
            if node in attached:
                continue
            parent = min(attached, key=lambda i: np.linalg.norm(position[node] - position[i]))
            pairs.append((parent, node))
            attached.add(node)
    else:
        raise ValueError(topology)
    return np.asarray(sorted({tuple(sorted(pair)) for pair in pairs}), dtype=np.int64)


def make_system(seed: int, n: int, topology: str = "random") -> System:
    rng = np.random.default_rng(seed)
    position = sample_positions(rng, n)
    mass = rng.uniform(0.8, 1.2, size=n)
    center_of_mass = np.sum(mass[:, None] * position, axis=0) / mass.sum()
    position -= center_of_mass
    velocity = rng.normal(0.0, 0.025, size=(n, 2))
    velocity -= np.sum(mass[:, None] * velocity, axis=0) / mass.sum()
    edges = topology_edges(position, rng, topology)
    edge_distance = np.linalg.norm(position[edges[:, 1]] - position[edges[:, 0]], axis=1)
    # Non-equilibrium rest lengths make every force family observable at t=0.
    rest_length = edge_distance * rng.uniform(0.95, 1.05, size=len(edges))
    spring_k = K_DEFAULT * rng.uniform(0.85, 1.15, size=len(edges))
    dashpot_gamma = GAMMA_DEFAULT * rng.uniform(0.85, 1.15, size=len(edges))
    return System(position, velocity, mass, edges, rest_length, spring_k, dashpot_gamma, G_DEFAULT)


def forces(system: System, position: np.ndarray, velocity: np.ndarray, epsilon: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = len(position)
    gravity = np.zeros_like(position)
    spring = np.zeros_like(position)
    dashpot = np.zeros_like(position)
    for i in range(n):
        for j in range(i + 1, n):
            delta = position[j] - position[i]
            distance2 = float(delta @ delta)
            pair = system.gravity_g * system.mass[i] * system.mass[j] * delta / (distance2 + epsilon**2) ** 1.5
            gravity[i] += pair
            gravity[j] -= pair
    for edge_index, (i, j) in enumerate(system.edges):
        delta = position[j] - position[i]
        distance = max(float(np.linalg.norm(delta)), 1e-12)
        direction = delta / distance
        spring_pair = system.spring_k[edge_index] * (distance - system.rest_length[edge_index]) * direction
        relative_radial_speed = float((velocity[j] - velocity[i]) @ direction)
        dashpot_pair = system.dashpot_gamma[edge_index] * relative_radial_speed * direction
        spring[i] += spring_pair
        spring[j] -= spring_pair
        dashpot[i] += dashpot_pair
        dashpot[j] -= dashpot_pair
    total = gravity + spring + dashpot
    return total, {"gravity": gravity, "spring": spring, "dashpot": dashpot}
