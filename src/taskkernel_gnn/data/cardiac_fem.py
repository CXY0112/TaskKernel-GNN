"""Lumped P1 tetrahedral diffusion prior in mm and ms.

This adapter is not the full-mass openCARP reference integrator. Negative
individual off-diagonal weights are retained; the assembled stiffness matrix
is positive semidefinite. It implements natural zero-flux boundaries.
"""
from __future__ import annotations
import numpy as np
import torch
from .graph import GraphSample


def monodomain_diffusivities(parameters):
    """Convert S/m to mm²/ms using beta=.14/um and Cm=1 uF/cm²."""
    intracellular = np.array([parameters[k] for k in ("g_il", "g_it", "g_in")])
    extracellular = np.array([.62, .24, .24])
    effective = intracellular * extracellular / (intracellular + extracellular)
    return effective / 1.4  # .001 S/mm / (140/mm * 1e-8 F/mm²) / 1000


def anisotropic_fem_graph(trajectory) -> GraphSample:
    points = trajectory.points.numpy().astype(np.float64)
    if trajectory.parameters.get("coordinate_unit", "um") == "um":
        points *= .001
    tetrahedra = trajectory.tetrahedra.numpy()
    vertices = points[tetrahedra]
    first, second, third = [vertices[:, i] - vertices[:, 0] for i in (1, 2, 3)]
    cross = np.cross(second, third)
    determinant = np.einsum("ti,ti->t", first, cross)
    volume = np.abs(determinant) / 6
    if np.any(volume <= 1e-15):
        raise ValueError("degenerate tetrahedron")
    other_gradients = np.stack([cross, np.cross(third, first), np.cross(first, second)], 1) / determinant[:, None, None]
    gradients = np.concatenate([-other_gradients.sum(1, keepdims=True), other_gradients], 1)
    diffusion = monodomain_diffusivities(trajectory.parameters)
    stiffness = np.einsum("tik,k,tjk,t->tij", gradients, diffusion, gradients, volume)
    nodes = len(points)
    mass = np.zeros(nodes)
    np.add.at(mass, tetrahedra.ravel(), np.repeat(volume / 4, 4))
    if np.any(mass <= 0):
        raise ValueError("isolated FEM node")
    row = np.repeat(tetrahedra, 4, axis=1).ravel()
    col = np.tile(tetrahedra, (1, 4)).ravel()
    keys, inverse = np.unique(row * nodes + col, return_inverse=True)
    values = np.bincount(inverse, weights=stiffness.ravel())
    targets, sources = keys // nodes, keys % nodes
    keep = targets != sources
    targets, sources, values = targets[keep], sources[keep], values[keep]
    weight = -values / mass[targets]
    coordinates = torch.tensor(points, dtype=torch.float32)
    source, target = torch.from_numpy(sources).long(), torch.from_numpy(targets).long()
    distance = (coordinates[source] - coordinates[target]).norm(dim=-1)
    graph = GraphSample(coordinates, source, target, distance,
                        torch.tensor(weight, dtype=torch.float32),
                        torch.bincount(target, minlength=nodes).float())
    graph.context.update(position=coordinates, edge_weight=graph.weight,
                         fem_mass=torch.tensor(mass, dtype=torch.float64),
                         diffusion_mm2_per_ms=torch.tensor(diffusion, dtype=torch.float32),
                         parameters=trajectory.parameters, coordinate_unit="mm")
    return graph


def diffusion_rate(state, graph):
    output = torch.zeros_like(state)
    output.index_add_(0, graph.target,
                      graph.weight[:, None] * (state[graph.source] - state[graph.target]))
    return output


def conditioned_history(voltage_history, graph, time_ms, interval_ms=1.):
    """Shared causal features; history is supplied by observations or predictions.

    Input histories are normalized as (Vm + 85)/100. No future voltage or LAT
    enters this function. The known pulse overlap over the next interval is a
    forcing covariate, available equally to every model.
    """
    p = graph.context["parameters"]
    coordinates = graph.coordinates
    lo = coordinates.new_tensor([p.get(f"stimulus_p0_{i}_um", 0) / 1000 for i in range(3)])
    hi = coordinates.new_tensor([p.get(f"stimulus_p1_{i}_um", 1500) / 1000 for i in range(3)])
    mask = ((coordinates >= lo - 1e-6) & (coordinates <= hi + 1e-6)).all(-1).float()
    duration = p["stimulus_duration_ms"]
    overlap = max(0., min(time_ms + interval_ms, duration) - max(time_ms, 0.)) / interval_ms
    scalars = [p["g_il"] / .17, p["g_it"] / .019, p["g_in"] / .019,
               p["stimulus_strength"] / 35.71, duration / 2., time_ms / 150.]
    conditions = coordinates.new_tensor(scalars).expand(graph.nodes, -1)
    return torch.cat([voltage_history, conditions, mask[:, None],
                      (mask * overlap * p["stimulus_strength"] / 35.71)[:, None]], -1)
