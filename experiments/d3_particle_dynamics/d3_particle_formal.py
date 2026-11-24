"""Controlled TaskKernel-GNN D3 experiment on mixed-force particle dynamics."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from d3_particle_data import (
    DT_DEFAULT,
    EPS_DEFAULT,
    G_DEFAULT,
    HORIZON_DEFAULT,
    System,
    forces as numpy_forces,
    make_system,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "Ablation" / "d3" / "particles" / "formal"
CHECKPOINTS = RESULTS / "checkpoints"
sys.path.insert(0, str(ROOT / "src"))
from taskkernel_gnn.core.gating import annealed_temperature, temperature_softmax
from taskkernel_gnn.kernels.message.mechanics import dense_particle_components
DT = DT_DEFAULT
EPSILON = EPS_DEFAULT
ETA = 0.1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Trajectory:
    name: str
    system: System
    position: np.ndarray
    velocity: np.ndarray
    force: np.ndarray


def alter_system(system: System, seed: int, mode: str) -> System:
    system = copy.deepcopy(system)
    rng = np.random.default_rng(seed + 930_000)
    if mode == "parameter":
        system.gravity_g *= 1.70
        system.spring_k *= 1.50
        system.dashpot_gamma *= 1.60
        system.mass *= rng.uniform(0.70, 1.35, size=len(system.mass))
    elif mode == "initial":
        system.velocity *= 2.2
        system.velocity -= np.sum(system.mass[:, None] * system.velocity, axis=0) / system.mass.sum()
        distance = np.linalg.norm(system.position[system.edges[:, 1]] - system.position[system.edges[:, 0]], axis=1)
        system.rest_length = distance * rng.uniform(0.90, 1.10, size=len(system.edges))
    elif mode == "se2":
        theta = rng.uniform(-np.pi, np.pi)
        orthogonal = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        if seed % 2:
            orthogonal = np.diag([-1.0, 1.0]) @ orthogonal
        system.position = system.position @ orthogonal.T + rng.uniform(-1.5, 1.5, size=2)
        system.velocity = system.velocity @ orthogonal.T
    elif mode != "nominal":
        raise ValueError(mode)
    return system


def generate_trajectory(name: str, seed: int, n: int, topology: str, horizon: float, mode: str) -> Trajectory:
    system = make_system(seed, n, topology)
    # Small nominal task-to-task parameter variation prevents memorizing one constant.
    rng = np.random.default_rng(seed + 510_000)
    system.gravity_g *= rng.uniform(0.85, 1.15)
    system.spring_k *= rng.uniform(0.90, 1.10, size=len(system.edges))
    system.dashpot_gamma *= rng.uniform(0.90, 1.10, size=len(system.edges))
    system = alter_system(system, seed, mode)
    steps = int(round(horizon / DT))
    position = np.empty((steps + 1, n, 2), dtype=np.float32)
    velocity = np.empty_like(position)
    force = np.empty((steps, n, 2), dtype=np.float32)
    position[0] = system.position
    velocity[0] = system.velocity
    x = system.position.copy()
    v = system.velocity.copy()
    for step in range(steps):
        f, _ = numpy_forces(system, x, v, EPSILON)
        force[step] = f
        v = v + DT * f / system.mass[:, None]
        x = x + DT * v
        position[step + 1] = x
        velocity[step + 1] = v
    return Trajectory(name, system, position, velocity, force)


def make_splits(seed_offset: int = 0) -> dict[str, list[Trajectory]]:
    splits: dict[str, list[Trajectory]] = {"Train": [], "Validation": []}
    for index in range(12):
        n = 6 if index % 2 == 0 else 8
        splits["Train"].append(generate_trajectory("Train", 10_000 + seed_offset + index, n, "random", HORIZON_DEFAULT, "nominal"))
    for index in range(4):
        n = 6 if index % 2 == 0 else 8
        splits["Validation"].append(generate_trajectory("Validation", 20_000 + seed_offset + index, n, "random", HORIZON_DEFAULT, "nominal"))
    specifications = {
        "ID": [(6, "random", HORIZON_DEFAULT, "nominal"), (8, "random", HORIZON_DEFAULT, "nominal")],
        "SizeOOD-N12": [(12, "random", HORIZON_DEFAULT, "nominal")],
        "SizeOOD-N20": [(20, "random", HORIZON_DEFAULT, "nominal")],
        "ParameterOOD": [(8, "random", HORIZON_DEFAULT, "parameter")],
        "TopologyOOD-chain": [(8, "chain", HORIZON_DEFAULT, "nominal")],
        "TopologyOOD-ring": [(8, "ring", HORIZON_DEFAULT, "nominal")],
        "TopologyOOD-branched": [(8, "branched", HORIZON_DEFAULT, "nominal")],
        "SE2-OOD": [(8, "random", HORIZON_DEFAULT, "se2")],
        "InitialStateOOD": [(8, "random", HORIZON_DEFAULT, "initial")],
        "LongRolloutOOD": [(8, "random", 2 * HORIZON_DEFAULT, "nominal")],
    }
    scenario_index = 0
    for split, variants in specifications.items():
        splits[split] = []
        for n, topology, horizon, mode in variants:
            for repetition in range(4):
                seed = 30_000 + seed_offset + 100 * scenario_index + repetition
                splits[split].append(generate_trajectory(split, seed, n, topology, horizon, mode))
            scenario_index += 1
    return splits


def system_matrices(system: System) -> dict[str, torch.Tensor]:
    n = len(system.mass)
    bond = np.zeros((n, n), dtype=np.float32)
    rest = np.zeros((n, n), dtype=np.float32)
    spring = np.zeros((n, n), dtype=np.float32)
    dashpot = np.zeros((n, n), dtype=np.float32)
    for edge_index, (i, j) in enumerate(system.edges):
        bond[i, j] = bond[j, i] = 1.0
        rest[i, j] = rest[j, i] = system.rest_length[edge_index]
        spring[i, j] = spring[j, i] = system.spring_k[edge_index]
        dashpot[i, j] = dashpot[j, i] = system.dashpot_gamma[edge_index]
    return {
        "mass": torch.tensor(system.mass, dtype=torch.float32),
        "bond": torch.tensor(bond),
        "rest": torch.tensor(rest),
        "spring": torch.tensor(spring),
        "dashpot": torch.tensor(dashpot),
        "gravity_g": torch.tensor(system.gravity_g, dtype=torch.float32),
    }


def training_groups(trajectories: list[Trajectory], stride: int = 5) -> dict[int, dict[str, torch.Tensor]]:
    groups: dict[int, list[dict[str, torch.Tensor]]] = {}
    for trajectory in trajectories:
        matrices = system_matrices(trajectory.system)
        for step in range(0, len(trajectory.force), stride):
            item = {
                "position": torch.tensor(trajectory.position[step]),
                "velocity": torch.tensor(trajectory.velocity[step]),
                "next_position": torch.tensor(trajectory.position[step + 1]),
                "next_velocity": torch.tensor(trajectory.velocity[step + 1]),
                "target_force": torch.tensor(trajectory.force[step]),
                **matrices,
            }
            groups.setdefault(len(trajectory.system.mass), []).append(item)
    result = {}
    for n, items in groups.items():
        result[n] = {key: torch.stack([item[key] for item in items]) for key in items[0]}
    return result


def slice_batch(group: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value[indices] for key, value in group.items()}


def pair_geometry(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    position, velocity = batch["position"], batch["velocity"]
    delta = position[:, None, :, :] - position[:, :, None, :]  # j-i
    distance2 = delta.square().sum(-1).clamp_min(1e-12)
    distance = distance2.sqrt()
    direction = delta / distance.unsqueeze(-1)
    relative_velocity = velocity[:, None, :, :] - velocity[:, :, None, :]
    radial_velocity = (relative_velocity * direction).sum(-1)
    n = position.shape[1]
    mask = (1.0 - torch.eye(n, device=position.device)).unsqueeze(0)
    return delta, distance2, distance, direction, radial_velocity, mask


def analytic_components(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return dense_particle_components(batch, epsilon=EPSILON, gaussian_sigma=0.55)


def pair_features(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, distance2, distance, direction, radial_velocity, mask = pair_geometry(batch)
    mass_product = batch["mass"][:, :, None] * batch["mass"][:, None, :]
    stretch = batch["bond"] * (distance - batch["rest"])
    gravity = batch["gravity_g"][:, None, None].expand_as(distance)
    features = torch.stack([
        distance, radial_velocity, mass_product, batch["bond"], stretch,
        batch["spring"], batch["dashpot"], gravity,
    ], -1)
    scale = torch.tensor([2.0, 0.15, 1.2, 1.0, 0.25, 1.2, 0.35, 0.004], device=features.device)
    return features / scale, direction, mask


class PairMLP(nn.Module):
    def __init__(self, hidden: int = 48, output_scale: float = 0.06):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.output_scale = output_scale
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features, direction, mask = pair_features(batch)
        scalar = self.output_scale * torch.tanh(self.net(features).squeeze(-1)) * mask
        return (scalar.unsqueeze(-1) * direction).sum(2)


class ForceModel(nn.Module):
    def force(self, batch: dict[str, torch.Tensor], progress: float = 1.0) -> torch.Tensor:
        raise NotImplementedError

    def step(self, batch: dict[str, torch.Tensor], progress: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        force = self.force(batch, progress)
        acceleration = force / batch["mass"].unsqueeze(-1)
        next_velocity = batch["velocity"] + DT * acceleration
        next_position = batch["position"] + DT * next_velocity
        return next_position, next_velocity


class TaskKernelGNN(ForceModel):
    names = ("gravity", "spring", "dashpot", "gaussian")

    def __init__(self, calibration: dict[str, float]):
        super().__init__()
        self.register_buffer("calibration", torch.tensor([calibration[name] for name in self.names], dtype=torch.float32))
        self.message_logits = nn.Parameter(torch.zeros(5))
        self.log_config = nn.Parameter(torch.zeros(4))
        self.log_global_scale = nn.Parameter(torch.log(self.calibration[:3].sum()))
        self.mlp = PairMLP(hidden=48, output_scale=0.06)
        self.update_logits = nn.Parameter(torch.tensor([1.0, 0.0, -0.5, -0.5]))
        self.node_damping = nn.Parameter(torch.tensor(-1.0))
        self.node_net = nn.Sequential(nn.Linear(4, 24), nn.SiLU(), nn.Linear(24, 2))
        nn.init.zeros_(self.node_net[-1].weight)
        nn.init.zeros_(self.node_net[-1].bias)

    @staticmethod
    def temperatures(progress: float) -> tuple[float, float]:
        return (
            annealed_temperature(progress, 1.0, 0.5, 0.30),
            annealed_temperature(progress, 1.0, 0.3, 0.30),
        )

    def gates(self, progress: float) -> tuple[torch.Tensor, torch.Tensor]:
        tau_a, tau_b = self.temperatures(progress)
        return temperature_softmax(self.message_logits, tau_a), temperature_softmax(self.update_logits, tau_b)

    def force(self, batch, progress=1.0):
        parts = analytic_components(batch)
        candidates = [parts[name] * self.log_config[index].exp() / self.calibration[index] for index, name in enumerate(self.names)]
        mlp = ETA * self.mlp(batch) / self.calibration[:3].mean()
        candidates.append(mlp)
        a, _ = self.gates(progress)
        return self.log_global_scale.exp() * sum(a[index] * candidate for index, candidate in enumerate(candidates))

    def step(self, batch, progress=1.0):
        force = self.force(batch, progress)
        acceleration = force / batch["mass"].unsqueeze(-1)
        velocity = batch["velocity"]
        semi_velocity = velocity + DT * acceleration
        semi_position_velocity = semi_velocity
        explicit_velocity = semi_velocity
        explicit_position_velocity = velocity
        damping = torch.nn.functional.softplus(self.node_damping)
        damp_velocity = (1.0 - DT * damping) * velocity + DT * acceleration
        damp_position_velocity = damp_velocity
        speed = velocity.norm(dim=-1, keepdim=True)
        acc_norm = acceleration.norm(dim=-1, keepdim=True)
        coeff = ETA * torch.tanh(self.node_net(torch.cat([speed, acc_norm, batch["mass"].unsqueeze(-1), torch.ones_like(speed)], -1)))
        neural_velocity = semi_velocity + DT * (coeff[..., :1] * acceleration + coeff[..., 1:] * velocity)
        neural_position_velocity = neural_velocity
        _, b = self.gates(progress)
        next_velocity = b[0] * semi_velocity + b[1] * explicit_velocity + b[2] * damp_velocity + b[3] * neural_velocity
        position_velocity = b[0] * semi_position_velocity + b[1] * explicit_position_velocity + b[2] * damp_position_velocity + b[3] * neural_position_velocity
        return batch["position"] + DT * position_velocity, next_velocity


def calibration_from_groups(groups: dict[int, dict[str, torch.Tensor]]) -> dict[str, float]:
    values = {name: [] for name in TaskKernelGNN.names}
    for group in groups.values():
        count = min(256, len(group["position"]))
        batch = slice_batch(group, torch.arange(count))
        parts = analytic_components(batch)
        for name in values:
            values[name].append(parts[name].square().mean())
    return {name: float(torch.stack(items).mean().sqrt().clamp_min(1e-5)) for name, items in values.items()}


@torch.no_grad()
def validation_loss(model: ForceModel, groups: dict[int, dict[str, torch.Tensor]]) -> float:
    errors, references = [], []
    for group in groups.values():
        prediction = model.force(group, 1.0)
        errors.append((prediction - group["target_force"]).square().sum())
        references.append(group["target_force"].square().sum())
    return float(torch.stack(errors).sum() / torch.stack(references).sum().clamp_min(1e-12))


def train_model(model: ForceModel, train: dict[int, dict[str, torch.Tensor]], validation: dict[int, dict[str, torch.Tensor]], steps: int, seed: int) -> tuple[ForceModel, dict[str, float]]:
    set_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-6)
    target_rms = torch.cat([group["target_force"].reshape(-1) for group in train.values()]).square().mean().sqrt()
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    keys = sorted(train)
    for step in range(steps):
        progress = (step + 1) / steps
        n = keys[step % len(keys)]
        group = train[n]
        generator = torch.Generator().manual_seed(seed * 1_000_000 + step)
        indices = torch.randint(len(group["position"]), (min(96, len(group["position"])),), generator=generator)
        batch = slice_batch(group, indices)
        predicted_force = model.force(batch, progress)
        force_loss = (predicted_force - batch["target_force"]).square().mean() / target_rms.square().clamp_min(1e-12)
        predicted_position, predicted_velocity = model.step(batch, progress)
        acceleration_scale = (target_rms / batch["mass"].mean()).clamp_min(1e-6)
        velocity_rate_error = ((predicted_velocity - batch["next_velocity"]) / DT).square().mean() / acceleration_scale.square()
        position_rate_error = ((predicted_position - batch["next_position"]) / DT**2).square().mean() / acceleration_scale.square()
        loss = force_loss + 0.10 * velocity_rate_error + 0.10 * position_rate_error
        if isinstance(model, TaskKernelGNN):
            loss = loss + 2e-4 * model.log_config.square().sum()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if (step + 1) % 50 == 0 or step + 1 == steps:
            value = validation_loss(model, validation)
            if value < best_validation:
                best_validation = value
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, {"validation_relative_force_mse": best_validation, "target_force_rms": float(target_rms)}


def batch_state(trajectory: Trajectory, step: int, position: torch.Tensor, velocity: torch.Tensor) -> dict[str, torch.Tensor]:
    matrices = system_matrices(trajectory.system)
    return {
        "position": position.unsqueeze(0), "velocity": velocity.unsqueeze(0),
        "next_position": torch.tensor(trajectory.position[min(step + 1, len(trajectory.position) - 1)]).unsqueeze(0),
        "next_velocity": torch.tensor(trajectory.velocity[min(step + 1, len(trajectory.velocity) - 1)]).unsqueeze(0),
        "target_force": torch.tensor(trajectory.force[min(step, len(trajectory.force) - 1)]).unsqueeze(0),
        **{key: value.unsqueeze(0) for key, value in matrices.items()},
    }


@torch.no_grad()
def evaluate_trajectory(model: ForceModel, trajectory: Trajectory) -> dict[str, float]:
    position = torch.tensor(trajectory.position[0])
    velocity = torch.tensor(trajectory.velocity[0])
    predicted_positions, predicted_velocities = [], []
    force_errors, force_references = [], []
    first_unstable = len(trajectory.force)
    for step in range(len(trajectory.force)):
        batch = batch_state(trajectory, step, position, velocity)
        predicted_force = model.force(batch, 1.0)[0]
        true_force = batch["target_force"][0]
        force_errors.append((predicted_force - true_force).square().mean())
        force_references.append(true_force.square().mean())
        next_position, next_velocity = model.step(batch, 1.0)
        position, velocity = next_position[0], next_velocity[0]
        predicted_positions.append(position)
        predicted_velocities.append(velocity)
        if (not torch.isfinite(position).all() or not torch.isfinite(velocity).all()
                or position.abs().max() > 100 or velocity.abs().max() > 10):
            first_unstable = step + 1
            break
    exploded = first_unstable < len(trajectory.force)
    if exploded:
        return {"exploded": 1.0, "stable_fraction": first_unstable / len(trajectory.force), "position_nrmse": float("inf"),
                "velocity_nrmse": float("inf"), "force_nrmse": float("inf"), "momentum_drift": float("inf"), "equivariance_error": float("nan")}
    predicted_position = torch.stack(predicted_positions)
    predicted_velocity = torch.stack(predicted_velocities)
    true_position = torch.tensor(trajectory.position[1:])
    true_velocity = torch.tensor(trajectory.velocity[1:])
    position_nrmse = float((predicted_position - true_position).square().mean().sqrt() / true_position.square().mean().sqrt().clamp_min(1e-8))
    velocity_nrmse = float((predicted_velocity - true_velocity).square().mean().sqrt() / true_velocity.square().mean().sqrt().clamp_min(1e-8))
    force_nrmse = float(torch.stack(force_errors).mean().sqrt() / torch.stack(force_references).mean().sqrt().clamp_min(1e-8))
    mass = torch.tensor(trajectory.system.mass).float()
    momentum0 = (mass[:, None] * torch.tensor(trajectory.velocity[0])).sum(0)
    momentum = (mass[None, :, None] * predicted_velocity).sum(1)
    momentum_scale = (mass * torch.tensor(trajectory.velocity[0]).norm(dim=-1)).sum().clamp_min(1e-8)
    momentum_drift = float((momentum - momentum0).norm(dim=-1).max() / momentum_scale)
    # One-state rotation test of the complete learned force map.
    theta = 0.73
    rotation = torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=torch.float32)
    base = batch_state(trajectory, 0, torch.tensor(trajectory.position[0]), torch.tensor(trajectory.velocity[0]))
    rotated = {key: value.clone() for key, value in base.items()}
    rotated["position"] = base["position"] @ rotation.T + torch.tensor([0.7, -0.4])
    rotated["velocity"] = base["velocity"] @ rotation.T
    reference_force = model.force(base, 1.0)[0] @ rotation.T
    transformed_force = model.force(rotated, 1.0)[0]
    equivariance = float((transformed_force - reference_force).norm() / reference_force.norm().clamp_min(1e-8))
    return {"exploded": 0.0, "stable_fraction": 1.0, "position_nrmse": position_nrmse,
            "velocity_nrmse": velocity_nrmse, "force_nrmse": force_nrmse,
            "momentum_drift": momentum_drift, "equivariance_error": equivariance}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(exist_ok=True)
    print("Generating frozen D3 datasets...", flush=True)
    splits = make_splits()
    train_groups = training_groups(splits["Train"])
    validation_groups = training_groups(splits["Validation"])
    calibration = calibration_from_groups(train_groups)
    rows, training_rows = [], []
    for seed in range(args.seeds):
        print(f"Seed {seed + 1}/{args.seeds}", flush=True)
        model_name = "TaskKernel-GNN"
        training_seed = 7_204 + 100 * seed
        set_seed(training_seed)
        model, info = train_model(TaskKernelGNN(calibration), train_groups, validation_groups, args.steps, training_seed)
        training_rows.append({"seed": seed, "model": model_name, **info})
        torch.save(model.state_dict(), CHECKPOINTS / f"taskkernel_gnn_seed{seed}.pt")
        for split_name, trajectories in splits.items():
            if split_name in ("Train", "Validation"):
                continue
            for trajectory_index, trajectory in enumerate(trajectories):
                metrics = evaluate_trajectory(model, trajectory)
                rows.append({"seed": seed, "model": model_name, "split": split_name,
                             "trajectory": trajectory_index, "nodes": len(trajectory.system.mass), **metrics})
    pd.DataFrame(rows).to_csv(RESULTS / "rollout_metrics.csv", index=False)
    pd.DataFrame(training_rows).to_csv(RESULTS / "training_metrics.csv", index=False)
    (RESULTS / "settings.json").write_text(json.dumps({
        "dt": DT, "epsilon": EPSILON, "eta": ETA, "train_trajectories": len(splits["Train"]),
        "validation_trajectories": len(splits["Validation"]), "training_steps": args.steps,
        "seeds": args.seeds, "calibration": calibration, "aggregation": "ADD",
    }, indent=2) + "\n", encoding="utf-8")
    print("Formal D3 runs complete.", flush=True)


if __name__ == "__main__":
    main()

