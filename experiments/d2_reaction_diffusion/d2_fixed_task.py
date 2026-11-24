"""D2-A fixed-task, small-data validation on ring/grid/RGG graphs.

This stage evaluates B5 and B8 on held-out graphs and initial conditions.  All learned models share one parameter set across graph
types and node counts.  Training uses 20-step windows; evaluation is a free
200-step rollout.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from d2_data import BETA, D, DT, RHO1, RHO2, Graph, initial_state, make_graph, rollout


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "Ablation" / "d2" / "graph" / "fixed_task"
KINDS = ("ring", "grid", "rgg")


@dataclass
class Sample:
    kind: str
    nodes: int
    trajectory: torch.Tensor
    coordinates: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    distance: torch.Tensor
    weight: torch.Tensor
    degree: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_sample(graph: Graph, trajectory: torch.Tensor) -> Sample:
    dst, src = torch.where(graph.mask)
    degree = graph.mask.sum(1).float()
    return Sample(
        kind=graph.kind,
        nodes=len(graph.coordinates),
        trajectory=trajectory,
        coordinates=graph.coordinates,
        src=src,
        dst=dst,
        distance=graph.distance[dst, src],
        weight=graph.weight[dst, src],
        degree=degree,
    )


def make_split(count_per_kind_size: int, sizes: tuple[int, ...], seed: int, steps: int) -> list[Sample]:
    samples: list[Sample] = []
    for kind_index, kind in enumerate(KINDS):
        for nodes in sizes:
            for replicate in range(count_per_kind_size):
                graph_seed = seed + kind_index * 100_000 + nodes * 100 + replicate
                state_seed = seed + 500_000 + kind_index * 100_000 + nodes * 100 + replicate
                graph = make_graph(kind, nodes, graph_seed)
                trajectory, _ = rollout(graph, state_seed, steps=steps)
                samples.append(to_sample(graph, trajectory))
    return samples


def scatter_add(values: torch.Tensor, dst: torch.Tensor, nodes: int) -> torch.Tensor:
    result = values.new_zeros(nodes)
    result.index_add_(0, dst, values)
    return result


def positive(raw: torch.Tensor, floor: float = 1e-4) -> torch.Tensor:
    return F.softplus(raw) + floor


def inv_softplus(value: float) -> torch.Tensor:
    return torch.tensor(math.log(math.expm1(value)), dtype=torch.float32)


class MultiKernel(nn.Module):
    """B5 or B8; all update kernels contain the same aggregate.

    Consequently softmax(b) cannot rescale the spatial aggregate.  The MLP
    message is antisymmetric in the state difference, which makes its ADD
    aggregate conservative on the symmetric graph.
    """

    def __init__(self, include_mlp: bool, aggregation: str = "add") -> None:
        super().__init__()
        if aggregation not in {"add", "mean"}:
            raise ValueError(aggregation)
        self.aggregation = aggregation
        self.include_mlp = include_mlp
        self.name = "B8-TaskKernel" if include_mlp else "B5-NoMLP"
        count = 4 if include_mlp else 3
        initial = torch.zeros(count)
        if include_mlp:
            initial[-1] = -2.0
        self.msg_logits = nn.Parameter(initial.clone())
        self.upd_logits = nn.Parameter(initial.clone())
        self.msg_scales_raw = nn.Parameter(torch.stack([inv_softplus(0.10)] * 3))
        self.rho1_raw = nn.Parameter(inv_softplus(0.45))
        self.rho2_raw = nn.Parameter(inv_softplus(0.60))
        self.logistic_rate_raw = nn.Parameter(inv_softplus(0.40))
        self.capacity_raw = nn.Parameter(inv_softplus(1.5))
        self.decay_raw = nn.Parameter(inv_softplus(0.20))
        if include_mlp:
            self.msg_mlp = nn.Sequential(nn.Linear(4, 24), nn.Tanh(), nn.Linear(24, 1))
            self.upd_mlp = nn.Sequential(nn.Linear(2, 24), nn.Tanh(), nn.Linear(24, 1))
            initialize_small(self.msg_mlp)
            initialize_small(self.upd_mlp)

    def weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.msg_logits.softmax(0), self.upd_logits.softmax(0)

    def step(self, x: torch.Tensor, sample: Sample) -> torch.Tensor:
        dx = x[sample.src] - x[sample.dst]
        scales = positive(self.msg_scales_raw)
        messages = [
            scales[0] * sample.weight * dx,
            scales[1] * torch.exp(-2.0 * sample.distance.square()) * dx,
            scales[2] * dx / torch.sqrt(sample.degree[sample.src] * sample.degree[sample.dst]),
        ]
        if self.include_mlp:
            features = torch.stack(
                [(x[sample.src] + x[sample.dst]) / 2, dx.abs(), sample.distance, sample.weight], dim=-1
            )
            edge_scale = 0.10 * torch.tanh(self.msg_mlp(features).squeeze(-1))
            messages.append(edge_scale * dx)
        a, b = self.weights()
        mixed_message = (torch.stack(messages, dim=-1) * a).sum(-1)
        aggregate = scatter_add(mixed_message, sample.dst, sample.nodes)
        if self.aggregation == "mean":
            aggregate = aggregate / sample.degree.clamp_min(1)

        reactions = [
            positive(self.rho1_raw) * x - positive(self.rho2_raw) * x.pow(3),
            positive(self.logistic_rate_raw) * x * (1.0 - x / positive(self.capacity_raw)),
            -positive(self.decay_raw) * x,
        ]
        if self.include_mlp:
            reactions.append(torch.tanh(self.upd_mlp(torch.stack([x, aggregate], dim=-1)).squeeze(-1)))
        reaction = (torch.stack(reactions, dim=-1) * b).sum(-1)
        return x + DT * (aggregate + reaction)


def initialize_small(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=0.20)
            nn.init.zeros_(layer.bias)


def window_loss(model: nn.Module, sample: Sample, start: int, window: int) -> torch.Tensor:
    current = sample.trajectory[start]
    scale = sample.trajectory[start + 1 : start + window + 1].square().mean().sqrt().detach() + 1e-4
    loss = current.new_zeros(())
    for offset in range(1, window + 1):
        current = model.step(current, sample)
        loss = loss + ((current - sample.trajectory[start + offset]) / scale).square().mean()
    return loss / window


@torch.no_grad()
def validation_loss(model: nn.Module, samples: list[Sample], window: int = 20) -> float:
    values = []
    for sample in samples:
        start = min(20, len(sample.trajectory) - window - 1)
        values.append(float(window_loss(model, sample, start, window)))
    return float(np.mean(values))


def train_model(
    model: nn.Module,
    train: list[Sample],
    validation: list[Sample],
    budget: int,
    seed: int,
    train_steps: int,
    window: int,
) -> nn.Module:
    # A budget is a number of trajectories, sampled in a stratified round-robin
    # order so that B>=4 contains all three graph families.
    rng = random.Random(17_000 + seed)
    ordered = []
    pools = {kind: [sample for sample in train if sample.kind == kind] for kind in KINDS}
    for pool in pools.values():
        rng.shuffle(pool)
    for index in range(max(len(pool) for pool in pools.values())):
        for kind in KINDS:
            if index < len(pools[kind]):
                ordered.append(pools[kind][index])
    support = ordered[:budget]
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_steps, eta_min=3e-4)
    best_loss = float("inf")
    best_state = None
    for step_index in range(train_steps):
        sample = support[rng.randrange(len(support))]
        max_start = len(sample.trajectory) - window - 1
        start = rng.randrange(max_start + 1)
        loss = window_loss(model, sample, start, window)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()
        if (step_index + 1) % 50 == 0 or step_index + 1 == train_steps:
            current = validation_loss(model, validation, window)
            if current < best_loss:
                best_loss = current
                best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, samples: list[Sample]) -> list[dict[str, float | int | str]]:
    rows = []
    horizons = (20, 50, 100, 200)
    for sample_index, sample in enumerate(samples):
        one_step_squared_error = sample.trajectory.new_zeros(())
        one_step_squared_reference = sample.trajectory.new_zeros(())
        for time_index in range(200):
            teacher_prediction = model.step(sample.trajectory[time_index], sample)
            one_step_squared_error += (teacher_prediction - sample.trajectory[time_index + 1]).square().sum()
            one_step_squared_reference += sample.trajectory[time_index + 1].square().sum()
        one_step_nrmse = float(torch.sqrt(one_step_squared_error / (one_step_squared_reference + 1e-12)))
        current = sample.trajectory[0]
        predictions = [current]
        diffusion_residual = 0.0
        exploded = False
        for _ in range(200):
            previous = current
            current = model.step(current, sample)
            predictions.append(current)
            if not torch.isfinite(current).all() or current.abs().max() > 10:
                exploded = True
                break
            # Total mass-rate mismatch is evaluated directly below; this term
            # catches numerical failures without assuming access to internals.
            diffusion_residual = max(diffusion_residual, float((current - previous).abs().mean()))
        if exploded:
            rows.append({"sample": sample_index, "graph": sample.kind, "nodes": sample.nodes, "exploded": 1.0,
                         **{f"nrmse_{h}": float("inf") for h in horizons}, "rollout_nrmse": float("inf"),
                         "final_nrmse": float("inf"), "mass_change_rmse": float("inf"),
                         "one_step_nrmse": one_step_nrmse})
            continue
        prediction = torch.stack(predictions)
        truth = sample.trajectory[:201]
        error = prediction - truth
        denom_all = truth[1:].square().mean().sqrt() + 1e-8
        row: dict[str, float | int | str] = {
            "sample": sample_index,
            "graph": sample.kind,
            "nodes": sample.nodes,
            "exploded": 0.0,
            "one_step_nrmse": one_step_nrmse,
            "rollout_nrmse": float(error[1:].square().mean().sqrt() / denom_all),
            "final_nrmse": float(error[-1].square().mean().sqrt() / (truth[-1].square().mean().sqrt() + 1e-8)),
            "mass_change_rmse": float(
                ((prediction[1:].mean(1) - prediction[:-1].mean(1)) -
                 (truth[1:].mean(1) - truth[:-1].mean(1))).square().mean().sqrt()
            ),
        }
        for horizon in horizons:
            err = error[1 : horizon + 1]
            ref = truth[1 : horizon + 1]
            row[f"nrmse_{horizon}"] = float(err.square().mean().sqrt() / (ref.square().mean().sqrt() + 1e-8))
        rows.append(row)
    return rows


def model_metadata(model: nn.Module) -> dict[str, float]:
    result = {"parameter_count": float(sum(parameter.numel() for parameter in model.parameters()))}
    if isinstance(model, MultiKernel):
        a, b = model.weights()
        result.update({f"a_{i}": float(value.detach()) for i, value in enumerate(a)})
        result.update({f"b_{i}": float(value.detach()) for i, value in enumerate(b)})
        result.update({
            "c_msg_weighted": float(positive(model.msg_scales_raw[0]).detach()),
            "c_msg_broad": float(positive(model.msg_scales_raw[1]).detach()),
            "c_msg_normalized": float(positive(model.msg_scales_raw[2]).detach()),
            "c_rho1": float(positive(model.rho1_raw).detach()),
            "c_rho2": float(positive(model.rho2_raw).detach()),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--train-count", type=int, default=12, help="per graph kind and node size")
    parser.add_argument("--validation-count", type=int, default=2)
    parser.add_argument("--test-count", type=int, default=3)
    args = parser.parse_args()
    if args.steps < 200:
        parser.error("--steps must be at least 200 for the paper's evaluation horizons")
    RESULTS.mkdir(parents=True, exist_ok=True)
    train = make_split(args.train_count, (32, 64), 11_000, args.steps)
    validation = make_split(args.validation_count, (32, 64), 22_000, args.steps)
    test = make_split(args.test_count, (32, 64), 33_000, args.steps)
    rows: list[dict[str, float | int | str]] = []
    for budget in args.budgets:
        for seed in range(args.seeds):
            for model_index, constructor in ((0, lambda: MultiKernel(False)), (2, lambda: MultiKernel(True))):
                set_seed(100_000 + budget * 100 + seed * 10 + model_index)
                model = constructor()
                train_model(model, train, validation, budget, seed, args.train_steps, args.window)
                metadata = model_metadata(model)
                for metrics in evaluate(model, test):
                    rows.append({"model": model.name, "budget": budget, "seed": seed, **metadata, **metrics})
                print(f"finished {model.name} budget={budget} seed={seed}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "runs.csv", index=False)
    per_run = frame.groupby(["model", "budget", "seed"], dropna=False).agg(
        one_step_nrmse=("one_step_nrmse", "mean"),
        rollout_nrmse=("rollout_nrmse", "mean"),
        final_nrmse=("final_nrmse", "mean"),
        nrmse_20=("nrmse_20", "mean"),
        nrmse_50=("nrmse_50", "mean"),
        nrmse_100=("nrmse_100", "mean"),
        nrmse_200=("nrmse_200", "mean"),
        mass_change_rmse=("mass_change_rmse", "mean"),
        explosion_rate=("exploded", "mean"),
    ).reset_index()
    summary = per_run.groupby(["model", "budget"], dropna=False).agg(["mean", "std"])
    per_run.to_csv(RESULTS / "per_run.csv", index=False)
    summary.to_csv(RESULTS / "summary.csv")
    settings = vars(args) | {
        "stage": "D2-A ID small-data comparison",
        "status": "complete",
        "aggregation": "ADD",
        "validation_selection": "all held-out trajectories, balanced across ring/grid/rgg and N=32/64",
        "train_trajectories_available": len(train),
        "validation_trajectories": len(validation),
        "test_trajectories": len(test),
        "fixed_parameters": {"D": D, "rho1": RHO1, "rho2": RHO2, "dt": DT},
    }
    (RESULTS / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(per_run.groupby(["model", "budget"])["rollout_nrmse"].agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()

