"""D1-Single controlled experiment for TaskKernel-GNN.

The ground-truth system uses one Gaussian diffusion message kernel and one
cubic-reaction update kernel.  The experiment evaluates:
  A) gate recovery with physical configurations fixed,
  B) physical configuration recovery with oracle one-hot gates,
  C) joint gate/configuration learning.

Results are written to results/Ablation/d1/single/.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "Ablation" / "d1" / "single"
DEVICE = torch.device("cpu")
DT = 0.025
TRUE_CONFIG = {"D": 0.18, "beta": 2.0, "rho": 0.55, "kappa": 0.70}
TRUE_MSG_INDEX = 1
TRUE_UPD_INDEX = 1
N_MSG_KERNELS = 6
N_UPD_KERNELS = 6


@dataclass
class Dataset:
    x: torch.Tensor
    dist: torch.Tensor
    mask: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_graphs(num_graphs: int, n_nodes: int, seed: int, k: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    pos = torch.rand(num_graphs, n_nodes, 2, generator=generator)
    dist = torch.cdist(pos, pos)
    eye = torch.eye(n_nodes, dtype=torch.bool).unsqueeze(0)
    ranking_dist = dist.masked_fill(eye, float("inf"))
    neighbors = ranking_dist.topk(k=min(k, n_nodes - 1), largest=False).indices
    directed = torch.zeros(num_graphs, n_nodes, n_nodes, dtype=torch.bool)
    directed.scatter_(2, neighbors, True)
    mask = (directed | directed.transpose(1, 2)) & ~eye
    return dist, mask.float()


def true_step(x: torch.Tensor, dist: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dx = x.unsqueeze(-2) - x.unsqueeze(-1)
    msg = TRUE_CONFIG["D"] * torch.exp(-TRUE_CONFIG["beta"] * dist.square()) * dx * mask
    aggregate = msg.sum(dim=-1)
    reaction = TRUE_CONFIG["rho"] * x - TRUE_CONFIG["kappa"] * x.pow(3)
    return x + DT * (aggregate + reaction)


def make_dataset(num_trajectories: int, n_nodes: int, steps: int, seed: int) -> Dataset:
    generator = torch.Generator().manual_seed(seed)
    dist, mask = make_graphs(num_trajectories, n_nodes, seed + 17)
    x = torch.empty(num_trajectories, steps + 1, n_nodes)
    x[:, 0] = 0.7 * torch.randn(num_trajectories, n_nodes, generator=generator)
    for t in range(steps):
        x[:, t + 1] = true_step(x[:, t], dist, mask)
    return Dataset(x=x, dist=dist, mask=mask)


class TaskKernel(nn.Module):
    def __init__(self, mode: str) -> None:
        super().__init__()
        if mode not in {"gates", "config", "joint"}:
            raise ValueError(mode)
        self.mode = mode
        # All analytic kernels start with identical logits.  Only the flexible
        # MLP kernel receives the method-level conservative prior.
        msg_init = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, -2.0])
        upd_init = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, -2.0])
        self.msg_logits = nn.Parameter(msg_init, requires_grad=mode != "config")
        self.upd_logits = nn.Parameter(upd_init, requires_grad=mode != "config")

        # Positive physical configurations. Initial values intentionally differ
        # from the ground truth in config/joint modes.
        initial = {"D": 0.10, "beta": 1.1, "rho": 0.30, "kappa": 0.38}
        self.raw_D = nn.Parameter(torch.tensor(inv_softplus(initial["D"])), requires_grad=mode != "gates")
        self.raw_beta = nn.Parameter(torch.tensor(inv_softplus(initial["beta"])), requires_grad=mode != "gates")
        self.raw_rho = nn.Parameter(torch.tensor(inv_softplus(initial["rho"])), requires_grad=mode != "gates")
        self.raw_kappa = nn.Parameter(torch.tensor(inv_softplus(initial["kappa"])), requires_grad=mode != "gates")

        self.msg_mlp = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 1))
        self.upd_mlp = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
        initialize_small(self.msg_mlp)
        initialize_small(self.upd_mlp)

        if mode == "config":
            with torch.no_grad():
                self.msg_logits.fill_(-12.0)
                self.msg_logits[TRUE_MSG_INDEX] = 12.0
                self.upd_logits.fill_(-12.0)
                self.upd_logits[TRUE_UPD_INDEX] = 12.0

    @property
    def configs(self) -> dict[str, torch.Tensor]:
        if self.mode == "gates":
            return {key: torch.tensor(value, device=DEVICE) for key, value in TRUE_CONFIG.items()}
        return {
            "D": F.softplus(self.raw_D),
            "beta": F.softplus(self.raw_beta),
            "rho": F.softplus(self.raw_rho),
            "kappa": F.softplus(self.raw_kappa),
        }

    def weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.msg_logits.softmax(0), self.upd_logits.softmax(0)

    def forward(self, x: torch.Tensor, dist: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        c = self.configs
        dx = x.unsqueeze(-2) - x.unsqueeze(-1)
        xi = x.unsqueeze(-1).expand_as(dx)
        xj = x.unsqueeze(-2).expand_as(dx)
        safe_dist = dist + 0.20

        msg_candidates = [
            0.12 * dx,
            c["D"] * torch.exp(-c["beta"] * dist.square()) * dx,
            0.055 * dx / safe_dist,
            0.075 * torch.sin(dx),
            0.050 * torch.tanh(xj),
            0.10 * torch.tanh(self.msg_mlp(torch.stack([xi, xj, dist, dx], dim=-1)).squeeze(-1)),
        ]
        msg_stack = torch.stack(msg_candidates, dim=-1) * mask.unsqueeze(-1)
        a, b = self.weights()
        aggregate = (msg_stack * a).sum(dim=-1).sum(dim=-1)

        update_candidates = [
            DT * (aggregate + 0.35 * x),
            DT * (aggregate + c["rho"] * x - c["kappa"] * x.pow(3)),
            DT * (aggregate + 0.45 * x * (1.0 - x / 1.6)),
            DT * (aggregate - 0.25 * x),
            DT * torch.tanh(aggregate + 0.25 * x),
            DT * torch.tanh(self.upd_mlp(torch.stack([x, aggregate], dim=-1)).squeeze(-1)),
        ]
        update_stack = torch.stack(update_candidates, dim=-1)
        delta = (update_stack * b).sum(dim=-1)
        return x + delta

    def regularization(self, x: torch.Tensor, dist: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        a, b = self.weights()
        mlp_gate_penalty = a[-1].square() + b[-1].square()
        return 1e-3 * mlp_gate_penalty


def inv_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def initialize_small(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=0.15)
            nn.init.zeros_(layer.bias)


def sample_batch(data: Dataset, trajectory_count: int, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, ...]:
    trajectories = torch.randint(0, trajectory_count, (batch_size,), generator=generator)
    times = torch.randint(0, data.x.shape[1] - 1, (batch_size,), generator=generator)
    return (
        data.x[trajectories, times],
        data.x[trajectories, times + 1],
        data.dist[trajectories],
        data.mask[trajectories],
    )


def train_model(
    model: nn.Module,
    data: Dataset,
    validation: Dataset,
    trajectory_count: int,
    seed: int,
    steps: int,
) -> nn.Module:
    set_seed(seed)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=7e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=5e-4)
    generator = torch.Generator().manual_seed(seed + 1000)
    validation_generator = torch.Generator().manual_seed(42_000 + seed)
    validation_batch = sample_batch(
        validation,
        validation.x.shape[0],
        min(512, validation.x.shape[0] * (validation.x.shape[1] - 1)),
        validation_generator,
    )
    best_loss = float("inf")
    best_state = None
    for step in range(steps):
        x, target, dist, mask = sample_batch(data, trajectory_count, min(32, trajectory_count * 4), generator)
        pred = model(x, dist, mask)
        # Fit the state-change rate rather than the next state.  Direct next-state
        # MSE is dominated by the identity term when DT is small.
        pred_rate = (pred - x) / DT
        target_rate = (target - x) / DT
        loss = F.mse_loss(pred_rate, target_rate) + model.regularization(x, dist, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        if (step + 1) % 50 == 0 or step + 1 == steps:
            validation_loss = rate_loss(model, validation_batch)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def rate_loss(model: nn.Module, batch: tuple[torch.Tensor, ...]) -> float:
    x, target, dist, mask = batch
    pred = model(x, dist, mask)
    return float(F.mse_loss((pred - x) / DT, (target - x) / DT))


@torch.no_grad()
def evaluate(model: nn.Module, data: Dataset) -> dict[str, float]:
    x_current = data.x[:, 0].to(DEVICE)
    target_all = data.x.to(DEVICE)
    dist = data.dist.to(DEVICE)
    mask = data.mask.to(DEVICE)
    predictions = [x_current]
    for _ in range(target_all.shape[1] - 1):
        x_current = model(x_current, dist, mask)
        predictions.append(x_current)
    pred = torch.stack(predictions, dim=1)
    error = pred - target_all
    rollout_nrmse = torch.sqrt(error[:, 1:].square().mean()) / (torch.sqrt(target_all[:, 1:].square().mean()) + 1e-8)
    final_nrmse = torch.sqrt(error[:, -1].square().mean()) / (torch.sqrt(target_all[:, -1].square().mean()) + 1e-8)

    x = target_all[:, :-1].reshape(-1, target_all.shape[-1])
    y = target_all[:, 1:].reshape(-1, target_all.shape[-1])
    time_steps = target_all.shape[1] - 1
    repeated_dist = dist[:, None].expand(-1, time_steps, -1, -1).reshape(-1, dist.shape[-1], dist.shape[-1])
    repeated_mask = mask[:, None].expand(-1, time_steps, -1, -1).reshape_as(repeated_dist)
    one_step = model(x, repeated_dist, repeated_mask)
    one_step_nrmse = torch.sqrt((one_step - y).square().mean()) / (torch.sqrt(y.square().mean()) + 1e-8)

    metrics = {
        "one_step_nrmse": float(one_step_nrmse),
        "rollout_nrmse": float(rollout_nrmse),
        "final_nrmse": float(final_nrmse),
        "parameter_count": float(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
    }
    if isinstance(model, TaskKernel):
        a, b = model.weights()
        configs = model.configs
        metrics.update({
            "msg_true_weight": float(a[TRUE_MSG_INDEX]),
            "msg_mlp_weight": float(a[-1]),
            "upd_true_weight": float(b[TRUE_UPD_INDEX]),
            "upd_mlp_weight": float(b[-1]),
            "D": float(configs["D"]),
            "beta": float(configs["beta"]),
            "rho": float(configs["rho"]),
            "kappa": float(configs["kappa"]),
        })
    return metrics


def run(args: argparse.Namespace) -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train = make_dataset(args.train_pool, args.nodes, args.steps, seed=101)
    validation = make_dataset(args.validation_trajectories, args.nodes, args.steps, seed=202)
    test = make_dataset(args.test_trajectories, args.nodes, args.steps, seed=303)
    rows: list[dict[str, float | int | str]] = []

    for budget in args.budgets:
        for seed in range(args.seeds):
            set_seed(10_000 + 100 * budget + seed)
            model = train_model(TaskKernel("gates"), train, validation, budget, seed=seed, steps=args.train_steps)
            rows.append({"experiment": "taskkernel_gates", "budget": budget, "seed": seed, **evaluate(model, test)})
            print(f"finished taskkernel_gates budget={budget} seed={seed}", flush=True)

    parameter_budget = min(args.parameter_budget, args.train_pool)
    for seed in range(args.seeds):
        for mode in ("config", "joint"):
            set_seed(20_000 + 100 * seed + (0 if mode == "config" else 1))
            model = train_model(
                TaskKernel(mode),
                train,
                validation,
                parameter_budget,
                seed=seed + 100,
                steps=args.train_steps,
            )
            rows.append({"experiment": f"taskkernel_{mode}", "budget": parameter_budget, "seed": seed, **evaluate(model, test)})
            print(f"finished taskkernel_{mode} budget={parameter_budget} seed={seed}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / "runs.csv", index=False)
    summary = frame.groupby(["experiment", "budget"], dropna=False).agg(["mean", "std"])
    summary.to_csv(RESULTS_DIR / "summary.csv")
    write_summary_json(frame, args)
    plot_results(frame)
    return frame


def write_summary_json(frame: pd.DataFrame, args: argparse.Namespace) -> None:
    numeric_columns = [column for column in frame.columns if column not in {"experiment", "budget", "seed"}]
    records: dict[str, dict[str, dict[str, float]]] = {}
    for (experiment, budget), group in frame.groupby(["experiment", "budget"]):
        key = f"{experiment}/budget={budget}"
        records[key] = {}
        for column in numeric_columns:
            values = group[column].dropna()
            if len(values):
                records[key][column] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    payload = {"settings": vars(args), "true_config": TRUE_CONFIG, "results": records}
    (RESULTS_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_results(frame: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    budget_frame = frame[frame["experiment"] == "taskkernel_gates"]
    for name, group in budget_frame.groupby("experiment"):
        stats = group.groupby("budget")["rollout_nrmse"].agg(["mean", "std"]).reset_index()
        axes[0].errorbar(stats["budget"], stats["mean"], yerr=stats["std"].fillna(0), marker="o", label=name)
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("training trajectories")
    axes[0].set_ylabel("rollout NRMSE")
    axes[0].set_title("D1-Single sample efficiency")
    axes[0].legend()

    gates = frame[frame["experiment"] == "taskkernel_gates"]
    stats = gates.groupby("budget")[["msg_true_weight", "upd_true_weight"]].mean()
    axes[1].plot(stats.index, stats["msg_true_weight"], marker="o", label="true message kernel")
    axes[1].plot(stats.index, stats["upd_true_weight"], marker="o", label="true update kernel")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xscale("log", base=2)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("training trajectories")
    axes[1].set_ylabel("softmax weight")
    axes[1].set_title("Ground-truth kernel recovery")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "d1_single_results.png", dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--train-pool", type=int, default=64)
    parser.add_argument("--test-trajectories", type=int, default=64)
    parser.add_argument("--validation-trajectories", type=int, default=32)
    parser.add_argument("--nodes", type=int, default=24)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--train-steps", type=int, default=900)
    parser.add_argument("--parameter-budget", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments)

