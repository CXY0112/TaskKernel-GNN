"""D1-Mixture and D1-Distractor experiments for TaskKernel-GNN.

The ground-truth system combines three message kernels and three update
kernels. We compare a minimal correct library with one that also contains two analytic
distractors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from d1_single import DT, Dataset, make_graphs, sample_batch, set_seed


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "Ablation" / "d1" / "mixture"
MSG_TRUE = torch.tensor([0.25, 0.55, 0.0, 0.20, 0.0, 0.0])
UPD_TRUE = torch.tensor([0.25, 0.55, 0.0, 0.20, 0.0, 0.0])
ACTIVE_ANALYTIC = [0, 1, 3]
DISTRACTOR_ANALYTIC = [2, 4]


def analytic_messages(x: torch.Tensor, dist: torch.Tensor) -> list[torch.Tensor]:
    dx = x.unsqueeze(-2) - x.unsqueeze(-1)
    xj = x.unsqueeze(-2).expand_as(dx)
    return [
        0.12 * dx,
        0.18 * torch.exp(-2.0 * dist.square()) * dx,
        0.055 * dx / (dist + 0.20),
        0.075 * torch.sin(dx),
        0.050 * torch.tanh(xj),
    ]


def analytic_updates(x: torch.Tensor, aggregate: torch.Tensor) -> list[torch.Tensor]:
    return [
        DT * (aggregate + 0.35 * x),
        DT * (aggregate + 0.55 * x - 0.70 * x.pow(3)),
        DT * (aggregate + 0.45 * x * (1.0 - x / 1.6)),
        DT * (aggregate - 0.25 * x),
        DT * torch.tanh(aggregate + 0.25 * x),
    ]


def true_step(x: torch.Tensor, dist: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    messages = torch.stack(analytic_messages(x, dist), dim=-1) * mask.unsqueeze(-1)
    mixed_message = (messages * MSG_TRUE[:5]).sum(dim=-1)
    aggregate = mixed_message.sum(dim=-1)
    updates = torch.stack(analytic_updates(x, aggregate), dim=-1)
    delta = (updates * UPD_TRUE[:5]).sum(dim=-1)
    return x + delta


def make_dataset(num_trajectories: int, n_nodes: int, steps: int, seed: int) -> Dataset:
    generator = torch.Generator().manual_seed(seed)
    dist, mask = make_graphs(num_trajectories, n_nodes, seed + 17)
    x = torch.empty(num_trajectories, steps + 1, n_nodes)
    x[:, 0] = 0.75 * torch.randn(num_trajectories, n_nodes, generator=generator)
    for t in range(steps):
        x[:, t + 1] = true_step(x[:, t], dist, mask)
    return Dataset(x=x, dist=dist, mask=mask)


class MultiKernel(nn.Module):
    def __init__(self, library: str) -> None:
        super().__init__()
        if library == "minimal":
            self.indices = ACTIVE_ANALYTIC + [5]
        elif library == "distractor":
            self.indices = list(range(6))
        else:
            raise ValueError(library)
        self.library = library
        init = torch.tensor([0.0 if index < 5 else -2.0 for index in self.indices])
        self.msg_logits = nn.Parameter(init.clone())
        self.upd_logits = nn.Parameter(init.clone())
        self.msg_mlp = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 1))
        self.upd_mlp = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
        initialize_small(self.msg_mlp)
        initialize_small(self.upd_mlp)

    def full_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        msg = torch.zeros(6, device=self.msg_logits.device)
        upd = torch.zeros(6, device=self.upd_logits.device)
        msg[self.indices] = self.msg_logits.softmax(0)
        upd[self.indices] = self.upd_logits.softmax(0)
        return msg, upd

    def forward(self, x: torch.Tensor, dist: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        dx = x.unsqueeze(-2) - x.unsqueeze(-1)
        xi = x.unsqueeze(-1).expand_as(dx)
        xj = x.unsqueeze(-2).expand_as(dx)
        messages = analytic_messages(x, dist)
        mlp_message = 0.10 * torch.tanh(
            self.msg_mlp(torch.stack([xi, xj, dist, dx], dim=-1)).squeeze(-1)
        )
        message_stack = torch.stack(messages + [mlp_message], dim=-1) * mask.unsqueeze(-1)
        a, b = self.full_weights()
        aggregate = (message_stack * a).sum(dim=-1).sum(dim=-1)

        updates = analytic_updates(x, aggregate)
        mlp_update = DT * torch.tanh(
            self.upd_mlp(torch.stack([x, aggregate], dim=-1)).squeeze(-1)
        )
        update_stack = torch.stack(updates + [mlp_update], dim=-1)
        return x + (update_stack * b).sum(dim=-1)

    def regularization(self, x: torch.Tensor, dist: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        a, b = self.full_weights()
        return 1e-3 * (a[-1].square() + b[-1].square())


def initialize_small(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=0.15)
            nn.init.zeros_(layer.bias)


def rate_loss(model: nn.Module, batch: tuple[torch.Tensor, ...]) -> float:
    with torch.no_grad():
        x, target, dist, mask = batch
        pred = model(x, dist, mask)
        return float(F.mse_loss((pred - x) / DT, (target - x) / DT))


def train_model(
    model: nn.Module,
    train: Dataset,
    validation: Dataset,
    budget: int,
    seed: int,
    steps: int,
) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=7e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=5e-4)
    train_generator = torch.Generator().manual_seed(1000 + seed)
    validation_generator = torch.Generator().manual_seed(42000 + seed)
    validation_batch = sample_batch(
        validation,
        validation.x.shape[0],
        min(512, validation.x.shape[0] * (validation.x.shape[1] - 1)),
        validation_generator,
    )
    best_loss = float("inf")
    best_state = None
    for step in range(steps):
        x, target, dist, mask = sample_batch(train, budget, min(32, budget * 4), train_generator)
        pred = model(x, dist, mask)
        loss = F.mse_loss((pred - x) / DT, (target - x) / DT) + model.regularization(x, dist, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        if (step + 1) % 50 == 0 or step + 1 == steps:
            current = rate_loss(model, validation_batch)
            if current < best_loss:
                best_loss = current
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, data: Dataset) -> dict[str, float]:
    current = data.x[:, 0]
    predictions = [current]
    for _ in range(data.x.shape[1] - 1):
        current = model(current, data.dist, data.mask)
        predictions.append(current)
    prediction = torch.stack(predictions, dim=1)
    error = prediction - data.x
    rollout = torch.sqrt(error[:, 1:].square().mean()) / (torch.sqrt(data.x[:, 1:].square().mean()) + 1e-8)
    final = torch.sqrt(error[:, -1].square().mean()) / (torch.sqrt(data.x[:, -1].square().mean()) + 1e-8)

    time_steps = data.x.shape[1] - 1
    x = data.x[:, :-1].reshape(-1, data.x.shape[-1])
    y = data.x[:, 1:].reshape_as(x)
    dist = data.dist[:, None].expand(-1, time_steps, -1, -1).reshape(-1, data.dist.shape[-1], data.dist.shape[-1])
    mask = data.mask[:, None].expand(-1, time_steps, -1, -1).reshape_as(dist)
    one_step = model(x, dist, mask)
    one_step_nrmse = torch.sqrt((one_step - y).square().mean()) / (torch.sqrt(y.square().mean()) + 1e-8)
    result = {
        "one_step_nrmse": float(one_step_nrmse),
        "rollout_nrmse": float(rollout),
        "final_nrmse": float(final),
        "parameter_count": float(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }
    if isinstance(model, MultiKernel):
        a, b = model.full_weights()
        result.update({
            "msg_weight_l1": float((a - MSG_TRUE).abs().sum()),
            "upd_weight_l1": float((b - UPD_TRUE).abs().sum()),
            "msg_active_mass": float(a[ACTIVE_ANALYTIC].sum()),
            "upd_active_mass": float(b[ACTIVE_ANALYTIC].sum()),
            "msg_distractor_mass": float(a[DISTRACTOR_ANALYTIC].sum()),
            "upd_distractor_mass": float(b[DISTRACTOR_ANALYTIC].sum()),
            "msg_mlp_weight": float(a[-1]),
            "upd_mlp_weight": float(b[-1]),
            **{f"a_{index}": float(a[index]) for index in range(6)},
            **{f"b_{index}": float(b[index]) for index in range(6)},
        })
    return result


def run(args: argparse.Namespace) -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train = make_dataset(args.train_pool, args.nodes, args.steps, seed=111)
    validation = make_dataset(args.validation_trajectories, args.nodes, args.steps, seed=222)
    test = make_dataset(args.test_trajectories, args.nodes, args.steps, seed=333)
    rows: list[dict[str, float | int | str]] = []

    for budget in args.budgets:
        for seed in range(args.seeds):
            for name in ("minimal", "distractor"):
                set_seed(10_000 + budget * 100 + seed * 10 + {"minimal": 1, "distractor": 2}[name])
                model: nn.Module = MultiKernel(name)
                model = train_model(model, train, validation, budget, seed, args.train_steps)
                rows.append({"experiment": name, "budget": budget, "seed": seed, **evaluate(model, test)})
                print(f"finished {name} budget={budget} seed={seed}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / "runs.csv", index=False)
    frame.groupby(["experiment", "budget"], dropna=False).agg(["mean", "std"]).to_csv(RESULTS_DIR / "summary.csv")
    write_json(frame, args)
    plot_results(frame)
    return frame


def write_json(frame: pd.DataFrame, args: argparse.Namespace) -> None:
    excluded = {"experiment", "budget", "seed"}
    records: dict[str, dict[str, dict[str, float]]] = {}
    for (experiment, budget), group in frame.groupby(["experiment", "budget"]):
        key = f"{experiment}/budget={budget}"
        records[key] = {}
        for column in frame.columns:
            if column in excluded:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if len(values):
                records[key][column] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                }
    payload = {
        "settings": vars(args),
        "true_message_weights": MSG_TRUE.tolist(),
        "true_update_weights": UPD_TRUE.tolist(),
        "results": records,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_results(frame: pd.DataFrame) -> None:
    trained = frame
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, group in trained.groupby("experiment"):
        stats = group.groupby("budget")["rollout_nrmse"].agg(["mean", "std"])
        axes[0].errorbar(stats.index, stats["mean"], yerr=stats["std"], marker="o", label=name)
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("training trajectories")
    axes[0].set_ylabel("rollout NRMSE")
    axes[0].set_title("Prediction")
    axes[0].legend()

    for name in ("minimal", "distractor"):
        group = trained[trained["experiment"] == name]
        stats = group.groupby("budget")["msg_weight_l1"].agg(["mean", "std"])
        axes[1].errorbar(stats.index, stats["mean"], yerr=stats["std"], marker="o", label=name)
        stats = group.groupby("budget")["upd_weight_l1"].agg(["mean", "std"])
        axes[2].errorbar(stats.index, stats["mean"], yerr=stats["std"], marker="o", label=name)
    for axis, title in zip(axes[1:], ["Message weight L1", "Update weight L1"]):
        axis.set_xscale("log", base=2)
        axis.set_xlabel("training trajectories")
        axis.set_ylabel("L1 error")
        axis.set_title(title)
        axis.legend()
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "d1_mixture_results.png", dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--train-pool", type=int, default=64)
    parser.add_argument("--validation-trajectories", type=int, default=32)
    parser.add_argument("--test-trajectories", type=int, default=64)
    parser.add_argument("--nodes", type=int, default=24)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--train-steps", type=int, default=900)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

