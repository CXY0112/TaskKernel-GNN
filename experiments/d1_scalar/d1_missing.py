"""D1-Missing: can neural kernels compensate for missing analytic kernels?"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from d1_single import DT, Dataset, sample_batch, set_seed
from d1_mixture import (
    ACTIVE_ANALYTIC,
    MSG_TRUE,
    UPD_TRUE,
    analytic_messages,
    analytic_updates,
    initialize_small,
    make_dataset,
    true_step,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "Ablation" / "d1" / "missing"
DOMINANT_KERNEL = 1

SCENARIOS = {
    "complete": {"msg": [0, 1, 3, 5], "upd": [0, 1, 3, 5]},
    "missing_msg_no_mlp": {"msg": [0, 3], "upd": [0, 1, 3]},
    "missing_msg_mlp": {"msg": [0, 3, 5], "upd": [0, 1, 3, 5]},
    "missing_upd_no_mlp": {"msg": [0, 1, 3], "upd": [0, 3]},
    "missing_upd_mlp": {"msg": [0, 1, 3, 5], "upd": [0, 3, 5]},
    "missing_both_no_mlp": {"msg": [0, 3], "upd": [0, 3]},
    "missing_both_mlp": {"msg": [0, 3, 5], "upd": [0, 3, 5]},
}


class MissingKernelModel(nn.Module):
    def __init__(self, scenario: str) -> None:
        super().__init__()
        if scenario not in SCENARIOS:
            raise ValueError(scenario)
        self.scenario = scenario
        self.msg_indices = SCENARIOS[scenario]["msg"]
        self.upd_indices = SCENARIOS[scenario]["upd"]
        self.msg_logits = nn.Parameter(torch.tensor([0.0 if index < 5 else -1.5 for index in self.msg_indices]))
        self.upd_logits = nn.Parameter(torch.tensor([0.0 if index < 5 else -1.5 for index in self.upd_indices]))
        self.msg_mlp = nn.Sequential(nn.Linear(4, 24), nn.Tanh(), nn.Linear(24, 24), nn.Tanh(), nn.Linear(24, 1))
        self.upd_mlp = nn.Sequential(nn.Linear(2, 24), nn.Tanh(), nn.Linear(24, 24), nn.Tanh(), nn.Linear(24, 1))
        initialize_small(self.msg_mlp)
        initialize_small(self.upd_mlp)

        if 5 not in self.msg_indices:
            for parameter in self.msg_mlp.parameters():
                parameter.requires_grad_(False)
        if 5 not in self.upd_indices:
            for parameter in self.upd_mlp.parameters():
                parameter.requires_grad_(False)

    def full_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.zeros(6, device=self.msg_logits.device)
        b = torch.zeros(6, device=self.upd_logits.device)
        a[self.msg_indices] = self.msg_logits.softmax(0)
        b[self.upd_indices] = self.upd_logits.softmax(0)
        return a, b

    def components(
        self, x: torch.Tensor, dist: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dx = x.unsqueeze(-2) - x.unsqueeze(-1)
        xi = x.unsqueeze(-1).expand_as(dx)
        xj = x.unsqueeze(-2).expand_as(dx)
        msg_mlp = 0.15 * torch.tanh(
            self.msg_mlp(torch.stack([xi, xj, dist, dx], dim=-1)).squeeze(-1)
        )
        msg_stack = torch.stack(analytic_messages(x, dist) + [msg_mlp], dim=-1) * mask.unsqueeze(-1)
        a, b = self.full_weights()
        aggregate = (msg_stack * a).sum(dim=-1).sum(dim=-1)
        upd_mlp = 1.5 * DT * torch.tanh(
            self.upd_mlp(torch.stack([x, aggregate], dim=-1)).squeeze(-1)
        )
        upd_stack = torch.stack(analytic_updates(x, aggregate) + [upd_mlp], dim=-1)
        return msg_stack, a, aggregate, upd_stack, b

    def forward(self, x: torch.Tensor, dist: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, _, _, updates, b = self.components(x, dist, mask)
        return x + (updates * b).sum(dim=-1)

    def regularization(self, x: torch.Tensor, dist: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        a, b = self.full_weights()
        # A weak prior prevents an unnecessary neural branch from dominating,
        # while still allowing its weight to grow when an analytic kernel is absent.
        return 1e-4 * (a[-1].square() + b[-1].square())


def train_model(
    model: nn.Module,
    train: Dataset,
    validation: Dataset,
    budget: int,
    seed: int,
    steps: int,
) -> nn.Module:
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=6e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=4e-4)
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
        loss = F.mse_loss((pred - x) / DT, (target - x) / DT)
        if isinstance(model, MissingKernelModel):
            loss = loss + model.regularization(x, dist, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        if (step + 1) % 50 == 0 or step + 1 == steps:
            current = validation_rate_loss(model, validation_batch)
            if current < best_loss:
                best_loss = current
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def validation_rate_loss(model: nn.Module, batch: tuple[torch.Tensor, ...]) -> float:
    x, target, dist, mask = batch
    pred = model(x, dist, mask)
    return float(F.mse_loss((pred - x) / DT, (target - x) / DT))


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

    result = {
        "rollout_nrmse": float(rollout),
        "final_nrmse": float(final),
        "parameter_count": float(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }
    if isinstance(model, MissingKernelModel):
        a, b = model.full_weights()
        sample_x = data.x[:, :-1:8].reshape(-1, data.x.shape[-1])
        repeats = sample_x.shape[0] // data.x.shape[0]
        dist = data.dist[:, None].expand(-1, repeats, -1, -1).reshape(-1, data.dist.shape[-1], data.dist.shape[-1])
        mask = data.mask[:, None].expand(-1, repeats, -1, -1).reshape_as(dist)
        messages, _, _, updates, _ = model.components(sample_x, dist, mask)
        weighted_messages = messages.abs() * a
        weighted_updates = updates.abs() * b
        msg_total = weighted_messages.sum(dim=-1).mean() + 1e-12
        upd_total = weighted_updates.sum(dim=-1).mean() + 1e-12
        result.update({
            "msg_mlp_weight": float(a[-1]),
            "upd_mlp_weight": float(b[-1]),
            "msg_mlp_contribution_ratio": float(weighted_messages[..., -1].mean() / msg_total),
            "upd_mlp_contribution_ratio": float(weighted_updates[..., -1].mean() / upd_total),
            "msg_known_active_mass": float(a[ACTIVE_ANALYTIC].sum()),
            "upd_known_active_mass": float(b[ACTIVE_ANALYTIC].sum()),
            **{f"a_{index}": float(a[index]) for index in range(6)},
            **{f"b_{index}": float(b[index]) for index in range(6)},
        })
    return result


def run(args: argparse.Namespace) -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train = make_dataset(args.train_pool, args.nodes, args.steps, seed=121)
    validation = make_dataset(args.validation_trajectories, args.nodes, args.steps, seed=232)
    test = make_dataset(args.test_trajectories, args.nodes, args.steps, seed=343)
    rows: list[dict[str, float | int | str]] = []
    names = list(SCENARIOS)
    for budget in args.budgets:
        for seed in range(args.seeds):
            for model_number, name in enumerate(names):
                set_seed(10_000 + budget * 100 + seed * 20 + model_number)
                model: nn.Module = MissingKernelModel(name)
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
        "missing_kernel_index": DOMINANT_KERNEL,
        "true_message_weights": MSG_TRUE.tolist(),
        "true_update_weights": UPD_TRUE.tolist(),
        "results": records,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_results(frame: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    colors = {
        "complete": "black",
        "missing_msg_no_mlp": "#d62728",
        "missing_msg_mlp": "#2ca02c",
        "missing_upd_no_mlp": "#ff7f0e",
        "missing_upd_mlp": "#1f77b4",
        "missing_both_no_mlp": "#9467bd",
        "missing_both_mlp": "#17becf",
    }
    trained = frame[frame["budget"] > 0]
    for name, group in trained.groupby("experiment"):
        stats = group.groupby("budget")["rollout_nrmse"].agg(["mean", "std"])
        axes[0].errorbar(stats.index, stats["mean"], yerr=stats["std"], marker="o", label=name, color=colors[name])
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("training trajectories")
    axes[0].set_ylabel("rollout NRMSE")
    axes[0].set_title("Missing-kernel prediction")
    axes[0].legend(fontsize=7)

    mlp_names = ["complete", "missing_msg_mlp", "missing_upd_mlp", "missing_both_mlp"]
    for name in mlp_names:
        group = trained[trained["experiment"] == name]
        msg = group.groupby("budget")["msg_mlp_contribution_ratio"].mean()
        upd = group.groupby("budget")["upd_mlp_contribution_ratio"].mean()
        axes[1].plot(msg.index, msg.values, marker="o", label=name, color=colors[name])
        axes[2].plot(upd.index, upd.values, marker="o", label=name, color=colors[name])
    for axis, title in zip(axes[1:], ["MLP message contribution", "MLP update contribution"]):
        axis.set_xscale("log", base=2)
        axis.set_xlabel("training trajectories")
        axis.set_ylabel("contribution ratio")
        axis.set_ylim(bottom=0)
        axis.set_title(title)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "d1_missing_results.png", dpi=180)
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
    parser.add_argument("--train-steps", type=int, default=1200)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

