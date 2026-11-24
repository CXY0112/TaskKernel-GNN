"""D2 structural ablations for TaskKernel-GNN message and update kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from d2_fixed_task import (
    DT,
    KINDS,
    Sample,
    evaluate,
    initialize_small,
    inv_softplus,
    make_split,
    model_metadata,
    positive,
    scatter_add,
    set_seed,
    train_model,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "Ablation" / "d2" / "graph" / "ablation"


def edge_bases(x: torch.Tensor, sample: Sample) -> list[torch.Tensor]:
    dx = x[sample.src] - x[sample.dst]
    return [
        sample.weight * dx,
        torch.exp(-2.0 * sample.distance.square()) * dx,
        dx / torch.sqrt(sample.degree[sample.src] * sample.degree[sample.dst]),
    ]


def reaction_bases(model: nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
    return [
        positive(model.rho1_raw) * x - positive(model.rho2_raw) * x.pow(3),
        positive(model.logistic_rate_raw) * x * (1.0 - x / positive(model.capacity_raw)),
        -positive(model.decay_raw) * x,
    ]


class MultiMessage(nn.Module):
    """B3: multiple analytic message kernels and one cubic update kernel."""

    name = "B3-MultiMsg"

    def __init__(self) -> None:
        super().__init__()
        self.msg_logits = nn.Parameter(torch.zeros(3))
        self.msg_scales_raw = nn.Parameter(torch.stack([inv_softplus(0.10)] * 3))
        self.rho1_raw = nn.Parameter(inv_softplus(0.40))
        self.rho2_raw = nn.Parameter(inv_softplus(0.50))

    def step(self, x: torch.Tensor, sample: Sample) -> torch.Tensor:
        messages = torch.stack(edge_bases(x, sample), dim=-1) * positive(self.msg_scales_raw)
        message = (messages * self.msg_logits.softmax(0)).sum(-1)
        aggregate = scatter_add(message, sample.dst, sample.nodes)
        reaction = positive(self.rho1_raw) * x - positive(self.rho2_raw) * x.pow(3)
        return x + DT * (aggregate + reaction)


class MultiUpdate(nn.Module):
    """B4: one weighted-diffusion message kernel and multiple analytic updates."""

    name = "B4-MultiUpd"

    def __init__(self) -> None:
        super().__init__()
        self.diffusion_raw = nn.Parameter(inv_softplus(0.08))
        self.upd_logits = nn.Parameter(torch.zeros(3))
        self.rho1_raw = nn.Parameter(inv_softplus(0.45))
        self.rho2_raw = nn.Parameter(inv_softplus(0.60))
        self.logistic_rate_raw = nn.Parameter(inv_softplus(0.40))
        self.capacity_raw = nn.Parameter(inv_softplus(1.50))
        self.decay_raw = nn.Parameter(inv_softplus(0.20))

    def step(self, x: torch.Tensor, sample: Sample) -> torch.Tensor:
        message = positive(self.diffusion_raw) * edge_bases(x, sample)[0]
        aggregate = scatter_add(message, sample.dst, sample.nodes)
        reaction = (torch.stack(reaction_bases(self, x), dim=-1) * self.upd_logits.softmax(0)).sum(-1)
        return x + DT * (aggregate + reaction)


def metadata(model: nn.Module) -> dict[str, float]:
    result = {"parameter_count": float(sum(parameter.numel() for parameter in model.parameters()))}
    if isinstance(model, MultiMessage):
        weights = model.msg_logits.softmax(0).detach()
        result.update({f"a_{i}": float(weights[i]) for i in range(3)})
        result.update({f"msg_c_{i}": float(positive(model.msg_scales_raw[i]).detach()) for i in range(3)})
        result.update({
            "learned_rho1": float(positive(model.rho1_raw).detach()),
            "learned_rho2": float(positive(model.rho2_raw).detach()),
        })
    elif isinstance(model, MultiUpdate):
        weights = model.upd_logits.softmax(0).detach()
        result.update({f"b_{i}": float(weights[i]) for i in range(3)})
        result["learned_D"] = float(positive(model.diffusion_raw).detach())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--train-count", type=int, default=12)
    parser.add_argument("--validation-count", type=int, default=2)
    parser.add_argument("--test-count", type=int, default=3)
    args = parser.parse_args()
    if args.steps < 200:
        parser.error("--steps must be at least 200 for the paper's evaluation horizons")
    RESULTS.mkdir(parents=True, exist_ok=True)
    train = make_split(args.train_count, (32, 64), 11_000, args.steps)
    validation = make_split(args.validation_count, (32, 64), 22_000, args.steps)
    test = make_split(args.test_count, (32, 64), 33_000, args.steps)

    constructors = ((1, MultiMessage), (2, MultiUpdate))
    rows: list[dict[str, float | int | str]] = []
    for budget in args.budgets:
        for seed in range(args.seeds):
            for model_index, constructor in constructors:
                set_seed(200_000 + budget * 100 + seed * 10 + model_index)
                model = constructor()
                train_model(model, train, validation, budget, seed, args.train_steps, args.window)
                info = metadata(model)
                for metrics in evaluate(model, test):
                    rows.append({"model": model.name, "budget": budget, "seed": seed, **info, **metrics})
                print(f"finished {model.name} budget={budget} seed={seed}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "runs.csv", index=False)
    metrics = ["one_step_nrmse", "nrmse_20", "nrmse_50", "nrmse_100", "nrmse_200",
               "rollout_nrmse", "final_nrmse", "mass_change_rmse", "exploded"]
    per_run = frame.groupby(["model", "budget", "seed"])[metrics].mean().reset_index()
    per_run.to_csv(RESULTS / "per_run.csv", index=False)
    per_run.groupby(["model", "budget"])[metrics].agg(["mean", "std"]).to_csv(RESULTS / "summary.csv")
    settings = vars(args) | {
        "stage": "D2-A structural ablations",
        "status": "complete",
        "aggregation": "ADD",
        "validation_selection": "all balanced held-out validation trajectories",
        "models": [constructor().name for _, constructor in constructors],
    }
    (RESULTS / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(per_run.groupby(["model", "budget"])["rollout_nrmse"].agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()

