"""Train either TaskKernel-GNN diffusion variant on PDEBench."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from taskkernel_gnn.config import TaskKernelConfig
from taskkernel_gnn.data import PDEBenchDiffusionReaction, grid_graph
from taskkernel_gnn.models import build_external_taskkernel

CONFIG_ROOT = ROOT / "experiments" / "paper" / "configs" / "external"
RESULT_ROOT = ROOT / "results" / "SOTA_Baseline" / "external"


def load_config(dataset: str, config_path: str | None = None) -> dict:
    path = Path(config_path) if config_path else CONFIG_ROOT / f"{dataset}.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if values["dataset"]["name"] != dataset:
        raise ValueError(f"dataset name mismatch in {path}")
    return values


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class PDEBenchProblem:
    def __init__(self, config: dict, budget: int, device: torch.device, split: str = "train") -> None:
        values = config["dataset"]
        root = ROOT / values["root"]
        start, stop = values["train_time"]
        self.dataset = PDEBenchDiffusionReaction.from_split(
            root / values["data_file"], root / values["split_file"],
            f"fewshot_{budget}" if split == "train" else split,
            spatial_stride=int(values["train_spatial_stride"]), time_indices=slice(start, stop),
        )
        self.device = device
        self.cache: dict[int, tuple[torch.Tensor, object]] = {}

    def item(self, index: int) -> tuple[torch.Tensor, object]:
        if index not in self.cache:
            sample = self.dataset[index]
            trajectory = torch.from_numpy(sample.data).to(self.device)
            graph = grid_graph(
                torch.from_numpy(sample.grid["x"]).to(self.device),
                torch.from_numpy(sample.grid["y"]).to(self.device),
            )
            self.cache[index] = trajectory, graph
        return self.cache[index]

    def examples(self, count: int = 4):
        result = []
        for index in range(min(count, len(self.dataset))):
            trajectory, graph = self.item(index)
            for fraction in (0.0, 0.2, 0.6):
                time_index = min(int(fraction * (len(trajectory) - 1)), len(trajectory) - 2)
                result.append((trajectory[time_index].reshape(-1, 2), graph))
        return result

    def loss(self, model, rng: random.Random, rollout: int, progress: float,
             force_start: int | None = None) -> torch.Tensor:
        trajectory, graph = self.item(rng.randrange(len(self.dataset)))
        length = min(rollout, len(trajectory) - 1)
        start = force_start if force_start is not None else (
            0 if rng.random() < 0.5 else rng.randrange(0, len(trajectory) - length)
        )
        prediction = trajectory[start]
        losses = []
        for offset in range(length):
            flat = prediction.reshape(-1, prediction.shape[-1])
            prediction = model.step(flat, graph, progress=progress).reshape_as(prediction)
            losses.append(F.mse_loss(prediction, trajectory[start + offset + 1]))
        return torch.stack(losses).mean()

    def close(self) -> None:
        self.dataset.close()


def optimizer_for(model, config: dict):
    options = config["taskkernel"]["optimizer"]
    base, gates = [], []
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (gates if "gate" in parameter_name else base).append(parameter)
    learning_rate = float(options["learning_rate"])
    groups = [
        {"params": base, "lr": learning_rate},
        {"params": gates, "lr": learning_rate * float(options["gate_learning_rate_scale"])},
    ]
    return torch.optim.Adam(groups, weight_decay=float(options["weight_decay"]))


def parse_rollout_schedule(value: str | None, steps: int) -> list[tuple[int, int]] | None:
    if value is None:
        return None
    schedule = []
    for part in value.split(","):
        horizon, ending = part.split(":")
        horizon, ending = int(horizon), int(ending)
        if horizon < 1 or ending < 1 or (schedule and ending <= schedule[-1][1]):
            raise ValueError("rollout schedule needs positive horizons and increasing end steps")
        schedule.append((horizon, ending))
    if not schedule or schedule[-1][1] != steps:
        raise ValueError("last rollout schedule end step must equal total training steps")
    return schedule


def run(args: argparse.Namespace) -> dict:
    config = load_config(args.dataset, getattr(args, "config", None))
    allowed_budgets = tuple(config["training"]["budgets"])
    if args.budget not in allowed_budgets:
        raise ValueError(f"budget must be one of {allowed_budgets}")
    device = torch.device(args.device)
    set_seed(args.seed)
    model = build_external_taskkernel("pdebench", TaskKernelConfig.from_dict(config["taskkernel"])).to(device)
    if not args.dry_run and not args.execute_training:
        return {
            "mode": "plan_only", "training_started": False, "dataset": args.dataset,
            "model": args.model, "budget": args.budget, "seed": args.seed,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "message": "No data were loaded and no calibration or optimization was run.",
        }
    problem = PDEBenchProblem(config, args.budget, device)
    validation_problem = None
    try:
        calibration = model.fit_calibration(problem.examples())
        initial_checkpoint = getattr(args, "init_checkpoint", None)
        if initial_checkpoint:
            checkpoint = torch.load(initial_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
        rollout = int(config["training"].get("rollout_window", 1))
        rng = random.Random(args.seed)
        if not args.execute_training:
            report = {
                "mode": "forward_only_dry_run",
                "training_started": False,
                "dataset": args.dataset,
                "model": args.model,
                "budget": args.budget,
                "seed": args.seed,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "calibration": calibration,
            }
            if args.dry_run:
                model.eval()
                with torch.no_grad():
                    loss = problem.loss(model, rng, min(rollout, 2), 0.0)
                report.update({"finite_loss": bool(torch.isfinite(loss)), "forward_loss": float(loss)})
            return report

        steps = args.steps or int(config["training"]["steps"])
        rollout_schedule = parse_rollout_schedule(getattr(args, "rollout_schedule", None), steps)
        interval_override = getattr(args, "validation_interval", None)
        validation_interval = int(config["training"]["validation_interval"] if interval_override is None
                                  else interval_override)
        if validation_interval < 1:
            raise ValueError("validation interval must be positive")
        maximum_gradient = float(config.get("taskkernel", {}).get("optimizer", {}).get("max_gradient_norm", 2.0))
        validation_problem = PDEBenchProblem(config, args.budget, device, "validation")
        optimizer = optimizer_for(model, config)
        output_name = getattr(args, "output_name", None) or args.model
        output = RESULT_ROOT / args.dataset / output_name / f"budget_{args.budget}" / f"seed_{args.seed}"
        output.mkdir(parents=True, exist_ok=True)
        metrics = []
        best = float("inf")
        patience = 0 if getattr(args, "disable_early_stopping", False) else int(config["training"].get("early_stopping_patience", 0))
        stale_validations = 0
        stopped_early = False
        termination_reason = "maximum_steps"
        started = time.perf_counter()
        model.train()
        for step in range(steps):
            progress = step / max(steps - 1, 1)
            current_rollout = next(
                (horizon for horizon, ending in rollout_schedule if step < ending), rollout
            ) if rollout_schedule else rollout
            optimizer.zero_grad(set_to_none=True)
            prediction_loss = problem.loss(
                model, rng, current_rollout, progress,
            )
            regularization = model.gate_regularization(progress)
            loss = prediction_loss + regularization
            if not bool(torch.isfinite(loss)):
                metrics.append({
                    "step": step + 1,
                    "train_loss": float(prediction_loss.detach()),
                    "validation_loss": float("nan"),
                })
                stopped_early = True
                termination_reason = "nonfinite_training_loss"
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), maximum_gradient)
            optimizer.step()
            if step == 0 or (step + 1) % validation_interval == 0 or step + 1 == steps:
                model.eval()
                validation_rng = random.Random(args.seed + 1000003)
                with torch.no_grad():
                    validation_batches = int(config["training"].get("validation_batches", 1))
                    validation_losses = []
                    validation_losses.append(validation_problem.loss(
                        model, validation_rng, min(rollout, 10), progress, force_start=0,
                    ))
                    while len(validation_losses) < validation_batches:
                        validation_losses.append(validation_problem.loss(
                            model, validation_rng, min(rollout, 10), progress
                        ))
                    validation = float(torch.stack(validation_losses).mean())
                row = {"step": step + 1, "train_loss": float(prediction_loss.detach()), "validation_loss": validation}
                if rollout_schedule:
                    row["training_rollout"] = current_rollout
                metrics.append(row)
                if validation < best:
                    best = validation
                    stale_validations = 0
                    torch.save({"model": model.state_dict(), "config": config, "calibration": calibration}, output / "best.pt")
                else:
                    stale_validations += 1
                model.train()
                if patience and stale_validations >= patience:
                    stopped_early = True
                    termination_reason = "validation_patience"
                    break
        report = {
            "mode": "training", "training_started": True, "dataset": args.dataset,
            "model": args.model, "budget": args.budget, "seed": args.seed,
            "maximum_steps": steps, "completed_steps": metrics[-1]["step"],
            "stopped_early": stopped_early,
            "termination_reason": termination_reason,
            "best_validation_loss": best, "elapsed_seconds": time.perf_counter() - started,
            "initial_checkpoint": initial_checkpoint, "rollout_schedule": rollout_schedule,
            "calibration": calibration, "metrics": metrics,
        }
        (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        problem.close()
        if validation_problem is not None:
            validation_problem.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("pdebench",), default="pdebench")
    parser.add_argument("--model", choices=("taskkernel",), default="taskkernel")
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, help="explicit training-step override")
    parser.add_argument("--validation-interval", type=int,
                        help="checkpoint and validation interval override")
    parser.add_argument("--config", help="optional YAML configuration path")
    parser.add_argument("--output-name", help="optional result directory name; model construction is unchanged")
    parser.add_argument("--init-checkpoint", help="load model weights before training; optimizer starts fresh")
    parser.add_argument("--rollout-schedule", help="training horizon:end step pairs, e.g. 10:500,20:1500,30:3000")
    parser.add_argument("--disable-early-stopping", action="store_true", help="run until the step limit or a nonfinite loss")
    parser.add_argument("--dry-run", action="store_true", help="run calibration and one forward loss only")
    parser.add_argument(
        "--execute-training", action="store_true",
        help="required safety gate for backward passes, optimizer updates, and checkpoints",
    )
    args = parser.parse_args()
    if args.dry_run and args.execute_training:
        parser.error("--dry-run and --execute-training are mutually exclusive")
    report = run(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
