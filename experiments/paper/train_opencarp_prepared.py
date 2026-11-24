"""Formal E2 entry. Requires explicit --execute-training and passed preflight."""
from __future__ import annotations
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from taskkernel_gnn.data.opencarp import OpenCARPDataset
from taskkernel_gnn.data.cardiac_fem import anisotropic_fem_graph
from experiments.paper.opencarp_prepared import E2Predictor, initial_history

CONFIG = ROOT / "experiments/paper/configs/external/opencarp_prepared.yaml"
OUTPUT = ROOT / "results/SOTA_Baseline/external/opencarp/preparation"


class CaseCache:
    def __init__(self, dataset, capacity=2):
        self.dataset, self.capacity, self.items = dataset, capacity, OrderedDict()

    def __getitem__(self, index):
        if index not in self.items:
            trajectory = self.dataset[index]
            self.items[index] = (trajectory, anisotropic_fem_graph(trajectory))
        self.items.move_to_end(index)
        result = self.items[index]
        if len(self.items) > self.capacity:
            self.items.popitem(last=False)
        return result


def window_loss(model, trajectory, graph, start, length):
    history = initial_history(trajectory, start, model.history_frames, model.voltage_offset, model.voltage_scale)
    loss = history.new_zeros(())
    for frame in range(start, start + length):
        dt = float(trajectory.times[frame + 1] - trajectory.times[frame])
        history = model.step(history, graph, float(trajectory.times[frame]), dt)
        target = (trajectory.voltage[frame + 1] + model.voltage_offset) / model.voltage_scale
        loss = loss + (history[:, -1] - target).square().mean()
    return loss / length


def assert_ready(config_path=CONFIG):
    path = OUTPUT / "PROTOCOL_READINESS.json"
    if not path.is_file():
        raise RuntimeError("E2 preparation readiness receipt is missing")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not receipt.get("ready_for_formal_training"):
        raise RuntimeError("E2 preparation is not ready")
    if receipt["config_sha256"] != hashlib.sha256(config_path.read_bytes()).hexdigest():
        raise RuntimeError("E2 config changed after preflight; run preflight again")
    for relative, digest in receipt["input_manifest_hashes"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"E2 protocol changed: {relative}")
    for relative, digest in receipt["implementation_hashes"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"E2 implementation changed; rerun preflight: {relative}")
    for record in receipt["files"]:
        path = ROOT / record["path"]
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise RuntimeError(f"E2 prepared input missing or changed: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError(f"E2 data checksum changed: {path}")


@torch.no_grad()
def validation_loss(model, cache, length):
    model.eval()
    values = []
    for i in range(len(cache.dataset)):
        trajectory, graph = cache[i]
        for fraction in (.0, .2, .4, .6, .8):
            start = min(int(fraction * (len(trajectory.times) - 1)), len(trajectory.times) - length - 1)
            values.append(float(window_loss(model, trajectory, graph, start, length)))
    return float(np.mean(values))


def train(name, seed, config, output):
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(4)
    data = ROOT / config["dataset"]["root"]
    training = CaseCache(OpenCARPDataset(data, "train"), 16)
    validation = CaseCache(OpenCARPDataset(data, "validation"), 2)
    model = E2Predictor(name, config)
    model.fit_calibration([training[i] for i in range(len(training.dataset))])
    p = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
    rng = np.random.default_rng(seed)
    output.mkdir(parents=True, exist_ok=False)
    best, stale, log = float("inf"), 0, []
    beginning = time.perf_counter()
    for step in range(1, p["steps"] + 1):
        trajectory, graph = training[int(rng.integers(len(training.dataset)))]
        length = p["rollout_window"]
        start = int(rng.integers(len(trajectory.times) - length))
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = window_loss(model, trajectory, graph, start, length)
        if name == "taskkernel":
            loss = loss + model.model.gate_regularization(step / p["steps"])
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss at step {step}")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), p["max_gradient_norm"])
        if not torch.isfinite(gradient):
            raise RuntimeError(f"nonfinite gradient at step {step}")
        optimizer.step()
        if step % p["validation_interval"] == 0:
            value = validation_loss(model, validation, length)
            row = {"step": step, "train_mse_normalized": float(loss.detach()),
                   "validation_mse_normalized": value, "seconds": time.perf_counter() - beginning}
            log.append(row); print(json.dumps(row), flush=True)
            (output / "training.json").write_text(json.dumps(log, indent=2))
            if value < best:
                best, stale = value, 0
                torch.save({"state_dict": model.state_dict(), "model": name, "seed": seed,
                            "config": config, "step": step, "validation_loss": best}, output / "best.pt")
            else:
                stale += 1
            if stale >= p["early_stopping_patience"]:
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["taskkernel"], default="taskkernel")
    parser.add_argument("--seed", type=int, choices=[0, 42, 37], default=0)
    parser.add_argument("--execute-training", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if not args.execute_training:
        print(json.dumps({"training_started": False, "model": args.model, "seed": args.seed,
                          "requires": "passed preparation and explicit --execute-training"}))
        return
    assert_ready()
    train(args.model, args.seed, config,
          ROOT / f"results/SOTA_Baseline/external/opencarp/prepared_v1/{args.model}/seed_{args.seed}")

if __name__ == "__main__":
    main()
