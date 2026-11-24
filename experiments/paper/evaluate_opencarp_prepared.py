"""Evaluate an authorized, trained E2 checkpoint; never performs training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from taskkernel_gnn.data.opencarp import OpenCARPDataset
from experiments.paper.opencarp_prepared import E2Predictor, initial_history
from experiments.paper.train_opencarp_prepared import CaseCache


def sampled_activation(voltage_mv, times_ms, threshold=0.):
    """First upward crossing with linear interpolation; absent => NaN."""
    above = voltage_mv >= threshold
    crossing = above[1:] & ~above[:-1]
    present = crossing.any(0)
    index = crossing.float().argmax(0)
    node = torch.arange(voltage_mv.shape[1])
    before, after = voltage_mv[index, node], voltage_mv[index + 1, node]
    fraction = (threshold - before) / (after - before).clamp_min(1e-12)
    activation = times_ms[index] + fraction * (times_ms[index + 1] - times_ms[index])
    return torch.where(present, activation, torch.full_like(activation, float("nan")))


def metrics(prediction, reference, times):
    error = prediction - reference
    mse = float(error.square().mean())
    lat_pred, lat_ref = sampled_activation(prediction, times), sampled_activation(reference, times)
    joint = torch.isfinite(lat_pred) & torch.isfinite(lat_ref)
    expected, actual = torch.isfinite(lat_ref), torch.isfinite(lat_pred)
    return {"mse_mv2": mse, "rmse_mv": mse ** .5,
            "relative_l2": float(error.norm() / reference.norm().clamp_min(1e-8)),
            "activation_time_mae_ms_jointly_activated": float((lat_pred[joint] - lat_ref[joint]).abs().mean()) if joint.any() else None,
            "reference_activation_fraction": float(expected.float().mean()),
            "predicted_activation_fraction": float(actual.float().mean()),
            "missed_activation_fraction": float((expected & ~actual).float().mean()),
            "spurious_activation_fraction": float((~expected & actual).float().mean()),
            "jointly_activated_nodes": int(joint.sum())}


@torch.no_grad()
def evaluate(checkpoint, output):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = E2Predictor(payload["model"], config)
    model.load_state_dict(payload["state_dict"]); model.eval()
    torch.set_num_threads(4)
    rows = []
    for suite in config["evaluation"]["suites"]:
        cache = CaseCache(OpenCARPDataset(ROOT / config["dataset"]["root"], suite), 1)
        for index in range(len(cache.dataset)):
            trajectory, graph = cache[index]
            history = initial_history(trajectory, frames=model.history_frames, offset=model.voltage_offset, scale=model.voltage_scale)
            values = [trajectory.voltage[0]]
            for frame in range(max(config["evaluation"]["rollout_steps"])):
                history = model.step(history, graph, float(trajectory.times[frame]))
                voltage = history[:, -1] * model.voltage_scale - model.voltage_offset
                if not torch.isfinite(voltage).all():
                    raise RuntimeError(f"nonfinite rollout: {suite}/{trajectory.name}/{frame+1}")
                values.append(voltage)
                if frame + 1 in config["evaluation"]["rollout_steps"]:
                    predicted = torch.stack(values)
                    reference = trajectory.voltage[:len(values)]
                    row = {"suite": suite, "case": trajectory.name, "horizon_steps": frame + 1,
                           **metrics(predicted, reference, trajectory.times[:len(values)])}
                    rows.append(row)
            if index == 0:
                output.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output / f"qualitative_{suite}.npz", points_mm=graph.coordinates.numpy(),
                                    reference_mv=reference.numpy(), prediction_mv=predicted.numpy(),
                                    times_ms=trajectory.times[:len(values)].numpy())
            print(f"{suite}: {index+1}/{len(cache.dataset)}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps({"model": payload["model"], "seed": payload["seed"],
        "parameters": sum(p.numel() for p in model.parameters()), "rows": rows}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output or args.checkpoint.parent / "evaluation")

if __name__ == "__main__":
    main()
