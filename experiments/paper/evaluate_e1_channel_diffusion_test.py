"""Evaluate a paper TaskKernel-GNN model on the independent PDEBench test set.

Writes per-sample records after every trajectory so interrupted runs can resume.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from taskkernel_gnn.config import TaskKernelConfig
from taskkernel_gnn.data import PDEBenchDiffusionReaction, grid_graph
from taskkernel_gnn.models import build_external_taskkernel
from e1_structure_metrics import structure_metrics
from e1_morphology_metrics import morphology_metrics

CONFIG_DIR = Path(__file__).resolve().parent / "configs/external"
RESULT_DIR = ROOT / "results/SOTA_Baseline/external/pdebench"
VARIANTS = {
    "shared": ("pdebench.yaml", "taskkernel"),
    "channel-wise": ("pdebench_channel_diffusion.yaml", "taskkernel_channel_diffusion"),
}
HORIZONS = (1, 10, 50)
FIELDS = (
    "seed", "suite", "sample_index", "sample_key", "horizon", "rollout_mse", "final_mse",
    "relative_l2", "finite", "ssim", "amplitude_ratio", "spectral_error", "gradient_ratio",
    "minkowski_distance", "minkowski_area_distance", "minkowski_perimeter_distance",
    "minkowski_euler_distance", "persistence_wasserstein", "spectral_wasserstein", "collapsed",
)


def build_model(path: Path, values: dict):
    model = build_external_taskkernel("pdebench", TaskKernelConfig.from_dict(values["taskkernel"]))
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    return model.eval()


@torch.no_grad()
def score_sample(model, trajectory, graph, substeps: int, seed: int, suite: str, index: int, key: str):
    prediction = trajectory[0]
    cumulative_mse = 0.0
    rows = []
    for step in range(1, 51):
        prediction = model.step(prediction.reshape(-1, 2), graph, substeps=substeps).reshape_as(prediction)
        target = trajectory[step]
        error = prediction - target
        step_mse = float(error.square().mean())
        cumulative_mse += step_mse
        if step not in HORIZONS:
            continue
        finite = math.isfinite(cumulative_mse) and bool(torch.isfinite(prediction).all())
        base = {"seed": seed, "suite": suite, "sample_index": index, "sample_key": key,
                "horizon": step, "rollout_mse": cumulative_mse / step, "final_mse": step_mse,
                "relative_l2": float(error.norm() / target.norm().clamp_min(1e-12)),
                "finite": int(finite)}
        if finite:
            shape = structure_metrics(prediction, target)
            base.update(shape)
            base["collapsed"] = int(shape["amplitude_ratio"] < 0.2)
            if step == 50:
                base.update(morphology_metrics(prediction, target))
        rows.append(base)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="channel-wise")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--sample-limit", type=int, default=200)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    config_name, output_name = VARIANTS[args.variant]
    values = yaml.safe_load((CONFIG_DIR / config_name).read_text(encoding="utf-8"))
    run_dir = RESULT_DIR / output_name
    model = build_model(run_dir / "budget_16" / f"seed_{args.seed}" / "best.pt", values)
    path = run_dir / f"test_per_sample_seed_{args.seed}.csv"
    completed = set()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            completed = {(row["suite"], int(row["sample_index"])) for row in csv.DictReader(stream)
                         if int(row["horizon"]) == 50}
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, FIELDS)
        if not path.stat().st_size:
            writer.writeheader()
        for suite in ("id", "temporal_ood", "mesh_ood"):
            spec = values["evaluation"]["suites"][suite]
            start, stop = spec["time"]
            root = ROOT / values["dataset"]["root"]
            dataset = PDEBenchDiffusionReaction.from_split(
                root / values["dataset"]["data_file"], root / values["dataset"]["split_file"],
                "test", spatial_stride=int(spec["spatial_stride"]), time_indices=slice(start, stop),
            )
            try:
                total = min(args.sample_limit, len(dataset))
                for index in range(total):
                    if (suite, index) in completed:
                        continue
                    sample = dataset[index]
                    trajectory = torch.from_numpy(sample.data)
                    graph = grid_graph(torch.from_numpy(sample.grid["x"]),
                                       torch.from_numpy(sample.grid["y"]))
                    rows = score_sample(model, trajectory, graph,
                                        int(spec.get("taskkernel_substeps", 1)), args.seed,
                                        suite, index, sample.key)
                    writer.writerows(rows)
                    stream.flush()
                    if (index + 1) % 10 == 0 or index + 1 == total:
                        print(json.dumps({"seed": args.seed, "suite": suite,
                                          "completed": index + 1, "total": total}), flush=True)
            finally:
                dataset.close()
    print(path, flush=True)


if __name__ == "__main__":
    main()
