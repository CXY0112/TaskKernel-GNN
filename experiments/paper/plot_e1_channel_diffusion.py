"""Compare original and channel-specific diffusion E1 checkpoints visually."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from taskkernel_gnn.config import TaskKernelConfig
from taskkernel_gnn.data import PDEBenchDiffusionReaction, grid_graph
from taskkernel_gnn.models import build_external_taskkernel

BASE = ROOT / "results/SOTA_Baseline/external/pdebench"
OUT = BASE / "taskkernel_channel_diffusion"
CONFIG_DIR = ROOT / "experiments/paper/configs/external"


def load_model(seed: int, variant: str):
    config_name = "pdebench_channel_diffusion.yaml" if variant == "new" else "pdebench.yaml"
    config = yaml.safe_load((CONFIG_DIR / config_name).read_text(encoding="utf-8"))
    run_name = "taskkernel_channel_diffusion" if variant == "new" else "taskkernel"
    checkpoint = torch.load(BASE / run_name / "budget_16" / f"seed_{seed}" / "best.pt",
                            map_location="cpu", weights_only=False)
    model = build_external_taskkernel("pdebench", TaskKernelConfig.from_dict(config["taskkernel"]))
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def load_sample(suite: str):
    config = yaml.safe_load((CONFIG_DIR / "pdebench_channel_diffusion.yaml").read_text(encoding="utf-8"))
    values = config["dataset"]
    path = ROOT / values["root"]
    stride = values["mesh_ood_spatial_stride"] if suite == "mesh_ood" else values["id_spatial_stride"]
    frames = values["temporal_ood_time"] if suite == "temporal_ood" else values["train_time"]
    dataset = PDEBenchDiffusionReaction.from_split(
        path / values["data_file"], path / values["split_file"], "validation",
        spatial_stride=int(stride), time_indices=slice(*frames),
    )
    try:
        sample = dataset[0]
        trajectory = torch.from_numpy(sample.data)
        graph = grid_graph(torch.from_numpy(sample.grid["x"]), torch.from_numpy(sample.grid["y"]))
        return trajectory, graph
    finally:
        dataset.close()


@torch.no_grad()
def snapshots(model, trajectory, graph, suite: str):
    prediction = trajectory[0]
    result = {}
    for step in range(1, 51):
        prediction = model.step(prediction.reshape(-1, 2), graph,
                                substeps=4 if suite == "mesh_ood" else 1).reshape_as(prediction)
        if step in (1, 10, 50):
            result[step] = prediction[..., 0].numpy().copy()
    return result


def draw_field(axis, field, limit):
    axis.imshow(field, cmap="coolwarm", vmin=-limit, vmax=limit, origin="lower",
                interpolation="nearest")
    axis.set_xticks([])
    axis.set_yticks([])


def plot_id_seed(seed: int, trajectory, graph):
    old = snapshots(load_model(seed, "old"), trajectory, graph, "id")
    new = snapshots(load_model(seed, "new"), trajectory, graph, "id")
    fig, axes = plt.subplots(3, 3, figsize=(10, 9.2))
    for row, step in enumerate((1, 10, 50)):
        reference = trajectory[step, ..., 0].numpy()
        limit = max(np.abs(reference).max(), 1e-9)
        for col, field in enumerate((reference, old[step], new[step])):
            draw_field(axes[row, col], field, limit)
        axes[row, 0].set_ylabel(f"step {step}", fontsize=12)
    for col, title in enumerate(("Reference", "Original shared D", "New channel D")):
        axes[0, col].set_title(title, fontsize=12)
    fig.suptitle(f"E1 ID validation | trajectory 0 | seed {seed} | field u\n"
                 "Each row uses the reference color range", fontsize=13)
    fig.tight_layout()
    output = OUT / f"qualitative_id_seed_{seed}.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output, flush=True)


def plot_ood_suite(suite: str, trajectory, graph):
    fig, axes = plt.subplots(3, 3, figsize=(10, 9.2))
    reference = trajectory[50, ..., 0].numpy()
    limit = max(np.abs(reference).max(), 1e-9)
    for seed in (0, 1, 2):
        old = snapshots(load_model(seed, "old"), trajectory, graph, suite)[50]
        new = snapshots(load_model(seed, "new"), trajectory, graph, suite)[50]
        for col, field in enumerate((reference, old, new)):
            draw_field(axes[seed, col], field, limit)
        axes[seed, 0].set_ylabel(f"seed {seed}", fontsize=12)
    for col, title in enumerate(("Reference", "Original shared D", "New channel D")):
        axes[0, col].set_title(title, fontsize=12)
    label = "Temporal OOD: frames 50-100" if suite == "temporal_ood" else "Mesh OOD: spatial stride 1"
    fig.suptitle(f"E1 {label} | step 50 | trajectory 0 | field u\n"
                 "Same reference color range in every panel", fontsize=13)
    fig.tight_layout()
    output = OUT / f"qualitative_{suite}_seeds.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output, flush=True)


def main():
    torch.set_num_threads(4)
    trajectory, graph = load_sample("id")
    for seed in (0, 1, 2):
        plot_id_seed(seed, trajectory, graph)
    for suite in ("temporal_ood", "mesh_ood"):
        trajectory, graph = load_sample(suite)
        plot_ood_suite(suite, trajectory, graph)


if __name__ == "__main__":
    main()
