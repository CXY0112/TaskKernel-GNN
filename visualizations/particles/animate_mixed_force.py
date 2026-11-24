"""Render reference and TaskKernel-GNN mixed-force particle rollouts side by side."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments" / "d3_particle_dynamics"
RESULTS = ROOT / "results" / "Ablation" / "d3" / "particles" / "formal"
CHECKPOINT = RESULTS / "checkpoints" / "taskkernel_gnn_seed0.pt"
OUTPUT = RESULTS / "mixed_force_evolution.gif"
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(ROOT / "src"))

from d3_particle_formal import (  # noqa: E402
    DT,
    HORIZON_DEFAULT,
    TaskKernelGNN,
    batch_state,
    generate_trajectory,
)


@torch.no_grad()
def predict(model: TaskKernelGNN, trajectory) -> np.ndarray:
    position = torch.tensor(trajectory.position[0])
    velocity = torch.tensor(trajectory.velocity[0])
    positions = [position.numpy().copy()]
    for step in range(len(trajectory.force)):
        batch = batch_state(trajectory, step, position, velocity)
        next_position, next_velocity = model.step(batch, 1.0)
        position, velocity = next_position[0], next_velocity[0]
        if not torch.isfinite(position).all() or not torch.isfinite(velocity).all():
            raise RuntimeError(f"TaskKernel-GNN rollout became unstable at step {step + 1}")
        positions.append(position.numpy().copy())
    return np.stack(positions)


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing {CHECKPOINT}. Recreate the frozen D3 seed-0 checkpoint before rendering."
        )

    # This is the first ID trajectory from the frozen formal D3 split.
    trajectory = generate_trajectory("ID", 30_000, 6, "random", HORIZON_DEFAULT, "nominal")
    settings = json.loads((RESULTS / "settings.json").read_text(encoding="utf-8"))
    model = TaskKernelGNN(settings["calibration"])
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    model.eval()

    truth = trajectory.position
    predicted = predict(model, trajectory)
    position_nrmse = float(
        np.sqrt(np.mean((predicted[1:] - truth[1:]) ** 2))
        / max(np.sqrt(np.mean(truth[1:] ** 2)), 1e-8)
    )
    frame_indices = np.linspace(0, len(truth) - 1, 72).astype(int)
    all_positions = np.concatenate([truth, predicted], axis=1)
    margin = 0.18
    limits = (
        all_positions[..., 0].min() - margin,
        all_positions[..., 0].max() + margin,
        all_positions[..., 1].min() - margin,
        all_positions[..., 1].max() + margin,
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.6))
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.09, top=0.86, wspace=0.12)
    colors = np.arange(len(trajectory.system.mass))

    def update(frame: int):
        index = int(frame_indices[frame])
        frame_rmse = float(np.sqrt(np.mean((predicted[index] - truth[index]) ** 2)))
        panels = (
            (axes[0], truth[index], "Reference solution"),
            (axes[1], predicted[index], "TaskKernel-GNN prediction"),
        )
        artists = []
        for ax, points, title in panels:
            ax.clear()
            for i, j in trajectory.system.edges:
                line = ax.plot(
                    points[[i, j], 0],
                    points[[i, j], 1],
                    color="#858585",
                    linewidth=1.4,
                    zorder=1,
                )[0]
                artists.append(line)
            scatter = ax.scatter(
                points[:, 0],
                points[:, 1],
                s=130 * trajectory.system.mass,
                c=colors,
                cmap="viridis",
                vmin=0,
                vmax=len(trajectory.system.mass) - 1,
                edgecolor="white",
                linewidth=0.8,
                zorder=2,
            )
            artists.append(scatter)
            ax.set(xlim=limits[:2], ylim=limits[2:], xlabel="x", ylabel="y")
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
            ax.set_title(title)
            ax.text(
                0.02,
                0.02,
                "all pairs: gravity\nbonds: spring + dashpot",
                transform=ax.transAxes,
                fontsize=9,
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
        fig.suptitle(
            f"Mixed-force particle dynamics   t={index * DT:.2f}   frame position RMSE={frame_rmse:.5f}",
            fontsize=13,
        )
        return tuple(artists)

    animation = FuncAnimation(fig, update, frames=len(frame_indices), interval=75)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    animation.save(OUTPUT, writer=PillowWriter(fps=13), dpi=105)
    plt.close(fig)
    print(f"saved {OUTPUT}")
    print(f"rollout position NRMSE={position_nrmse:.6f}")


if __name__ == "__main__":
    main()
