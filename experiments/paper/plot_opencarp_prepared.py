"""Fixed-color 3D reference/model comparisons and an optional paired GIF."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def visible_points(points):
    # Show the three near faces as a surface visualization, without interpolation.
    low, high = points.min(0), points.max(0)
    return np.isclose(points[:, 0], high[0]) | np.isclose(points[:, 1], low[1]) | np.isclose(points[:, 2], high[2])


def axes3d(ax, points, values, title):
    mask = visible_points(points)
    scatter = ax.scatter(*points[mask].T, c=values[mask], vmin=-95, vmax=60,
                         cmap="turbo", s=12, linewidths=0, depthshade=False)
    ax.set(xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)", title=title)
    ax.set_box_aspect([20, 7, 3]); ax.view_init(elev=25, azim=-65)
    ax.set(xlim=(0,20), ylim=(0,7), zlim=(0,3))
    ax.set_xticks([0,10,20]); ax.set_yticks([0,3.5,7]); ax.set_zticks([0,1.5,3])
    ax.tick_params(labelsize=9, pad=0)
    return scatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.input, allow_pickle=False) as data:
        if args.reference_only:
            points, reference, times = data["points"] / 1000., data["voltage"], data["times"]
            prediction = None
        else:
            points, reference, prediction, times = data["points_mm"], data["reference_mv"], data["prediction_mv"], data["times_ms"]
    if args.reference_only:
        fig = plt.figure(figsize=(12, 8))
        for panel, requested in enumerate([0, 10, 30, 50], 1):
            frame = int(np.abs(times-requested).argmin())
            scatter = axes3d(fig.add_subplot(2,2,panel,projection="3d"), points, reference[frame],
                             f"openCARP reference, t={times[frame]:g} ms")
        fig.subplots_adjust(left=.02, right=.83, top=.93, bottom=.06, wspace=.08)
        fig.colorbar(scatter, cax=fig.add_axes([.9,.22,.02,.56]), label="Vm (mV)")
        fig.savefig(args.output / "reference_3d.png", dpi=160); plt.close(fig)
        return
    fig = plt.figure(figsize=(12,12))
    for row, requested in enumerate([10,30,50]):
        frame = int(np.abs(times-requested).argmin())
        left = fig.add_subplot(3,2,2*row+1,projection="3d")
        right = fig.add_subplot(3,2,2*row+2,projection="3d")
        scatter = axes3d(left, points, reference[frame], f"Reference, t={times[frame]:g} ms")
        axes3d(right, points, prediction[frame], f"Model, t={times[frame]:g} ms")
    fig.subplots_adjust(left=.02, right=.83, top=.93, bottom=.06, wspace=.08)
    fig.colorbar(scatter, cax=fig.add_axes([.9,.22,.02,.56]), label="Vm (mV)")
    fig.savefig(args.output / "comparison_3d.png", dpi=160); plt.close(fig)
    if args.gif:
        fig = plt.figure(figsize=(12,5))
        left, right = [fig.add_subplot(1,2,i,projection="3d") for i in (1,2)]
        a = axes3d(left, points, reference[0], "Reference")
        b = axes3d(right, points, prediction[0], "Model")
        mask = visible_points(points)
        fig.colorbar(a, ax=[left,right], shrink=.6, label="Vm (mV)")
        def update(frame):
            a.set_array(reference[frame,mask]); b.set_array(prediction[frame,mask])
            fig.suptitle(f"t={times[frame]:g} ms; identical color limits")
            return a,b
        animation = FuncAnimation(fig, update, frames=range(len(times)), interval=100, blit=False)
        animation.save(args.output / "comparison_3d.gif", writer=PillowWriter(fps=10))
        plt.close(fig)

if __name__ == "__main__":
    main()
