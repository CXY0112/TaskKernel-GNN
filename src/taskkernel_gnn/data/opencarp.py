"""Prepared openCARP trajectory format used by the external benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .graph import GraphSample, tetrahedral_edges


@dataclass(frozen=True)
class OpenCARPTrajectory:
    name: str
    points: torch.Tensor
    tetrahedra: torch.Tensor
    voltage: torch.Tensor
    times: torch.Tensor
    parameters: dict[str, float | str]
    activation_times: torch.Tensor | None = None

    def graph(self) -> GraphSample:
        # Official CARP PTS coordinates are micrometres; model geometry uses millimetres.
        unit = self.parameters.get("coordinate_unit", "um")
        if unit not in {"um", "mm"}:
            raise ValueError(f"unsupported openCARP coordinate unit: {unit}")
        coordinates = self.points * (0.001 if unit == "um" else 1.0)
        graph = tetrahedral_edges(coordinates, self.tetrahedra)
        graph.trajectory = self.voltage.unsqueeze(-1)
        graph.context.update({"position": coordinates, "parameters": self.parameters,
                              "coordinate_unit": "mm"})
        if self.activation_times is not None:
            graph.context["native_activation_times_ms"] = self.activation_times
        return graph


class OpenCARPDataset:
    """Load generated trajectories exported as compressed, schema-checked NPZ files."""

    REQUIRED = ("points", "tetrahedra", "voltage", "times")

    def __init__(self, directory: str | Path, split: str) -> None:
        self.directory = Path(directory)
        split_file = self.directory / "splits.json"
        if not split_file.is_file():
            raise FileNotFoundError(
                f"openCARP trajectories are not generated yet: missing {split_file}"
            )
        manifest = json.loads(split_file.read_text(encoding="utf-8"))
        self.files = tuple(self.directory / name for name in manifest["splits"][split])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, item: int) -> OpenCARPTrajectory:
        path = self.files[item]
        with np.load(path, allow_pickle=False) as data:
            missing = set(self.REQUIRED) - set(data.files)
            if missing:
                raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
            parameters = json.loads(str(data["parameters_json"])) if "parameters_json" in data else {}
            result = OpenCARPTrajectory(
                path.stem,
                torch.from_numpy(data["points"].astype(np.float32, copy=False)),
                torch.from_numpy(data["tetrahedra"].astype(np.int64, copy=False)),
                torch.from_numpy(data["voltage"].astype(np.float32, copy=False)),
                torch.from_numpy(data["times"].astype(np.float32, copy=False)),
                parameters,
                torch.from_numpy(data["native_activation_times_ms"].astype(np.float32, copy=False))
                if "native_activation_times_ms" in data else None,
            )
        if result.voltage.shape != (len(result.times), len(result.points)):
            raise ValueError(f"invalid voltage shape in {path}: {tuple(result.voltage.shape)}")
        if result.points.ndim != 2 or result.points.shape[1] != 3:
            raise ValueError(f"invalid point shape in {path}")
        if result.tetrahedra.ndim != 2 or result.tetrahedra.shape[1] != 4 or not result.tetrahedra.numel():
            raise ValueError(f"invalid tetrahedra in {path}")
        if result.tetrahedra.min() < 0 or result.tetrahedra.max() >= len(result.points):
            raise ValueError(f"tetrahedral node indices out of bounds in {path}")
        if not all(torch.isfinite(value).all() for value in (result.points, result.voltage, result.times)):
            raise ValueError(f"nonfinite trajectory values in {path}")
        if len(result.times) < 2 or not (result.times[1:] > result.times[:-1]).all():
            raise ValueError(f"trajectory times must be strictly increasing in {path}")
        if result.activation_times is not None:
            if result.activation_times.shape != (len(result.points),) or not torch.isfinite(result.activation_times).all():
                raise ValueError(f"invalid native activation times in {path}")
        return result
