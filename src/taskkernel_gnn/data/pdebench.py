"""Lazy reader for the public PDEBench 2D diffusion-reaction dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class PDEBenchSample:
    """One trajectory and the coordinates stored with it."""

    key: str
    data: np.ndarray
    grid: dict[str, np.ndarray]


def load_pdebench_splits(path: str | Path) -> dict[str, Any]:
    """Load the frozen train/validation/test index manifest."""

    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


class PDEBenchDiffusionReaction:
    """Read trajectories on demand without loading the 12.3 GiB file into RAM.

    Parameters
    ----------
    path:
        Path to ``2D_diff-react_NA_NA.h5``.
    indices:
        Optional sample indices into the lexicographically sorted top-level keys.
    spatial_stride:
        Positive spatial subsampling stride. ``1`` keeps the native 128x128 grid;
        ``2`` returns the frozen 64x64 training view used for Mesh-OOD.
    time_indices:
        Optional integer indices or slice along the trajectory time axis.
    """

    def __init__(
        self,
        path: str | Path,
        indices: Sequence[int] | None = None,
        *,
        spatial_stride: int = 1,
        time_indices: slice | Sequence[int] | None = None,
    ) -> None:
        self.path = Path(path)
        self._file = None
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if spatial_stride < 1:
            raise ValueError("spatial_stride must be >= 1")
        self.spatial_stride = int(spatial_stride)
        self.time_indices = slice(None) if time_indices is None else time_indices
        with self._open_new() as archive:
            self._all_keys = tuple(sorted(archive.keys()))
        selected = range(len(self._all_keys)) if indices is None else indices
        self.indices = tuple(int(index) for index in selected)
        invalid = [index for index in self.indices if not 0 <= index < len(self._all_keys)]
        if invalid:
            raise IndexError(f"sample indices out of range: {invalid[:5]}")

    @classmethod
    def from_split(
        cls,
        path: str | Path,
        split_path: str | Path,
        split: str,
        **kwargs: Any,
    ) -> "PDEBenchDiffusionReaction":
        manifest = load_pdebench_splits(split_path)
        try:
            indices = manifest["splits"][split]
        except KeyError as error:
            available = ", ".join(sorted(manifest.get("splits", {})))
            raise KeyError(f"unknown split {split!r}; available: {available}") from error
        return cls(path, indices, **kwargs)

    def _open_new(self):
        try:
            import h5py
        except ImportError as error:
            raise ImportError("PDEBench HDF5 loading requires h5py>=3.11") from error
        return h5py.File(self.path, "r")

    @property
    def file(self):
        if self._file is None:
            self._file = self._open_new()
        return self._file

    @property
    def sample_keys(self) -> tuple[str, ...]:
        return tuple(self._all_keys[index] for index in self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> PDEBenchSample:
        key = self._all_keys[self.indices[item]]
        group = self.file[key]
        data = np.asarray(group["data"][self.time_indices])[
            :, :: self.spatial_stride, :: self.spatial_stride, :
        ]
        grid: dict[str, np.ndarray] = {}
        if "grid" in group:
            grid_group = group["grid"]
            for name, value in grid_group.items():
                values = np.asarray(value)
                if name == "t":
                    values = values[self.time_indices]
                elif name in {"x", "y"}:
                    values = values[:: self.spatial_stride]
                grid[name] = values
        return PDEBenchSample(key=key, data=data, grid=grid)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "PDEBenchDiffusionReaction":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __del__(self) -> None:
        self.close()
