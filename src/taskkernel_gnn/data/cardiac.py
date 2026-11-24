from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import load_npz


@dataclass(frozen=True)
class CardiacMesh:
    points: np.ndarray
    tetrahedra: np.ndarray
    boundary_triangles: np.ndarray
    fibers: np.ndarray


def load_cardiac_mesh(path: str | Path) -> CardiacMesh:
    data = load_npz(path)
    return CardiacMesh(data["points"], data["tetrahedra"], data["boundary_triangles"], data["fibers"])

