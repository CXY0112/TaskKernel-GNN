from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import load_npz


@dataclass(frozen=True)
class ParticleSystem:
    position: np.ndarray
    velocity: np.ndarray
    mass: np.ndarray
    edges: np.ndarray
    rest_length: np.ndarray
    spring_k: np.ndarray
    dashpot_gamma: np.ndarray
    gravity_g: float


def save_particle_system(path: str | Path, system: ParticleSystem) -> None:
    np.savez_compressed(Path(path), **system.__dict__)


def load_particle_system(path: str | Path) -> ParticleSystem:
    data = load_npz(path)
    return ParticleSystem(
        data["position"], data["velocity"], data["mass"], data["edges"], data["rest_length"],
        data["spring_k"], data["dashpot_gamma"], float(data["gravity_g"]),
    )

