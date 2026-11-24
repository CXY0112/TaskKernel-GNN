"""Read prepared ISO17 test arrays and select molecule groups."""

from __future__ import annotations

import numpy as np
from iso17_data import ATOMS, PROJECT, Frames


CACHE = PROJECT / "data/external/iso17/evaluation_cache"


def cached_frames(name: str, count: int, block_size: int, offset: int) -> tuple[Frames, np.ndarray]:
    folder = CACHE / name
    if not (folder / "complete.txt").exists():
        raise FileNotFoundError(folder / "complete.txt")
    position = np.load(folder / "position.npy", mmap_mode="r")
    force = np.load(folder / "force.npy", mmap_mode="r")
    numbers = np.load(folder / "numbers.npy", mmap_mode="r")
    energy = np.load(folder / "energy.npy", mmap_mode="r")
    if position.shape != (count, ATOMS, 3) or force.shape != position.shape or energy.shape != (count,):
        raise ValueError(f"unexpected cached arrays for {name}")
    molecule = np.arange(count, dtype=np.int64) // block_size + offset
    return Frames(position, force, numbers, molecule, list(range(1, count + 1))), energy


def subset(frames: Frames, energies: np.ndarray, allowed: set[int]) -> tuple[Frames, np.ndarray]:
    chosen = np.isin(frames.molecules, sorted(allowed))
    return (Frames(frames.position[chosen], frames.force[chosen], frames.numbers[chosen],
                   frames.molecules[chosen], list(range(int(chosen.sum())))), energies[chosen])
