"""ISO17 data loading, frozen splits, and molecular graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from itertools import combinations

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from ase.data import atomic_masses, covalent_radii  # noqa: E402
from ase.db import connect  # noqa: E402
RAW = PROJECT / "data" / "external" / "iso17" / "raw"
ATOMS = 19
ELEMENT_INDEX = {1: 0, 6: 1, 8: 2}
PAIR_INDEX = {(1, 1): 0, (1, 6): 1, (1, 8): 2, (6, 6): 3, (6, 8): 4, (8, 8): 5}
@dataclass
class Frames:
    position: np.ndarray
    force: np.ndarray
    numbers: np.ndarray
    molecules: np.ndarray
    ids: list[int]

    def __len__(self):
        return len(self.ids)


@dataclass
class Topology:
    numbers: np.ndarray
    pair_type: np.ndarray
    bond_edges: np.ndarray
    nonbond_edges: np.ndarray
    triplets: np.ndarray
    triplet_edges: np.ndarray
    center_type: np.ndarray
    bond_pairs: np.ndarray


def read_frames(path: Path, ids: list[int], block_size: int, offset: int = 0) -> Frames:
    with connect(path) as conn:
        rows = [conn.get(id=int(entry)) for entry in ids]
    position = np.stack([row.positions for row in rows]).astype(np.float32)
    force = np.stack([row.data["atomic_forces"] for row in rows]).astype(np.float32)
    numbers = np.stack([row.numbers for row in rows]).astype(np.int64)
    if position.shape != force.shape or position.shape[1:] != (ATOMS, 3):
        raise ValueError(f"invalid ISO17 shapes from {path}")
    if not np.isfinite(position).all() or not np.isfinite(force).all():
        raise ValueError(f"nonfinite ISO17 values from {path}")
    molecules = np.asarray([(int(entry) - 1) // block_size + offset for entry in ids], dtype=np.int64)
    return Frames(position, force, numbers, molecules, ids)


def edge_position(source: int, target: int) -> int:
    return source * (ATOMS - 1) + target - (target > source)


def make_topology(numbers: np.ndarray, position: np.ndarray, valence_cap: bool = False) -> Topology:
    source = np.repeat(np.arange(ATOMS), ATOMS - 1)
    target = np.concatenate([np.delete(np.arange(ATOMS), i) for i in range(ATOMS)])
    pair_type = np.array([PAIR_INDEX[tuple(sorted((numbers[i], numbers[j])))] for i, j in zip(source, target)], dtype=np.int64)
    distance = np.linalg.norm(position[:, None, :] - position[None, :, :], axis=-1)
    radii = covalent_radii[numbers]
    bonds = (distance < 1.25 * (radii[:, None] + radii[None, :])) & ~np.eye(ATOMS, dtype=bool)
    if valence_cap:
        # A distance cutoff can assign five neighbours to carbon. Remove the
        # weakest candidate incident to over-coordinated atoms, favouring an
        # edge that fixes both endpoints in one step.
        maximum = np.array([{1: 1, 6: 4, 8: 2}[int(z)] for z in numbers])
        while np.any(bonds.sum(axis=1) > maximum):
            excess = bonds.sum(axis=1) > maximum
            candidates = np.argwhere(np.triu(bonds, k=1) & (excess[:, None] | excess[None, :]))
            score = [
                (int(excess[i]) + int(excess[j]), distance[i, j] / (radii[i] + radii[j]))
                for i, j in candidates
            ]
            i, j = candidates[max(range(len(score)), key=score.__getitem__)]
            bonds[i, j] = bonds[j, i] = False
    # Exclude both directly bonded and one-bond-separated pairs from nonbonded forces.
    near = bonds | (bonds.astype(np.int16) @ bonds.astype(np.int16) > 0)
    triplets = []
    triplet_edges = []
    center_type = []
    for center in range(ATOMS):
        neighbors = np.flatnonzero(bonds[center]).tolist()
        for first, last in combinations(neighbors, 2):
            triplets.append((first, center, last))
            triplet_edges.append((
                edge_position(center, first), edge_position(center, last),
                edge_position(first, center), edge_position(last, center),
            ))
            center_type.append(ELEMENT_INDEX[int(numbers[center])])
    return Topology(
        numbers.copy(), pair_type, bonds[source, target].astype(np.float32),
        (~near[source, target]).astype(np.float32),
        np.asarray(triplets, dtype=np.int64).reshape(-1, 3),
        np.asarray(triplet_edges, dtype=np.int64).reshape(-1, 4),
        np.asarray(center_type, dtype=np.int64),
        np.asarray([(i, j) for i in range(ATOMS) for j in range(i + 1, ATOMS) if bonds[i, j]], dtype=np.int64).reshape(-1, 2),
    )


def topology_map(valence_cap: bool = False) -> dict[int, Topology]:
    mapping = {}
    with connect(RAW / "reference.db") as conn:
        for mol in range(101):
            row = conn.get(id=mol * 4000 + 1)
            mapping[mol] = make_topology(row.numbers, row.positions, valence_cap=valence_cap)
    with connect(RAW / "test_other.db") as conn:
        for mol in range(26):
            row = conn.get(id=mol * 5000 + 1)
            mapping[101 + mol] = make_topology(row.numbers, row.positions, valence_cap=valence_cap)
    return mapping


def protocol() -> dict:
    rng = np.random.default_rng(20260926)
    train_allowed = set(np.loadtxt(RAW / "train_ids.txt", dtype=np.int32).tolist())
    valid_allowed = set(np.loadtxt(RAW / "validation_ids.txt", dtype=np.int32).tolist())
    molecules = sorted(rng.choice(101, 16, replace=False).tolist())
    train_by_mol = {}
    valid_ids = []
    test_within_seen = []
    test_within_unseen = []
    for mol in range(101):
        block = range(mol * 4000 + 1, (mol + 1) * 4000 + 1)
        if mol in molecules:
            available = [entry for entry in block if entry in train_allowed]
            train_by_mol[mol] = rng.choice(available, 128, replace=False).tolist()
            valid = [entry for entry in block if entry in valid_allowed]
            valid_ids.extend(rng.choice(valid, 16, replace=False).tolist())
        # Every reference molecule has 1,000 held-out frames in test_within.
        test_block = np.arange(mol * 1000 + 1, (mol + 1) * 1000 + 1)
        sample = rng.choice(test_block, 64, replace=False).tolist()
        (test_within_seen if mol in molecules else test_within_unseen).extend(sample)
    other_force = []
    rollout_starts = []
    for mol in range(26):
        block = np.arange(mol * 5000 + 1, (mol + 1) * 5000 + 1)
        other_force.extend(rng.choice(block, 64, replace=False).tolist())
        rollout_starts.extend([mol * 5000 + 101, mol * 5000 + 2501])
    return {
        "protocol_seed": 20260926,
        "train_molecules": molecules,
        "train_ids_by_molecule": train_by_mol,
        "validation_ids": valid_ids,
        "test_within_seen_ids": test_within_seen,
        "test_within_unseen_ids": test_within_unseen,
        "test_other_force_ids": other_force,
        "rollout_starts": rollout_starts,
        "budgets_per_molecule": {"128": 8, "512": 32, "2048": 128},
    }


def graph(position: np.ndarray, molecules: np.ndarray, tops: dict[int, Topology]):
    batch = len(position)
    base_source = np.repeat(np.arange(ATOMS), ATOMS - 1)
    base_target = np.concatenate([np.delete(np.arange(ATOMS), i) for i in range(ATOMS)])
    offset = (np.arange(batch, dtype=np.int64) * ATOMS)[:, None]
    edge_source = (base_source[None, :] + offset).reshape(-1)
    edge_target = (base_target[None, :] + offset).reshape(-1)
    types, bonds, nonbonds = [], [], []
    triplets, angle_edges, centers = [], [], []
    for index, molecule in enumerate(molecules):
        top = tops[int(molecule)]
        types.append(top.pair_type)
        bonds.append(top.bond_edges)
        nonbonds.append(top.nonbond_edges)
        triplets.append(top.triplets + index * ATOMS)
        angle_edges.append(top.triplet_edges + index * ATOMS * (ATOMS - 1))
        centers.append(top.center_type)
    edge_index = torch.from_numpy(np.stack([edge_source, edge_target]))
    context = {
        "position": torch.from_numpy(position.reshape(-1, 3)),
        "atomic_numbers": torch.from_numpy(np.stack([tops[int(mol)].numbers for mol in molecules]).reshape(-1)),
        "pair_type": torch.from_numpy(np.concatenate(types)),
        "bond_edge_mask": torch.from_numpy(np.concatenate(bonds)),
        "nonbonded_edge_mask": torch.from_numpy(np.concatenate(nonbonds)),
        "angle_triplets": torch.from_numpy(np.concatenate(triplets)),
        "angle_edge_indices": torch.from_numpy(np.concatenate(angle_edges)),
        "angle_center_type": torch.from_numpy(np.concatenate(centers)),
        "molecule_index": torch.arange(batch).repeat_interleave(ATOMS),
    }
    return edge_index, context
