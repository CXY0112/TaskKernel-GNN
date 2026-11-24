"""Cache complete official ISO17 test arrays for repeated read-only scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from iso17_data import ATOMS, RAW, connect

OUT = RAW.parent / "evaluation_cache"


def cache(name: str, count: int):
    source = RAW / name
    target = OUT / source.stem
    target.mkdir(parents=True, exist_ok=True)
    complete = target / "complete.txt"
    if complete.exists():
        return
    xyz = np.lib.format.open_memmap(target / "position.npy", mode="w+", dtype="float32",
                                    shape=(count, ATOMS, 3))
    force = np.lib.format.open_memmap(target / "force.npy", mode="w+", dtype="float32",
                                      shape=(count, ATOMS, 3))
    numbers = np.lib.format.open_memmap(target / "numbers.npy", mode="w+", dtype="int64",
                                        shape=(count, ATOMS))
    with connect(source) as conn:
        for index, row in enumerate(conn.select()):
            if index >= count or row.id != index + 1:
                raise ValueError(f"unexpected frame count/order in {source}: {index}, {row.id}")
            xyz[index] = row.positions
            force[index] = row.data["atomic_forces"]
            numbers[index] = row.numbers
            if index % 25000 == 0:
                print(f"{name}: {index}/{count}", flush=True)
    if index + 1 != count:
        raise ValueError(f"unexpected frame count in {source}: {index + 1}/{count}")
    xyz.flush(); force.flush(); numbers.flush()
    if not np.isfinite(xyz).all() or not np.isfinite(force).all():
        raise ValueError(f"nonfinite cached values in {source}")
    complete.write_text(f"{name}: {count} official frames in ascending database row ID\n", encoding="utf-8")


def main():
    cache("test_within.db", 101000)
    cache("test_other.db", 130000)


if __name__ == "__main__":
    main()
