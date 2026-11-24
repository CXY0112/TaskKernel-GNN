"""Cache ISO17 official-test energy labels beside the existing force arrays."""

from __future__ import annotations

import numpy as np

from iso17_data import RAW, connect


def cache(name: str, count: int) -> None:
    folder = RAW.parent / "evaluation_cache" / name
    if not (folder / "complete.txt").exists():
        raise FileNotFoundError(f"force/position cache must already be complete: {folder}")
    target = folder / "energy.npy"
    if target.exists():
        values = np.load(target, mmap_mode="r")
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError(f"invalid existing energy cache: {target}")
        return
    values = np.lib.format.open_memmap(target, mode="w+", dtype="float64", shape=(count,))
    with connect(RAW / f"{name}.db") as conn:
        for index, row in enumerate(conn.select()):
            if index >= count or row.id != index + 1:
                raise ValueError(f"unexpected row count/order: {name}, {index}, {row.id}")
            values[index] = float(row.total_energy)
            if index % 25000 == 0:
                print(f"{name}: {index}/{count}", flush=True)
    if index + 1 != count:
        raise ValueError(f"unexpected final row count: {name}, {index + 1}")
    values.flush()
    if not np.isfinite(values).all():
        raise ValueError(f"nonfinite cached energy: {target}")


def main() -> None:
    cache("test_within", 101000)
    cache("test_other", 130000)


if __name__ == "__main__":
    main()
