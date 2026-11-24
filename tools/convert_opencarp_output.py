"""Convert openCARP PTS/ELEM/IGB output into the frozen NPZ adapter format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_points(path: Path) -> np.ndarray:
    count = int(path.read_text(encoding="utf-8", errors="ignore").splitlines()[0])
    values = np.loadtxt(path, skiprows=1, max_rows=count, usecols=(0, 1, 2), dtype=np.float32)
    if len(values) != count:
        raise ValueError(f"PTS count mismatch in {path}")
    return values


def read_tetrahedra(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    count = int(lines[0])
    tetrahedra = []
    for line in lines[1:]:
        fields = line.split()
        if fields and fields[0].lower() in {"tt", "tet"}:
            tetrahedra.append([int(value) for value in fields[1:5]])
    result = np.asarray(tetrahedra, dtype=np.int64)
    if len(result) != count:
        raise ValueError(f"only tetrahedral meshes are supported: {len(result)} != {count}")
    return result


def read_igb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        header_raw = stream.read(1024)
    header_text = header_raw.decode("latin1", errors="ignore").replace("\x00", " ")
    fields: dict[str, str] = {}
    for token in header_text.split():
        if ":" in token:
            key, value = token.split(":", 1)
            fields[key.lower()] = value.strip()
    nodes = int(fields["x"]) * int(fields.get("y", "1")) * int(fields.get("z", "1"))
    frames = int(fields.get("t", "1"))
    if nodes < 1 or frames < 1:
        raise ValueError(f"invalid IGB dimensions in {path}")
    data_type = fields.get("type", "float").lower()
    if data_type not in {"float", "float32"}:
        raise ValueError(f"unsupported IGB type: {data_type}")
    system = fields.get("systeme", fields.get("system", "little_endian")).lower()
    dtype = np.dtype(">f4" if "big" in system else "<f4")
    values = np.fromfile(path, dtype=dtype, offset=1024, count=nodes * frames).astype(np.float32)
    if values.size != nodes * frames:
        raise ValueError(f"IGB payload size mismatch in {path}")
    increment = float(fields.get("inc_t", "1"))
    origin = float(fields.get("org_t", "0"))
    if not np.isfinite(values).all() or not np.isfinite([increment, origin]).all() or increment <= 0:
        raise ValueError(f"invalid IGB values or time interval in {path}")
    return values.reshape(frames, nodes), origin + increment * np.arange(frames, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--elements", type=Path, required=True)
    parser.add_argument("--voltage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameters-json", default="{}")
    parser.add_argument("--coordinate-unit", choices=("um", "mm"), default="um")
    args = parser.parse_args()
    points = read_points(args.points)
    tetrahedra = read_tetrahedra(args.elements)
    voltage, times = read_igb(args.voltage)
    if voltage.shape[1] != len(points):
        raise ValueError("voltage node count does not match mesh")
    parameters = json.loads(args.parameters_json)
    parameters.update({"coordinate_unit": args.coordinate_unit, "time_unit": "ms", "voltage_unit": "mV"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, points=points, tetrahedra=tetrahedra, voltage=voltage, times=times,
        parameters_json=np.asarray(json.dumps(parameters, sort_keys=True)),
    )
    print(f"wrote {args.output}: voltage={voltage.shape}, tetrahedra={tetrahedra.shape}")


if __name__ == "__main__":
    main()
