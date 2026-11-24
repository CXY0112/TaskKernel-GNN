"""TensorFlow-free reader for DeepMind Learning-to-Simulate TFRecords."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .graph import GraphSample, radius_graph


def _varint(data: memoryview, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
        if shift > 70:
            raise ValueError("invalid protobuf varint")


def _fields(payload: bytes) -> Iterator[tuple[int, int, int | bytes]]:
    data = memoryview(payload)
    offset = 0
    while offset < len(data):
        tag, offset = _varint(data, offset)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, offset = _varint(data, offset)
        elif wire == 1:
            value = bytes(data[offset:offset + 8]); offset += 8
        elif wire == 2:
            length, offset = _varint(data, offset)
            value = bytes(data[offset:offset + length]); offset += length
        elif wire == 5:
            value = bytes(data[offset:offset + 4]); offset += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield number, wire, value


def _message_values(payload: bytes, field_number: int) -> list[bytes]:
    return [value for number, wire, value in _fields(payload)
            if number == field_number and wire == 2 and isinstance(value, bytes)]


def _map(payload: bytes) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for entry in _message_values(payload, 1):
        key_values = _message_values(entry, 1)
        value_values = _message_values(entry, 2)
        if key_values and value_values:
            result[key_values[0].decode("utf-8")] = value_values[0]
    return result


def _feature_bytes(feature: bytes) -> list[bytes]:
    bytes_lists = _message_values(feature, 1)
    return _message_values(bytes_lists[0], 1) if bytes_lists else []


def _decode_sequence_example(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    top = {number: value for number, wire, value in _fields(payload) if wire == 2}
    context = _map(top[1])
    feature_lists = _map(top[2])
    particle_values = _feature_bytes(context["particle_type"])
    if not particle_values:
        raise ValueError("particle_type is missing from SequenceExample")
    particle_type = np.frombuffer(particle_values[0], dtype=np.int64).copy()

    feature_list = feature_lists["position"]
    frames = []
    for feature in _message_values(feature_list, 1):
        values = _feature_bytes(feature)
        if values:
            frames.append(np.frombuffer(values[0], dtype=np.float32).reshape(-1, 2).copy())
    positions = np.stack(frames)
    if positions.shape[1] != len(particle_type):
        raise ValueError("position and particle_type counts do not match")
    return positions, particle_type


def _scan(path: Path) -> tuple[tuple[int, int], ...]:
    records: list[tuple[int, int]] = []
    size = path.stat().st_size
    with path.open("rb") as stream:
        while stream.tell() < size:
            header_offset = stream.tell()
            header = stream.read(12)
            if len(header) != 12:
                raise ValueError(f"truncated TFRecord header at {header_offset}")
            length = struct.unpack("<Q", header[:8])[0]
            payload_offset = stream.tell()
            stream.seek(length + 4, 1)
            if stream.tell() > size:
                raise ValueError(f"truncated TFRecord payload at {payload_offset}")
            records.append((payload_offset, length))
    return tuple(records)


@dataclass(frozen=True)
class WaterDropTrajectory:
    positions: torch.Tensor
    particle_type: torch.Tensor
    key: int


class WaterDropDataset:
    def __init__(self, path: str | Path, indices: list[int] | tuple[int, ...] | None = None) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.records = _scan(self.path)
        self.indices = tuple(range(len(self.records))) if indices is None else tuple(indices)
        if any(index < 0 or index >= len(self.records) for index in self.indices):
            raise IndexError("WaterDrop record index out of range")

    @classmethod
    def from_split(cls, directory: str | Path, split: str) -> "WaterDropDataset":
        directory = Path(directory)
        if split.startswith("fewshot_"):
            values = json.loads((directory / "splits.json").read_text(encoding="utf-8"))
            return cls(directory / "train_128.tfrecord", values["train_indices"][split])
        filename = {"validation": "valid.tfrecord", "test": "test.tfrecord"}.get(split)
        if filename is None:
            raise KeyError(f"unknown WaterDrop split: {split}")
        return cls(directory / filename)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> WaterDropTrajectory:
        key = self.indices[item]
        offset, length = self.records[key]
        with self.path.open("rb") as stream:
            stream.seek(offset)
            payload = stream.read(length)
        positions, particle_type = _decode_sequence_example(payload)
        return WaterDropTrajectory(torch.from_numpy(positions), torch.from_numpy(particle_type), key)


def waterdrop_graph(sample: WaterDropTrajectory, time_index: int, metadata: dict,
                    *, max_neighbors: int | None = None) -> GraphSample:
    graph = radius_graph(sample.positions[time_index], float(metadata["default_connectivity_radius"]),
                         max_neighbors=max_neighbors)
    graph.context.update({
        "particle_type": sample.particle_type,
        "position": sample.positions[time_index],
        "velocity": sample.positions[time_index] - sample.positions[max(time_index - 1, 0)],
        "mass": torch.ones(len(sample.particle_type)),
        "dt": float(metadata["dt"]),
        "bounds": torch.as_tensor(metadata["bounds"], device=sample.positions.device,
                                   dtype=sample.positions.dtype),
        # WaterDrop particles represent fluid parcels rather than astronomical
        # point masses.  This small effective coefficient keeps the optional
        # inverse-square candidate in the scale of per-step accelerations.
        "gravity_g": 1.0e-6,
    })
    return graph
