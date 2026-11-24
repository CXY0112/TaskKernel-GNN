"""Verify PDEBench 2D diffusion-reaction and freeze reproducible splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import h5py
import numpy as np

EXPECTED_BYTES = 13_243_560_688
EXPECTED_MD5 = "b8d0b86064193195ddc30c33be5dc949"
SEED = 20_260_917


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def dataset_schema(group: h5py.Group) -> dict[str, dict[str, object]]:
    schema: dict[str, dict[str, object]] = {}

    def visit(name: str, value: h5py.Dataset | h5py.Group) -> None:
        if isinstance(value, h5py.Dataset):
            schema[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}

    group.visititems(visit)
    return schema


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    output_dir = (args.output_dir or dataset.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    byte_size = dataset.stat().st_size
    if byte_size != EXPECTED_BYTES:
        raise ValueError(f"size mismatch: {byte_size} != {EXPECTED_BYTES}")
    checksum = file_md5(dataset)
    if checksum != EXPECTED_MD5:
        raise ValueError(f"MD5 mismatch: {checksum} != {EXPECTED_MD5}")

    with h5py.File(dataset, "r") as archive:
        keys = sorted(archive.keys())
        if len(keys) != 1_000:
            raise ValueError(f"expected 1000 samples, found {len(keys)}")
        first_schema = dataset_schema(archive[keys[0]])
        if "data" not in first_schema:
            raise ValueError("first sample does not contain a data dataset")
        shapes = {tuple(archive[key]["data"].shape) for key in keys}
        dtypes = {str(archive[key]["data"].dtype) for key in keys}
        if len(shapes) != 1 or len(dtypes) != 1:
            raise ValueError(f"inconsistent data arrays: shapes={shapes}, dtypes={dtypes}")
        first_data = archive[keys[0]]["data"]
        probe = np.stack((np.asarray(first_data[0]), np.asarray(first_data[-1])))
        if not np.isfinite(probe).all():
            raise ValueError("non-finite values found in first trajectory probe")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(keys)).tolist()
    train_pool = order[:700]
    validation = order[700:800]
    test = order[800:]
    splits = {
        "fewshot_4": train_pool[:4],
        "fewshot_8": train_pool[:8],
        "fewshot_16": train_pool[:16],
        "fewshot_32": train_pool[:32],
        "fewshot_64": train_pool[:64],
        "fewshot_128": train_pool[:128],
        "train_pool": train_pool,
        "validation": validation,
        "test": test,
    }
    if set(train_pool) & set(validation) or set(train_pool) & set(test) or set(validation) & set(test):
        raise AssertionError("train/validation/test overlap")

    split_manifest = {
        "dataset": "PDEBench 2D Diffusion-Reaction",
        "source_file": dataset.name,
        "source_md5": checksum,
        "seed": SEED,
        "sample_order": "indices into lexicographically sorted HDF5 top-level keys",
        "rules": {
            "fewshot_subsets_are_nested": True,
            "test_set_is_frozen": True,
            "test_set_tuning": False,
        },
        "splits": splits,
    }
    (output_dir / "splits.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "dataset": "PDEBench 2D Diffusion-Reaction",
        "official_file": dataset.name,
        "official_url": "https://darus.uni-stuttgart.de/api/access/datafile/133017",
        "doi": "10.18419/DARUS-2986",
        "license": "CC BY 4.0",
        "prepared_on": date.today().isoformat(),
        "bytes": byte_size,
        "md5": checksum,
        "sample_count": len(keys),
        "first_sample_key": keys[0],
        "last_sample_key": keys[-1],
        "sample_schema": first_schema,
        "data_shapes": [list(shape) for shape in sorted(shapes)],
        "data_dtypes": sorted(dtypes),
        "first_sample_probe": {
            "finite": True,
            "minimum": float(probe.min()),
            "maximum": float(probe.max()),
        },
        "split_file": "splits.json",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
