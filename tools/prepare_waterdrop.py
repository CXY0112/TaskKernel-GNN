"""Validate the compact official WaterDrop download and freeze few-shot splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from validate_tfrecord import inspect

SEED = 20_260_917
OFFICIAL_BASE = "https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/WaterDrop"


def main() -> None:
    base = Path("data/external/deepmind/waterdrop")
    train = inspect(base / "train_128.tfrecord")
    valid = inspect(base / "valid.tfrecord")
    test = inspect(base / "test.tfrecord")
    expected = {
        "train_128.tfrecord": (128, 582_365_383, "5eb60f3d1727c81551570d77ab26b847"),
        "valid.tfrecord": (30, 137_190_408, "b387c27a4848ba7bddc4d0e31c5431eb"),
        "test.tfrecord": (30, 136_885_800, "f3adcb63381dfa08e4b61f1eafdab43e"),
    }
    for item in (train, valid, test):
        actual = (item["records"], item["bytes"], item["md5"])
        if actual != expected[item["file"]]:
            raise ValueError(f"verification failed for {item['file']}: {actual}")

    order = np.random.default_rng(SEED).permutation(128).tolist()
    splits = {
        "dataset": "DeepMind Learning to Simulate / WaterDrop",
        "seed": SEED,
        "rules": {
            "fewshot_subsets_are_nested": True,
            "validation_is_official_complete_split": True,
            "test_is_official_complete_split": True,
        },
        "train_indices": {
            f"fewshot_{size}": order[:size] for size in (4, 8, 16, 32, 64, 128)
        },
    }
    (base / "splits.json").write_text(
        json.dumps(splits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "dataset": "DeepMind Learning to Simulate / WaterDrop",
        "local_selection": "first 128 complete records from official training TFRecord",
        "official_full_train": {
            "url": f"{OFFICIAL_BASE}/train.tfrecord",
            "bytes": 4_541_246_980,
            "etag_md5": "80ce396abd6628336246b525577ec8b2",
            "records": 1000,
        },
        "local_files": [train, valid, test],
        "metadata": {
            "url": f"{OFFICIAL_BASE}/metadata.json",
            "bytes": 361,
            "md5": "f23537048cb1540e97086bf23c431c3f",
        },
        "total_local_bytes": sum(int(item["bytes"]) for item in (train, valid, test)) + 361,
        "split_file": "splits.json",
    }
    (base / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
