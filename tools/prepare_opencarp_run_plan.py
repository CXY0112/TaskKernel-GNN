"""Freeze the external openCARP parameter scan without executing simulations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 20_260_917
ROOT = Path("data/external/opencarp/generated")


def parameters(rng: np.random.Generator) -> dict[str, float]:
    return {
        "g_il": float(rng.uniform(0.1445, 0.1955)),
        "g_it": float(rng.uniform(0.01615, 0.02185)),
        "g_in": float(rng.uniform(0.01615, 0.02185)),
        "stimulus_strength": float(rng.uniform(30.3535, 41.0665)),
        "stimulus_duration_ms": float(rng.uniform(1.7, 2.3)),
    }


def entry(name: str, split: str, index: int, resolution_mm: float, duration_ms: float,
          values: dict[str, float]) -> dict[str, object]:
    return {
        "name": name,
        "split": split,
        "index": index,
        "resolution_mm": resolution_mm,
        "dx_um": int(round(1000 * resolution_mm)),
        "dt_us": 20,
        "duration_ms": duration_ms,
        "output_interval_ms": 1.0,
        "parameters": values,
        "output_npz": f"{split}/{name}.npz",
    }


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    runs: list[dict[str, object]] = []
    split_files: dict[str, list[str]] = {}
    for split, count in (("train", 128), ("validation", 16), ("id_test", 30)):
        files = []
        for index in range(count):
            name = f"{split}_{index:04d}"
            item = entry(name, split, index, 0.5, 150.0, parameters(rng))
            runs.append(item); files.append(item["output_npz"])
        split_files[split] = files

    mesh_files = []
    id_parameters = [item["parameters"] for item in runs if item["split"] == "id_test"]
    for index, values in enumerate(id_parameters):
        name = f"mesh_ood_{index:04d}"
        item = entry(name, "mesh_ood", index, 0.2, 70.0, dict(values))
        runs.append(item); mesh_files.append(item["output_npz"])
    split_files["mesh_ood"] = mesh_files

    canonical = {"g_il": 0.17, "g_it": 0.019, "g_in": 0.019,
                 "stimulus_strength": 35.71, "stimulus_duration_ms": 2.0}
    reference = entry("reference_0000", "reference", 0, 0.1, 50.0, canonical)
    reference["save_full_voltage"] = True
    runs.append(reference); split_files["reference"] = [reference["output_npz"]]

    order = np.random.default_rng(SEED).permutation(128).tolist()
    for size in (4, 8, 16, 32, 64, 128):
        split_files[f"fewshot_{size}"] = [split_files["train"][index] for index in order[:size]]
    manifest = {
        "status": "planned_not_generated",
        "seed": SEED,
        "official_input": "../experiments/examples/02_EP_tissue/03G_study_resolution",
        "runs": runs,
    }
    (ROOT / "run_plan.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (ROOT / "splits.json").write_text(
        json.dumps({"status": "planned_not_generated", "seed": SEED, "splits": split_files}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"planned {len(runs)} openCARP runs in {ROOT}")


if __name__ == "__main__":
    main()
