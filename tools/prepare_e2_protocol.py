"""Version the main E2 protocol without overwriting the original 128-run pool."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/external/opencarp/generated"

def main():
    old = json.loads((DATA / "run_plan.json").read_text())
    splits = json.loads((DATA / "splits.json").read_text())["splits"]
    chosen = set(splits["fewshot_16"])
    runs = [r for r in old["runs"] if r["split"] != "train" or r["output_npz"] in chosen]
    rng = np.random.default_rng(20261009)
    stimuli = []
    for i in range(16):
        item = dict(next(r for r in runs if r["split"] == "id_test" and r["index"] == i))
        item.update(name=f"stimulus_ood_{i:04d}", split="stimulus_ood",
                    output_npz=f"stimulus_ood/stimulus_ood_{i:04d}.npz")
        item["parameters"] = dict(item["parameters"], stimulus_strength=float(rng.uniform(45, 55)),
                                  stimulus_duration_ms=float(rng.uniform(2.5, 3.0)))
        runs.append(item); stimuli.append(item["output_npz"])
    protocol = {"version": "e2_prepared_v1", "seeds": [0, 42, 37], "primary_budget": 16,
                "training_pool_cap": 128, "status": "planned", "runs": runs,
                "parent": "run_plan.json", "stimulus_ood_definition": "higher amplitude and longer pulse; same corner box"}
    (DATA / "run_plan_v1.json").write_text(json.dumps(protocol, indent=2) + "\n")
    new_splits = {key: value for key, value in splits.items() if key in ("validation", "id_test", "mesh_ood", "reference", "fewshot_4", "fewshot_8", "fewshot_16")}
    new_splits.update(train=splits["fewshot_16"], stimulus_ood=stimuli)
    destination = DATA / "protocol_v1"
    destination.mkdir(exist_ok=True)
    # Dataset manifest uses paths relative to this directory, with no duplicate data.
    (destination / "splits.json").write_text(json.dumps({"version": protocol["version"], "splits": {
        key: ["../" + name for name in values] for key, values in new_splits.items()}}, indent=2) + "\n")
    print(f"Versioned {len(runs)} reference solves; original pool manifest preserved")

if __name__ == "__main__":
    main()
