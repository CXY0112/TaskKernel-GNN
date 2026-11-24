"""Train and evaluate the E2 TaskKernel-GNN seeds and render paired visuals."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import torch
import yaml
from experiments.paper.train_opencarp_prepared import CONFIG, assert_ready, train
from experiments.paper.evaluate_opencarp_prepared import evaluate

OUTPUT = ROOT / "results/SOTA_Baseline/external/opencarp/prepared_v1/taskkernel"
STATUS = OUTPUT / "RUN_STATUS.json"
state = {"pid": os.getpid(), "model": "taskkernel", "seeds": [0,42,37], "completed_seeds": []}

def status(**values):
    state.update(values)
    state["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assert_ready()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for seed in state["seeds"]:
        destination = OUTPUT / f"seed_{seed}"
        if destination.exists():
            raise FileExistsError(f"preserving existing run: {destination}")
    original_step = torch.optim.AdamW.step
    def observed_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        state["optimizer_updates_current_seed"] += 1
        count = state["optimizer_updates_current_seed"]
        if count in (1,10) or count % 25 == 0:
            status()
        return result
    torch.optim.AdamW.step = observed_step
    try:
        for seed in state["seeds"]:
            destination = OUTPUT / f"seed_{seed}"
            status(phase="training", seed=seed, optimizer_updates_current_seed=0,
                   started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
            train("taskkernel", seed, config, destination)
            status(phase="evaluation")
            evaluate(destination / "best.pt", destination / "evaluation")
            status(phase="visualization")
            for suite in config["evaluation"]["suites"]:
                subprocess.run([sys.executable, "-X", "utf8",
                    str(ROOT / "experiments/paper/plot_opencarp_prepared.py"),
                    str(destination / f"evaluation/qualitative_{suite}.npz"),
                    "--output", str(destination / f"evaluation/visuals_{suite}"),
                    *(["--gif"] if seed == 0 else [])], cwd=ROOT, check=True)
            state["completed_seeds"].append(seed)
            status()
        status(phase="completed")
    finally:
        torch.optim.AdamW.step = original_step

if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if OUTPUT.exists():
            status(phase="failed", error=str(error), traceback=traceback.format_exc())
        raise
