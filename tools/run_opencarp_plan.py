"""Execute the frozen openCARP plan and convert every run to the project NPZ schema.

The default mode only validates prerequisites and prints the selected runs.
Actual simulations require ``--execute``.  This script is intended for a Linux,
WSL, or container environment with openCARP, MPI, and carputils installed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "data" / "external" / "opencarp" / "generated"
OFFICIAL = ROOT / "data" / "external" / "opencarp" / "experiments" / "examples" / "02_EP_tissue" / "03G_study_resolution"
sys.path.insert(0, str(ROOT / "tools"))
from convert_opencarp_output import read_igb, read_points, read_tetrahedra


def replace_parameter(text: str, key: str, value: float) -> str:
    pattern = rf"(?m)^\s*{re.escape(key)}\s*=.*$"
    replacement = f"{key} = {value:.12g}"
    updated, count = re.subn(pattern, replacement, text)
    if count != 1:
        raise ValueError(f"expected one {key} entry in nversion.par, found {count}")
    return updated


def selected_runs(plan: dict, splits: set[str], limit: int | None) -> list[dict]:
    values = [item for item in plan["runs"] if not splits or item["split"] in splits]
    return values if limit is None else values[:limit]


def find_one(directory: Path, patterns: tuple[str, ...]) -> Path:
    candidates = []
    for pattern in patterns:
        candidates.extend(directory.rglob(pattern))
    candidates = sorted(set(path for path in candidates if path.is_file()))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one of {patterns} under {directory}, found {len(candidates)}")
    return candidates[0]


def execute_run(item: dict, python: str, processes: int = 1, workspace_root: Path | None = None) -> Path:
    destination = GENERATED / item["output_npz"]
    if destination.is_file():
        with np.load(destination, allow_pickle=False) as existing:
            if existing["voltage"].shape != (len(existing["times"]), len(existing["points"])):
                raise ValueError(f"invalid existing export: {destination}")
            if not np.isfinite(existing["voltage"]).all():
                raise ValueError(f"nonfinite existing export: {destination}")
            stored = json.loads(str(existing["parameters_json"]))
            if any(stored.get(key) != value for key, value in item["parameters"].items()):
                raise ValueError(f"existing export parameters differ: {destination}")
        print(f"Reusing validated export: {destination}", flush=True)
        return destination
    run_root = (workspace_root or GENERATED / "_runs").resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    workspace = run_root / item["name"]
    if not workspace.resolve().is_relative_to(run_root):
        raise ValueError(f"unsafe openCARP workspace: {workspace}")
    if workspace.exists():
        raise FileExistsError(f"run workspace already exists; preserve it and inspect before rerunning: {workspace}")
    shutil.copytree(OFFICIAL, workspace)
    parameters = item["parameters"]
    par_path = workspace / "nversion.par"
    par = par_path.read_text(encoding="utf-8")
    for key, value in (
        ("gregion[0].g_il", parameters["g_il"]),
        ("gregion[0].g_it", parameters["g_it"]),
        ("gregion[0].g_in", parameters["g_in"]),
        ("stim[0].pulse.strength", parameters["stimulus_strength"]),
        ("stim[0].ptcl.duration", parameters["stimulus_duration_ms"]),
        ("spacedt", item["output_interval_ms"]),
    ):
        par = replace_parameter(par, key, float(value))
    for axis in range(3):
        for corner in ("p0", "p1"):
            parameter = f"stimulus_{corner}_{axis}_um"
            if parameter in parameters:
                par = replace_parameter(par, f"stim[0].elec.{corner}[{axis}]", parameters[parameter])
    par_path.write_text(par, encoding="utf-8")

    output_dir = workspace / "output"
    command = [
        python, "run.py", "--dx", str(item["dx_um"]), "--dt", str(item["dt_us"]),
        "--tend", str(item["duration_ms"]), "--ID", str(output_dir),
        "--np", str(processes),
    ]
    subprocess.run(command, cwd=workspace, check=True)
    points = find_one(workspace, ("*.pts",))
    elements = find_one(workspace, ("*.elem",))
    voltage = find_one(output_dir, ("vm.igb", "*vm*.igb"))
    points_array = read_points(points)
    tetrahedra = read_tetrahedra(elements)
    voltage_array, times = read_igb(voltage)
    if voltage_array.shape[1] != len(points_array):
        raise ValueError(f"node mismatch for {item['name']}")
    destination = GENERATED / item["output_npz"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    exported_parameters = dict(parameters)
    exported_parameters.update({"coordinate_unit": "um", "time_unit": "ms", "voltage_unit": "mV"})
    temporary = destination.with_name(destination.stem + ".partial.npz")
    np.savez_compressed(
        temporary, points=points_array, tetrahedra=tetrahedra, voltage=voltage_array,
        times=times, parameters_json=np.asarray(json.dumps(exported_parameters, sort_keys=True)),
        native_activation_times_ms=np.loadtxt(output_dir / "init_acts_vm_act-thresh.dat").reshape(-1),
    )
    temporary.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default="train,validation,id_test,mesh_ood,reference")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--np", type=int, default=1, help="MPI process count for each reference solve")
    parser.add_argument("--budget", type=int, choices=(4, 8, 16, 32, 64, 128),
                        help="select the frozen fewshot subset when train is included")
    parser.add_argument("--canonical-smoke", action="store_true",
                        help="solve a separate official 0.5 mm case; do not alter frozen splits")
    parser.add_argument("--smoke-output-interval-ms", type=float, default=1.0,
                        help="output interval of the independent smoke case only")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan", type=Path, default=GENERATED / "run_plan.json")
    parser.add_argument("--workspace-root", type=Path,
                        help="scratch directory for solver outputs; NPZ exports remain in the project")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    splits = {value.strip() for value in args.splits.split(",") if value.strip()}
    if args.np < 1:
        parser.error("--np must be positive")
    runs = selected_runs(plan, splits, None)
    if args.budget is not None:
        manifest = json.loads((GENERATED / "splits.json").read_text(encoding="utf-8"))
        selected = set(manifest["splits"][f"fewshot_{args.budget}"])
        runs = [item for item in runs if item["split"] != "train" or item["output_npz"] in selected]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        runs = runs[:args.limit]
    if args.canonical_smoke:
        if args.smoke_output_interval_ms <= 0:
            parser.error("smoke output interval must be positive")
        smoke_name = "canonical_smoke"
        if args.smoke_output_interval_ms != 1.0:
            smoke_name += "_output_" + str(args.smoke_output_interval_ms).replace(".", "p")
        runs = [{
            "name": smoke_name, "split": "preflight", "dx_um": 500, "dt_us": 20,
            "duration_ms": 150.0, "output_interval_ms": args.smoke_output_interval_ms,
            "parameters": {"g_il": 0.17, "g_it": 0.019, "g_in": 0.019,
                           "stimulus_strength": 35.71, "stimulus_duration_ms": 2.0},
            "output_npz": f"_preflight/{smoke_name}.npz",
        }]
    prerequisites = {
        "openCARP": shutil.which("openCARP"),
        "mpiexec": shutil.which("mpiexec"),
        "carputils": False,
    }
    try:
        __import__("carputils")
        prerequisites["carputils"] = True
    except ImportError:
        pass
    print(json.dumps({
        "execute": args.execute, "selected_runs": len(runs), "splits": sorted(splits),
        "budget": args.budget, "mpi_processes": args.np,
        "prerequisites": prerequisites,
    }, indent=2))
    if not args.execute:
        return
    missing = [name for name, value in prerequisites.items() if not value]
    if missing:
        raise RuntimeError(f"cannot execute openCARP plan; missing: {', '.join(missing)}")
    for index, item in enumerate(runs, 1):
        destination = execute_run(item, args.python, args.np, args.workspace_root)
        print(f"[{index}/{len(runs)}] {destination}", flush=True)


if __name__ == "__main__":
    main()
