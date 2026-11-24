"""Audit real E2 labels and exercise models without optimizer updates."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from taskkernel_gnn.data.opencarp import OpenCARPDataset
from taskkernel_gnn.data.cardiac_fem import anisotropic_fem_graph, conditioned_history, diffusion_rate
from experiments.paper.opencarp_prepared import E2Predictor, initial_history
from experiments.paper.train_opencarp_prepared import CONFIG, OUTPUT, window_loss


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physics_checks(graph):
    ones = torch.ones(graph.nodes, 1)
    constant = float(diffusion_rate(ones, graph).abs().max())
    generator = torch.Generator().manual_seed(519)
    state = torch.randn(graph.nodes, 1, generator=generator)
    rate = diffusion_rate(state, graph).double()
    mass = graph.context["fem_mass"][:, None]
    conservation = float((mass * rate).sum().abs())
    energy = float((mass * state.double() * rate).sum())
    assert constant == 0 and conservation < .01 and energy <= 0
    assert abs(float(mass.sum()) - 420) < 1e-4
    history = ones.expand(-1, 8)
    before = conditioned_history(history, graph, 0.)
    after = conditioned_history(history, graph, 10.)
    assert before[:, -1].max() > 0 and after[:, -1].max() == 0
    assert torch.equal(before[:, :8], history)
    return {"constant_diffusion_rate": constant, "mass_conservation_residual": conservation,
            "mass_weighted_energy_rate": energy, "volume_mm3": float(mass.sum()),
            "forcing_switches_off": True, "no_future_voltage_input": True,
            "negative_edge_fraction": float((graph.weight < 0).float().mean()),
            "nodes": graph.nodes, "directed_edges": len(graph.source)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interfaces-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    provenance = json.loads((ROOT / "data/external/opencarp/manifest.json").read_text())
    official = ROOT / "data/external/opencarp/experiments/examples/02_EP_tissue/03G_study_resolution"
    official_integrity = {item["name"]: digest(official / item["name"]) == item["sha256"]
                          for item in provenance["files"]}
    assert all(official_integrity.values())
    data = ROOT / config["dataset"]["root"]
    plan_path = ROOT / config["dataset"]["protocol"]
    plan = json.loads(plan_path.read_text())
    records, missing, counts = [], [], {}
    generated = data.parent
    for item in plan["runs"]:
        path = generated / item["output_npz"]
        if not path.is_file():
            missing.append(item["output_npz"]); continue
        with np.load(path, allow_pickle=False) as values:
            voltage, times, points = values["voltage"], values["times"], values["points"]
            assert voltage.shape == (len(times), len(points))
            assert np.isfinite(voltage).all() and np.all(np.diff(times) > 0)
            assert np.isclose(times[-1], item["duration_ms"])
            assert np.allclose(np.diff(times), item["output_interval_ms"])
            parameters = json.loads(str(values["parameters_json"]))
            assert all(parameters[key] == value for key, value in item["parameters"].items())
            assert parameters["coordinate_unit"] == "um"
            assert np.allclose(points.max(0) - points.min(0), [20000, 7000, 3000])
            for axis in range(3):
                spacing = np.diff(np.unique(points[:, axis]))
                assert np.allclose(spacing, item["dx_um"])
            lat = values["native_activation_times_ms"]
            assert lat.shape == (len(points),) and np.isfinite(lat).all()
            fraction = float((lat >= 0).mean())
            nodes, frames = len(points), len(times)
        counts[item["split"]] = counts.get(item["split"], 0) + 1
        records.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                        "sha256": digest(path), "nodes": nodes, "frames": frames,
                        "native_activation_fraction": fraction})
    manifest = json.loads((data / "splits.json").read_text())["splits"]
    primary = [set(manifest[name]) for name in ("train", "validation", "id_test", "stimulus_ood")]
    assert all(not x & y for i, x in enumerate(primary) for y in primary[i+1:])
    ids = [r for r in plan["runs"] if r["split"] == "id_test"]
    meshes = [r for r in plan["runs"] if r["split"] == "mesh_ood"]
    assert all(x["parameters"] == y["parameters"] for x,y in zip(ids, meshes))
    train_set = OpenCARPDataset(data, "train")
    calibration = []
    for index in range(4 if args.interfaces_only else len(train_set)):
        trajectory = train_set[index]
        calibration.append((trajectory, anisotropic_fem_graph(trajectory)))
    suites = {"train": calibration[0]}
    for split in ("validation", "id_test", "mesh_ood", "stimulus_ood"):
        dataset = OpenCARPDataset(data, split)
        if dataset.files[0].is_file():
            trajectory = dataset[0]
            suites[split] = (trajectory, anisotropic_fem_graph(trajectory))
    models, physical = {}, {}
    for split, (trajectory, graph) in suites.items():
        physical[split] = physics_checks(graph)
    for name in config["models"]:
        torch.manual_seed(0)
        model = E2Predictor(name, config)
        model.fit_calibration(calibration)
        parameters_before = [p.detach().clone() for p in model.parameters()]
        checks = {}
        for split, (trajectory, graph) in suites.items():
            history = initial_history(trajectory, 10, model.history_frames, model.voltage_offset, model.voltage_scale)
            started = time.perf_counter()
            # Backward check on an independent validation window; this is not
            # parameter fitting and does not fit normalization/calibration.
            prediction = model.step(history, graph, 10.)
            target = (trajectory.voltage[11] + model.voltage_offset) / model.voltage_scale
            loss = (prediction[:, -1] - target).square().mean()
            loss.backward()
            assert torch.isfinite(loss)
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                for frame in range(11, 14):
                    prediction = model.step(prediction, graph, float(frame))
                assert torch.isfinite(prediction).all()
            checks[split] = {"forward_backward_short_rollout_finite": True,
                             "seconds": time.perf_counter() - started,
                             "internal_steps": model.internal_steps(graph, 1.)}
            print(json.dumps({"model": name, "split": split, **checks[split]}), flush=True)
        # Exercise the exact formal-training loss, not just a parallel toy path.
        trajectory, graph = suites["train"]
        started = time.perf_counter()
        loss = window_loss(model, trajectory, graph, 10, config["training"]["rollout_window"])
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        seconds = time.perf_counter() - started
        assert all(torch.equal(p, saved) for p,saved in zip(model.parameters(), parameters_before))
        models[name] = {"parameters": sum(p.numel() for p in model.parameters()), "checks": checks,
                        "training_window_forward_backward_seconds": seconds,
                        "optimizer_updates": 0, "parameters_unchanged": True}
    seed_checks = []
    trajectory, graph = suites["train"]
    for seed in config["training"]["seeds"]:
        torch.manual_seed(seed)
        model = E2Predictor("taskkernel", config)
        model.fit_calibration(calibration)
        before = [p.detach().clone() for p in model.parameters()]
        prediction = model.step(initial_history(trajectory, 0, model.history_frames,
                                model.voltage_offset, model.voltage_scale), graph, 0.)
        prediction[:, -1].square().mean().backward()
        assert torch.isfinite(prediction).all()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert all(torch.equal(p,saved) for p,saved in zip(model.parameters(),before))
        seed_checks.append({"seed": seed, "finite_forward_backward": True, "parameters_unchanged": True})
    hashes = {path.relative_to(ROOT).as_posix(): digest(path) for path in (plan_path, data / "splits.json")}
    implementation = [ROOT / "experiments/paper/opencarp_prepared.py",
                      ROOT / "experiments/paper/train_opencarp_prepared.py",
                      ROOT / "experiments/paper/evaluate_opencarp_prepared.py"]
    implementation += list((ROOT / "src/taskkernel_gnn").rglob("*.py"))
    complete = not missing and len(suites) == 5 and not args.interfaces_only
    report = {"ready_for_formal_training": complete, "training_started": False, "optimizer_updates": 0,
              "config_sha256": digest(CONFIG), "input_manifest_hashes": hashes, "files": records,
              "implementation_hashes": {path.relative_to(ROOT).as_posix(): digest(path) for path in implementation},
              "generated_counts": counts, "missing": missing, "split_disjointness": True,
              "official_commit": provenance["commit"], "official_file_sha256_matches": official_integrity,
              "taskkernel_seed_checks": seed_checks,
              "mesh_parameters_paired": True, "calibration_split": "train_only",
              "physics_checks": physical, "models": models,
              "runtime": {"torch": torch.__version__, "cuda": torch.cuda.is_available(), "threads": 2}}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / ("INTERFACE_PREFLIGHT.json" if args.interfaces_only else "PROTOCOL_READINESS.json")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(destination), "ready": complete, "missing": len(missing)}))

if __name__ == "__main__":
    main()
