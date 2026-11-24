"""Energy-and-force supervised ISO17 TaskKernel-GNN with conservative forces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from iso17_data import ATOMS, RAW, Frames, graph, protocol, read_frames, topology_map
from ase.db import connect


def read_energies(path: Path, ids: list[int]) -> np.ndarray:
    with connect(path) as conn:
        energy = np.asarray([float(conn.get(id=int(entry)).total_energy) for entry in ids], dtype=np.float64)
    if not np.isfinite(energy).all():
        raise ValueError(f"nonfinite ISO17 energy labels from {path}")
    return energy


def energy_force_metrics(model: torch.nn.Module, frames: Frames,
                         energies: np.ndarray, tops, offset: float, energy_scale: float,
                         force_scale: float, chunk: int = 16) -> dict:
    model.eval()
    e_abs = e_sq = f_abs = f_sq = 0.0
    for start in range(0, len(frames), chunk):
        end = min(start + chunk, len(frames))
        # Official full-test arrays are read-only memmaps; autograd needs a
        # writable per-chunk position tensor, not a view of that storage.
        edge_index, context = graph(np.array(frames.position[start:end], copy=True),
                                    frames.molecules[start:end], tops)
        context["position"].requires_grad_(True)
        prediction, force, _ = model(edge_index, context)
        e_error = prediction.detach().numpy().astype(np.float64) - (energies[start:end] - offset)
        f_error = force.detach().numpy().astype(np.float64) - frames.force[start:end]
        e_abs += float(np.abs(e_error).sum())
        e_sq += float(np.square(e_error).sum())
        f_abs += float(np.abs(f_error).sum())
        f_sq += float(np.square(f_error).sum())
    n = len(frames)
    e_mse = e_sq / n
    f_mse = f_sq / (n * ATOMS * 3)
    return {
        "frames": n,
        "energy_mae_ev_per_molecule": e_abs / n,
        "energy_rmse_ev_per_molecule": e_mse ** 0.5,
        "force_component_mae_ev_per_ang": f_abs / (n * ATOMS * 3),
        "force_component_rmse_ev_per_ang": f_mse ** 0.5,
        "normalized_joint_mse": e_mse / energy_scale ** 2 + f_mse / force_scale ** 2,
    }


def run(args: argparse.Namespace, *, model_factory, experiment_metadata) -> None:
    began = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    specification = protocol()
    per_molecule = specification["budgets_per_molecule"][str(args.budget)]
    train_ids = [entry for mol in specification["train_molecules"]
                 for entry in specification["train_ids_by_molecule"][mol][:per_molecule]]
    train = read_frames(RAW / "reference.db", train_ids, 4000)
    valid = read_frames(RAW / "reference.db", specification["validation_ids"], 4000)
    train_energy = read_energies(RAW / "reference.db", train_ids)
    valid_energy = read_energies(RAW / "reference.db", specification["validation_ids"])
    offset = float(train_energy.mean())
    energy_scale = float(train_energy.std())
    force_scale = float(np.sqrt(np.mean(np.square(train.force))))
    if energy_scale < 1e-6 or force_scale < 1e-6:
        raise ValueError("degenerate training-label scale")
    tops = topology_map(valence_cap=True)
    model = model_factory(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    best = float("inf")
    best_step = 0
    best_state = None
    history = []
    for step in range(args.steps + 1):
        if step % args.validate_every == 0 or step == args.steps:
            metrics = energy_force_metrics(model, valid, valid_energy, tops, offset, energy_scale, force_scale)
            score = metrics["normalized_joint_mse"]
            history.append({"step": step, **metrics})
            if score < best:
                best, best_step = score, step
                best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            print(f"budget={args.budget} seed={args.seed} step={step} "
                  f"val_energy_mae={metrics['energy_mae_ev_per_molecule']:.5f} "
                  f"val_force_mae={metrics['force_component_mae_ev_per_ang']:.5f} "
                  f"joint={score:.5f}", flush=True)
        if step == args.steps:
            break
        model.train()
        selection = np.random.choice(len(train), min(args.batch_size, len(train)), replace=False)
        edge_index, context = graph(train.position[selection], train.molecules[selection], tops)
        context["position"].requires_grad_(True)
        prediction, force, info = model(edge_index, context,
                                        progress=min(1.0, step / args.anneal_steps),
                                        create_graph=True)
        energy_target = torch.from_numpy((train_energy[selection] - offset).astype(np.float32))
        force_target = torch.from_numpy(train.force[selection])
        energy_loss = ((prediction - energy_target) / energy_scale).square().mean()
        force_loss = ((force - force_target) / force_scale).square().mean()
        loss = energy_loss + force_loss + info["gate_regularization"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite joint loss at step {step}")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.load_state_dict(best_state)
    model.eval()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "experiment_metadata": experiment_metadata,
        "budget_frames": args.budget, "seed": args.seed, "steps": args.steps,
        "identity_update_only": args.identity_update_only,
        "end_temperature": args.end_temperature,
        "triplet_mlp": args.triplet_mlp,
        "selected_step": best_step, "parameters": sum(p.numel() for p in model.parameters()),
        "energy_offset_train_only_ev": offset,
        "energy_scale_train_only_ev": energy_scale,
        "force_scale_train_only_ev_per_ang": force_scale,
        "loss": "MSE(centered energy / train SD) + MSE(force / train RMS) + gate regularization",
        "topology": "first-frame distance bonds with H1/C4/O2 valence cap",
        "validation_selection_metric": "normalized_joint_mse",
        "validation": energy_force_metrics(model, valid, valid_energy, tops, offset, energy_scale, force_scale),
        "message_gate": {name: float(weight.detach()) for name, weight in zip(model.layer.message_names, model.layer.message_gate())},
        "update_gate": {name: float(weight.detach()) for name, weight in zip(model.layer.update_names, model.layer.update_gate())},
        "history": history, "wall_time_seconds": time.perf_counter() - began,
    }
    if not args.validation_only:
        for name, database, ids, block_size, molecule_offset in (
            ("test_within_seen_fixed_subset", "test_within.db", specification["test_within_seen_ids"], 1000, 0),
            ("test_within_unseen_reference_fixed_subset", "test_within.db", specification["test_within_unseen_ids"], 1000, 0),
            ("test_other_fixed_subset", "test_other.db", specification["test_other_force_ids"], 5000, 101),
        ):
            frames = read_frames(RAW / database, ids, block_size, offset=molecule_offset)
            energies = read_energies(RAW / database, ids)
            metrics[name] = energy_force_metrics(model, frames, energies, tops, offset, energy_scale, force_scale)
    metrics["evaluation_scope"] = "validation_only" if args.validation_only else "validation_and_frozen_heldout_subset"
    metrics["wall_time_seconds"] = time.perf_counter() - began
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    torch.save({"model": model.state_dict(), "budget": args.budget, "seed": args.seed,
                "experiment_metadata": experiment_metadata,
                "hidden": args.hidden, "mlp_amplitude": args.mlp_amplitude,
                "identity_update_only": args.identity_update_only,
                "end_temperature": args.end_temperature,
                "triplet_mlp": args.triplet_mlp,
                "energy_offset_train_only_ev": offset, "energy_scale_train_only_ev": energy_scale,
                "force_scale_train_only_ev_per_ang": force_scale,
                "protocol": specification, "selected_step": best_step}, output / "checkpoint.pt")
    print(json.dumps({key: value for key, value in metrics.items() if key != "history"}, indent=2), flush=True)
