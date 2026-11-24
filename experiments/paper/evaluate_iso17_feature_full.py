"""Evaluate frozen C checkpoints on every official ISO17 test frame."""

import argparse
import hashlib
import json
import time

import numpy as np
import torch

from train_iso17_feature_kernels import ROOT
from iso17_data import Frames, topology_map, protocol
from iso17_joint_training import energy_force_metrics
from iso17_test_data import cached_frames, subset
from taskkernel_gnn.models.molecular_features import MolecularFeatureEnergyModel
from iso17_full_topology_adapter import adapt_trajectory_ends


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(model, frames, energies, tops, offset, e_scale, f_scale, scope):
    totals = {}
    n = len(frames)
    for start in range(0, n, 4096):
        stop = min(start + 4096, n)
        block = Frames(frames.position[start:stop], frames.force[start:stop],
                       frames.numbers[start:stop], frames.molecules[start:stop],
                       list(range(start, stop)))
        item = energy_force_metrics(model, block, energies[start:stop], tops,
                                    offset, e_scale, f_scale, chunk=32)
        for key, value in item.items():
            if key != "frames":
                assert np.isfinite(value), (scope, start, key)
                totals[key] = totals.get(key, 0.) + (value ** 2 if "rmse" in key else value) * (stop-start)
        print(f"{scope}: {stop}/{n}", flush=True)
    return {"frames": n, **{k: (v/n)**.5 if "rmse" in k else v/n for k,v in totals.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0,42,37))
    args = parser.parse_args()
    torch.set_num_threads(4)
    within, within_energy = cached_frames("test_within", 101000, 1000, 0)
    other, other_energy = cached_frames("test_other", 130000, 5000, 101)
    tops = topology_map(valence_cap=True)
    other, other_tops, topology_changes = adapt_trajectory_ends(other, tops, 5000)
    for seed in ([args.seed] if args.seed is not None else [0,42,37]):
        folder = ROOT / "C" / f"budget_128_seed_{seed}"
        checkpoint = folder / "checkpoint.pt"
        digest = sha256(checkpoint)
        output = folder / "full_test_metrics.json"
        if output.exists():
            previous = json.loads(output.read_text())
            if previous.get("checkpoint_sha256") == digest and previous.get("input_topology_adapter"):
                print(f"Already complete seed={seed}", flush=True)
                continue
        began = time.perf_counter()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["budget"] == 128 and payload["seed"] == seed
        assert payload["protocol"] == protocol()
        meta = payload["experiment_metadata"]
        assert meta["variant"] == "C"
        model = MolecularFeatureEnergyModel(variant="C", **meta["architecture"])
        model.load_state_dict(payload["model"])
        model.eval()
        train_molecules = set(payload["protocol"]["train_molecules"])
        seen, e_seen = subset(within, within_energy, train_molecules)
        unseen, e_unseen = subset(within, within_energy, set(range(101))-train_molecules)
        report = {"model":"TaskKernel-GNN C", "budget_frames":128, "seed":seed,
                  "parameters":sum(p.numel() for p in model.parameters()),
                  "selected_step":payload["selected_step"], "experiment_metadata":meta,
                  "checkpoint_sha256":digest,
                  "input_topology_adapter":{"policy":"trajectory-end topology derived from actual input numbers and coordinates with existing valence-capped rule; no labels; all frames retained",
                                            "changed_test_within_frames":0,"changed_test_other_frames":len(topology_changes),"changes":topology_changes},
                  "evaluation_scope":"complete official test_within and test_other; no checkpoint selection"}
        for scope, frames, energy in (("test_within_seen",seen,e_seen),
                                     ("test_within_unseen_reference",unseen,e_unseen),
                                     ("test_other_unseen_molecules",other,other_energy)):
            report[scope] = evaluate(model, frames, energy, other_tops if scope=="test_other_unseen_molecules" else tops,
                                    payload["energy_offset_train_only_ev"],
                                    payload["energy_scale_train_only_ev"],
                                    payload["force_scale_train_only_ev_per_ang"],scope)
        assert sha256(checkpoint) == digest
        report["wall_time_seconds"] = time.perf_counter()-began
        output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        print(f"DONE seed={seed} seconds={report['wall_time_seconds']:.1f}",flush=True)


if __name__ == "__main__":
    main()
