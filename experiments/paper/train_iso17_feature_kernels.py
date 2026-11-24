"""Frozen 128-frame A/B/C feature-message comparison; same joint trainer."""

import argparse
from pathlib import Path

from iso17_joint_training import run
from iso17_data import PROJECT
from taskkernel_gnn.models.molecular_features import MolecularFeatureEnergyModel


ROOT = PROJECT / "results/SOTA_Baseline/external/iso17/energy_force/feature_kernels"
CONFIG = dict(features=32, rounds=2, pair_hidden=64, triplet_hidden=32,
              mlp_amplitude=0.2, radial_basis=32, filter_hidden=32,
              update_hidden=64, cutoff=8.0, end_temperature=0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("A", "B", "C"), required=True)
    parser.add_argument("--seed", type=int, choices=(0,42,37), required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--validate-every", type=int, default=300)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.budget = 128
    args.anneal_steps, args.batch_size, args.learning_rate = 1000, 16, 1e-3
    args.hidden, args.mlp_amplitude = 64, 0.2
    args.identity_update_only, args.triplet_mlp, args.end_temperature = False, args.variant != "B", 0.5
    args.output = args.output or ROOT / args.variant / f"budget_128_seed_{args.seed}"
    run(args, model_factory=lambda _: MolecularFeatureEnergyModel(variant=args.variant, **CONFIG),
        experiment_metadata={"variant": args.variant, "architecture": CONFIG,
                             "scope": "128-frame pilot, frozen validation and heldout subsets",
                             "parameter_matching": False})


if __name__ == "__main__":
    main()
