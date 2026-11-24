"""Dataset-aware constructors for all frozen external baselines."""

from __future__ import annotations

from torch import nn

from .gino import GeometryInformedNeuralOperator
from .graph_pde import build_graph_pde_baseline
from .particles import EGNNParticleSimulator, GraphNetworkSimulator, ParticleMeshGraphNet
from .structured import FourierNeuralOperator2d, UNet2d


def build_external_baseline(name: str, dataset: str, track: str = "canonical") -> nn.Module:
    if track not in {"matched", "canonical"}:
        raise ValueError("track must be matched or canonical")
    small = track == "matched"
    if dataset == "pdebench":
        if name == "fno":
            return FourierNeuralOperator2d(width=8 if small else 32, modes=6 if small else 12,
                                           layers=2 if small else 4)
        if name == "unet":
            return UNet2d(width=8 if small else 32)
        if name in {"meshgraphnet", "gno", "mp_pde"}:
            return build_graph_pde_baseline(name, track, coordinate_dim=2, state_features=2, dt=1.0)
    elif dataset == "opencarp":
        if name in {"meshgraphnet", "gno", "mp_pde"}:
            return build_graph_pde_baseline(name, track, coordinate_dim=3, state_features=1, dt=1.0)
        if name == "gino":
            return GeometryInformedNeuralOperator(
                coordinate_dim=3, state_features=1, hidden_features=8 if small else 64,
                layers=1 if small else 4,
            )
    elif dataset == "waterdrop":
        common = {"hidden": 16 if small else 128, "dimensions": 2, "particle_types": 16}
        if name == "gns":
            return GraphNetworkSimulator(message_steps=2 if small else 10, **common)
        if name == "egnn":
            return EGNNParticleSimulator(layers=1 if small else 4, **common)
        if name == "meshgraphnet":
            return ParticleMeshGraphNet(message_steps=1 if small else 4, **common)
    raise ValueError(f"baseline {name!r} is not configured for dataset {dataset!r}")
