"""Adapters for graph baselines on cardiac electrophysiology trajectories."""

from __future__ import annotations

import torch
from torch import nn

from .graph_pde import build_graph_pde_baseline


class CardiacStimulusAdapter(nn.Module):
    """Apply the known pacing stimulus around a learned graph time step."""

    def __init__(self, graph_model: nn.Module, *, dt: float) -> None:
        super().__init__()
        self.graph_model = graph_model
        self.dt = dt

    def step(self, state: torch.Tensor, sample, time_index: int) -> torch.Tensor:
        result = self.graph_model.step(state, sample, time_index)
        if time_index < sample.stimulus_steps:
            result = result.clone()
            result[sample.stimulus_mask, 0] += self.dt * sample.stimulus_amplitude
        return result


def build_cardiac_baseline(name: str, track: str, *, dt: float = 0.02) -> nn.Module:
    graph_model = build_graph_pde_baseline(
        name,
        track,
        coordinate_dim=3,
        state_features=2,
        dt=dt,
    )
    return CardiacStimulusAdapter(graph_model, dt=dt)
