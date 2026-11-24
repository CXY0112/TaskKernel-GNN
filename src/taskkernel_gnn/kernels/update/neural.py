"""Neural and recurrent node-update kernels."""

from __future__ import annotations
import torch
from torch import nn


class GRUUpdate(nn.Module):
    def __init__(self,state_features,message_features): super().__init__(); self.cell=nn.GRUCell(message_features,state_features)
    def forward(self,node_state,aggregate,context=None): return self.cell(aggregate,node_state)-node_state


class MLPUpdate(nn.Module):
    def __init__(self,state_features,message_features,output_features,hidden=64,eta=0.1):
        super().__init__(); self.eta=float(eta); self.network=nn.Sequential(nn.Linear(state_features+message_features,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,output_features)); nn.init.zeros_(self.network[-1].weight); nn.init.zeros_(self.network[-1].bias)
    def forward(self,node_state,aggregate,context=None): return self.eta*torch.tanh(self.network(torch.cat([node_state,aggregate],-1)))


class ParticleResidualMLPUpdate(MLPUpdate):
    """Second-order kinematic step plus a learned node/message correction."""

    def __init__(self, state_features, message_features, output_features,
                 hidden=64, eta=0.1, dt=0.0025, spatial_dims=2):
        super().__init__(state_features, message_features, output_features, hidden, eta)
        self.dt = float(dt)
        self.spatial_dims = int(spatial_dims)
        if output_features != 2 * self.spatial_dims:
            raise ValueError("particle residual output must contain position and velocity increments")

    def forward(self, node_state, aggregate, context=None):
        velocity = node_state[..., self.spatial_dims:2 * self.spatial_dims]
        base = torch.cat((self.dt * velocity, torch.zeros_like(velocity)), dim=-1)
        return base + super().forward(node_state, aggregate, context)
