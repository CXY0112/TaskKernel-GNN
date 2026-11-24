"""Augmented-state wave update kernels."""

from __future__ import annotations
import torch
from torch import nn


class WaveStateUpdate(nn.Module):
    def __init__(self,dt,damping=0.0): super().__init__(); self.dt=float(dt); self.raw_damping=nn.Parameter(torch.tensor(float(max(damping,1e-8))).log())
    def forward(self,node_state,aggregate,context=None):
        if node_state.shape[-1]!=2: raise ValueError("WaveStateUpdate expects [...,2]=[u,v]")
        velocity=node_state[...,1:2]; acceleration=aggregate[...,:1]-self.raw_damping.exp()*velocity
        return self.dt*torch.cat([velocity,acceleration],-1)
