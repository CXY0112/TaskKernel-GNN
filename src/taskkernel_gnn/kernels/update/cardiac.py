"""Cardiac ionic-reaction update kernels."""

from __future__ import annotations
import torch
from torch import nn


def aliev_panfilov_reaction(state,k,a,b,epsilon,mu1,mu2):
    voltage,recovery=state.unbind(-1); dv=-k*voltage*(voltage-a)*(voltage-1)-voltage*recovery
    rate=epsilon+mu1*recovery/(voltage+mu2); dw=rate*(-recovery-k*voltage*(voltage-b-1))
    return torch.stack([dv,dw],-1)


def fitzhugh_nagumo_reaction(state,epsilon=0.02,offset=0.10,recovery_scale=0.80):
    voltage,recovery=state.unbind(-1)
    return torch.stack([voltage-voltage.pow(3)/3-recovery,epsilon*(voltage+offset-recovery_scale*recovery)],-1)


class AlievPanfilovUpdate(nn.Module):
    def __init__(self,dt=1.0,k=7.8,a=0.085,b=0.105,epsilon=0.0105,mu1=0.115,mu2=0.31):
        super().__init__(); self.dt=float(dt); self.raw=nn.Parameter(torch.tensor([k,a,b,epsilon,mu1,mu2]).clamp_min(1e-8).log())
    def forward(self,node_state,aggregate,context=None):
        reaction=aliev_panfilov_reaction(node_state,*self.raw.exp()).clone(); reaction[...,0]+=aggregate[...,0]; return self.dt*reaction


class FitzHughNagumoUpdate(nn.Module):
    def __init__(self,dt=1.0,epsilon=0.02): super().__init__(); self.dt=float(dt); self.raw_epsilon=nn.Parameter(torch.tensor(float(epsilon)).clamp_min(1e-8).log())
    def forward(self,node_state,aggregate,context=None):
        reaction=fitzhugh_nagumo_reaction(node_state,self.raw_epsilon.exp()).clone(); reaction[...,0]+=aggregate[...,0]; return self.dt*reaction


class CardiacDecayUpdate(nn.Module):
    def __init__(self,dt=1.0,rate=0.2): super().__init__(); self.dt=float(dt); self.raw_rate=nn.Parameter(torch.tensor(float(rate)).clamp_min(1e-8).log())
    def forward(self,node_state,aggregate,context=None):
        derivative=(-self.raw_rate.exp()*node_state).clone(); derivative[...,0]+=aggregate[...,0]; return self.dt*derivative
