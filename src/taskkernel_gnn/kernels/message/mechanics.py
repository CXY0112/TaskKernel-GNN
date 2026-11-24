"""Mechanics message kernels and dense mixed-force helpers."""

from __future__ import annotations
import torch
from torch import nn
from ..base import pair_geometry, positive, raw_positive, require_context


class ViscousDampingKernel(nn.Module):
    def __init__(self, damping=1.0, velocity_key="velocity"):
        super().__init__(); self.velocity_key = velocity_key; self.raw_damping = nn.Parameter(raw_positive(damping))
    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        source, target = edge_index; velocity = require_context(context, self.velocity_key)
        return positive(self.raw_damping) * (velocity[source] - velocity[target])


class SoftenedGravityKernel(nn.Module):
    def __init__(self, epsilon=0.1): super().__init__(); self.epsilon = float(epsilon)
    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        source, target, delta, distance, _ = pair_geometry(context, edge_index)
        mass = require_context(context, "mass"); gravity = torch.as_tensor(context["gravity_g"], device=delta.device, dtype=delta.dtype)
        scale = gravity * mass[source] * mass[target] / (distance.squeeze(-1).square()+self.epsilon**2).pow(1.5)
        return scale.unsqueeze(-1) * delta


class SpringKernel(nn.Module):
    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        source, target, _, distance, direction = pair_geometry(context, edge_index)
        stiffness, rest = context["spring_k"], context["rest_length"]
        return (stiffness * (distance.squeeze(-1)-rest)).unsqueeze(-1) * direction


class PairwiseDashpotKernel(nn.Module):
    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        source, target, _, _, direction = pair_geometry(context, edge_index)
        radial = ((context["velocity"][source]-context["velocity"][target])*direction).sum(-1)
        return (context["dashpot_gamma"]*radial).unsqueeze(-1)*direction


class ShortRangeRepulsionKernel(nn.Module):
    """Compact-support pair repulsion independent of a particular simulator."""

    def __init__(self, strength=1.0, radius=0.015):
        super().__init__()
        self.raw_strength = nn.Parameter(raw_positive(strength))
        self.radius = float(radius)
        if self.radius <= 0:
            raise ValueError("repulsion radius must be positive")

    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        _, _, _, distance, direction = pair_geometry(context, edge_index)
        overlap = (self.radius - distance).clamp_min(0) / self.radius
        return -positive(self.raw_strength) * overlap * direction


def dense_particle_components(batch, epsilon=0.1, gaussian_sigma=0.55):
    position, velocity = batch["position"], batch["velocity"]
    delta = position[:,None,:,:]-position[:,:,None,:]; distance2 = delta.square().sum(-1).clamp_min(1e-12)
    distance = distance2.sqrt(); direction = delta/distance.unsqueeze(-1)
    relative_velocity = velocity[:,None,:,:]-velocity[:,:,None,:]; radial = (relative_velocity*direction).sum(-1)
    n=position.shape[1]; mask=(1.0-torch.eye(n,device=position.device)).unsqueeze(0)
    mass_product=batch["mass"][:,:,None]*batch["mass"][:,None,:]
    gravity=((batch["gravity_g"][:,None,None]*mass_product/(distance2+epsilon**2).pow(1.5)).unsqueeze(-1)*delta*mask.unsqueeze(-1)).sum(2)
    spring=((batch["bond"]*batch["spring"]*(distance-batch["rest"])).unsqueeze(-1)*direction).sum(2)
    dashpot=((batch["bond"]*batch["dashpot"]*radial).unsqueeze(-1)*direction).sum(2)
    gaussian=((0.02*torch.exp(-distance2/(2*gaussian_sigma**2))*mask).unsqueeze(-1)*direction).sum(2)
    return {"gravity":gravity,"spring":spring,"dashpot":dashpot,"gaussian":gaussian}


class EquivariantPairMLP(nn.Module):
    def __init__(self,input_features=8,hidden=48,output_scale=0.06):
        super().__init__(); self.output_scale=float(output_scale)
        self.net=nn.Sequential(nn.Linear(input_features,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,features,direction,mask):
        scalar=self.output_scale*torch.tanh(self.net(features).squeeze(-1))*mask
        return (scalar.unsqueeze(-1)*direction).sum(2)
