"""Second-order mechanics update kernels."""

from __future__ import annotations
import torch
from torch import nn
from ..base import positive, raw_positive


class SecondOrderMechanicsUpdate(nn.Module):
    def __init__(self,dt,spatial_dims=2,damping=0.0): super().__init__(); self.dt=float(dt); self.spatial_dims=int(spatial_dims); self.raw_damping=nn.Parameter(torch.tensor(float(max(damping,1e-8))).log())
    def forward(self,node_state,force,context=None):
        d=self.spatial_dims; velocity=node_state[...,d:2*d]; acceleration=force/context["mass"].unsqueeze(-1)-self.raw_damping.exp()*velocity
        return torch.cat([self.dt*velocity,self.dt*acceleration],-1)


class ExplicitEulerUpdate(SecondOrderMechanicsUpdate): pass


class SemiImplicitEulerUpdate(SecondOrderMechanicsUpdate):
    def forward(self,node_state,force,context=None):
        d=self.spatial_dims; velocity=node_state[...,d:2*d]; acceleration=force/context["mass"].unsqueeze(-1)-self.raw_damping.exp()*velocity; next_velocity=velocity+self.dt*acceleration
        return torch.cat([self.dt*next_velocity,self.dt*acceleration],-1)


class BoundaryContactUpdate(SecondOrderMechanicsUpdate):
    """Second-order update with learnable uniform field and box-wall contact."""

    def __init__(self, dt, spatial_dims=2, damping=0.01, wall_strength=1.0,
                 contact_radius=0.015, field=None):
        super().__init__(dt, spatial_dims, damping)
        self.raw_wall_strength = nn.Parameter(raw_positive(wall_strength))
        self.contact_radius = float(contact_radius)
        if self.contact_radius <= 0:
            raise ValueError("contact_radius must be positive")
        initial_field = [0.0] * spatial_dims if field is None else field
        if len(initial_field) != spatial_dims:
            raise ValueError("field length must equal spatial_dims")
        self.field = nn.Parameter(torch.as_tensor(initial_field, dtype=torch.float32))

    def forward(self, node_state, force, context=None):
        dimensions = self.spatial_dims
        position = node_state[..., :dimensions]
        velocity = node_state[..., dimensions:2 * dimensions]
        bounds = torch.as_tensor(context["bounds"], device=position.device, dtype=position.dtype)
        left = ((self.contact_radius - (position - bounds[:, 0])).clamp_min(0)
                / self.contact_radius)
        right = ((self.contact_radius - (bounds[:, 1] - position)).clamp_min(0)
                 / self.contact_radius)
        acceleration = (force / context["mass"].unsqueeze(-1)
                        + self.field + positive(self.raw_wall_strength) * (left - right)
                        - self.raw_damping.exp() * velocity)
        next_velocity = velocity + self.dt * acceleration
        return torch.cat((self.dt * next_velocity, self.dt * acceleration), -1)
