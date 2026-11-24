"""Common identity, linear, and generic Euler update kernels."""

from __future__ import annotations
from collections.abc import Callable
from torch import nn


class IdentityMessageUpdate(nn.Module):
    def forward(self,node_state,aggregate,context=None): return aggregate


class LinearResidualUpdate(nn.Module):
    def __init__(self,message_features=None,output_features=None,step_size=1.0):
        super().__init__(); self.step_size=float(step_size)
        self.projection=nn.Identity() if message_features is None or output_features is None or message_features==output_features else nn.Linear(message_features,output_features,bias=False)
    def forward(self,node_state,aggregate,context=None): return self.step_size*self.projection(aggregate)


class StateMessageLinearUpdate(nn.Module):
    def __init__(self,state_features,message_features,output_features,step_size=1.0):
        super().__init__(); self.step_size=float(step_size); self.state=nn.Linear(state_features,output_features,bias=False); self.message=nn.Linear(message_features,output_features,bias=False)
    def forward(self,node_state,aggregate,context=None): return self.step_size*(self.state(node_state)+self.message(aggregate))


class _CallableDynamics(nn.Module):
    def __init__(self,function:Callable): super().__init__(); self.function=function
    def forward(self,node_state,aggregate,context=None): return self.function(node_state,aggregate,context)


class DynamicsEulerUpdate(nn.Module):
    def __init__(self,dynamics,dt): super().__init__(); self.dynamics=dynamics if isinstance(dynamics,nn.Module) else _CallableDynamics(dynamics); self.dt=float(dt)
    def forward(self,node_state,aggregate,context=None): return self.dt*self.dynamics(node_state,aggregate,context)


ResidualUpdate=IdentityMessageUpdate
