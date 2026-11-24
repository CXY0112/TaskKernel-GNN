"""History-conditioned local closure for unobserved internal dynamics."""
import torch
from torch import nn


class ConditionedMLPUpdate(nn.Module):
    def __init__(self, context_features, hidden=64, eta=.1, dt=1.):
        super().__init__()
        if context_features < 1 or hidden < 1 or eta < 0 or dt <= 0:
            raise ValueError("context_features/hidden/dt must be positive and eta nonnegative")
        self.eta, self.dt = float(eta), float(dt)
        self.network = nn.Sequential(nn.Linear(context_features + 1, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, node_state, aggregate, context=None):
        closure = self.network(torch.cat([context["node_features"], aggregate], -1))
        return self.dt * (aggregate + self.eta * closure)
