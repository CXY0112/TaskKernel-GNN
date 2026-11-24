"""Continuous coefficient correction with shared causal context features."""
import torch
from torch import nn


class ConditionedDiffusionMessage(nn.Module):
    def __init__(self, context_features, hidden=64, eta=.1, symmetric=False):
        super().__init__()
        if context_features < 1 or hidden < 1 or eta < 0:
            raise ValueError("context_features/hidden must be positive and eta nonnegative")
        self.eta, self.symmetric = float(eta), bool(symmetric)
        self.network = nn.Sequential(nn.Linear(2 * context_features + 3, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        source, target = edge_index
        features = context["node_features"]
        delta = context["position"][source] - context["position"][target]
        direction = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        coefficient = self.network(torch.cat([features[source], features[target], direction], -1))
        if self.symmetric:
            reverse = self.network(torch.cat([features[target], features[source], -direction], -1))
            coefficient = .5 * (coefficient + reverse)
        return self.eta * coefficient.tanh() * context["edge_weight"][:, None] * (node_state[source] - node_state[target])


class GaussianAdapterDiffusionMessage(nn.Module):
    def __init__(self, scale=1., beta=1.):
        super().__init__()
        if scale <= 0 or beta <= 0:
            raise ValueError("scale and beta must be positive")
        self.log_scale = nn.Parameter(torch.tensor(float(scale)).log())
        self.log_beta = nn.Parameter(torch.tensor(float(beta)).log())

    def forward(self, node_state, edge_index, edge_attr=None, context=None):
        source, target = edge_index
        distance2 = (context["position"][source] - context["position"][target]).square().sum(-1)
        weight = self.log_scale.exp() * context["edge_weight"] * torch.exp(-self.log_beta.exp() * distance2)
        return weight[:, None] * (node_state[source] - node_state[target])
