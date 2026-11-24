"""Reusable structural variants used by the experiment protocols."""

from __future__ import annotations

import torch
from torch import nn


class LearnedSingle(nn.Module):
    """One selected kernel with a positive learned configuration multiplier."""
    def __init__(self, kernel: nn.Module):
        super().__init__()
        self.kernel = kernel
        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, *args, **kwargs):
        return self.log_scale.exp() * self.kernel(*args, **kwargs)


class MLPOnly(nn.Module):
    """Named wrapper for an all-neural message/update model."""
    def __init__(self, message: nn.Module, update: nn.Module):
        super().__init__()
        self.message = message
        self.update = update

    def forward(self, *message_args, update_args=(), **message_kwargs):
        return self.update(*update_args, self.message(*message_args, **message_kwargs))


class PhysicsResidual(nn.Module):
    """Direct sum of an analytic predictor and a bounded neural residual."""
    def __init__(self, physics: nn.Module, residual: nn.Module, eta: float = 0.1):
        super().__init__()
        self.physics = physics
        self.residual = residual
        self.eta = float(eta)

    def forward(self, *args, **kwargs):
        return self.physics(*args, **kwargs) + self.eta * self.residual(*args, **kwargs)

