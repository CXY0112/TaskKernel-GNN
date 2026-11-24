"""Shared helpers for kernel implementations; this module contains no kernels."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def positive(raw: torch.Tensor) -> torch.Tensor:
    return F.softplus(raw)


def raw_positive(value: float | list[float] | tuple[float, ...]) -> torch.Tensor:
    values = torch.as_tensor(value, dtype=torch.float32).clamp_min(1e-8)
    if values.ndim > 1:
        raise ValueError("positive kernel configuration must be scalar or one-dimensional")
    return values + torch.log(-torch.expm1(-values))


def require_context(context, key: str):
    if context is None or key not in context:
        raise KeyError(f"kernel requires context[{key!r}]")
    return context[key]


def pair_geometry(context, edge_index, key: str = "position"):
    source, target = edge_index
    position = require_context(context, key)
    delta = position[source] - position[target]
    distance = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return source, target, delta, distance, delta / distance
