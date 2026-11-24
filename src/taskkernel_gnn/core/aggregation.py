"""Permutation-invariant neighborhood aggregation."""

from __future__ import annotations

import torch


def aggregate_messages(messages: torch.Tensor, target: torch.Tensor, nodes: int, reduction: str = "add") -> torch.Tensor:
    """Aggregate edge messages at target nodes using ADD or MEAN."""
    if messages.shape[0] != target.numel():
        raise ValueError("one target index is required per edge message")
    output = messages.new_zeros((nodes, *messages.shape[1:]))
    output.index_add_(0, target, messages)
    if reduction == "add":
        return output
    if reduction == "mean":
        count = messages.new_zeros(nodes)
        count.index_add_(0, target, torch.ones_like(target, dtype=messages.dtype))
        return output / count.clamp_min(1).view(nodes, *([1] * (messages.ndim - 1)))
    raise ValueError("reduction must be 'add' or 'mean'")

