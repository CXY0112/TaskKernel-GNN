"""Model wrapper for stacking TaskKernel layers."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn

from ..core.layer import TaskKernelLayer
from ..kernels.registry import build_kernel_bank


class TaskKernelGNN(nn.Module):
    def __init__(self, layers: Sequence[TaskKernelLayer]):
        super().__init__()
        if not layers:
            raise ValueError("TaskKernelGNN requires at least one layer")
        self.layers = nn.ModuleList(layers)

    def forward(self, node_state, edge_index, edge_attr=None, context=None, *, progress: float = 1.0):
        diagnostics = []
        for layer in self.layers:
            node_state, layer_diagnostics = layer(node_state, edge_index, edge_attr, context, progress=progress)
            diagnostics.append(layer_diagnostics)
        return node_state, diagnostics

    def gate_regularization(self, progress: float = 1.0):
        """Sum scheduled gate penalties across all TaskKernel layers."""
        return sum((layer.gate_regularization(progress) for layer in self.layers), start=0)

    @classmethod
    def from_specs(
        cls,
        message_specs,
        update_specs,
        *,
        config,
        message_calibration_scales=None,
        update_calibration_scales=None,
    ):
        """Build a one-layer model from registry specifications."""
        messages, message_names = build_kernel_bank(message_specs, stage="message")
        updates, update_names = build_kernel_bank(update_specs, stage="update")
        layer = TaskKernelLayer(
            messages,
            updates,
            message_names=message_names,
            update_names=update_names,
            calibration_scales=message_calibration_scales,
            update_calibration_scales=update_calibration_scales,
            config=config,
        )
        return cls([layer])

    @classmethod
    def from_config(cls, config, *, message_calibration_scales=None, update_calibration_scales=None):
        """Build directly from the kernel specifications stored in a config."""
        config.validate()
        if not config.kernels.message_specs or not config.kernels.update_specs:
            raise ValueError("config requires nonempty kernels.message_specs and kernels.update_specs")
        return cls.from_specs(
            config.kernels.message_specs,
            config.kernels.update_specs,
            config=config,
            message_calibration_scales=message_calibration_scales,
            update_calibration_scales=update_calibration_scales,
        )
