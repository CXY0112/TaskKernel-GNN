"""Reusable message-generation, aggregation, and node-update layer."""

from __future__ import annotations

from collections.abc import Sequence
import copy

import torch
from torch import nn

from ..config import GateConfig, TaskKernelConfig
from .aggregation import aggregate_messages
from .calibration import FixedRMSCalibrator
from .gating import TemperatureSoftmaxGate, simplex_entropy


class TaskKernelLayer(nn.Module):
    """Combine heterogeneous message and update kernels with separate gates.

    Message kernels must implement ``forward(node_state, edge_index, edge_attr,
    context) -> [E, F]``. Update kernels must implement ``forward(node_state,
    aggregate, context) -> [N, H]``. This common interface keeps task-specific
    physics in kernels and the adaptive selection mechanism in one place.
    """

    def __init__(
        self,
        message_kernels: Sequence[nn.Module],
        update_kernels: Sequence[nn.Module],
        *,
        message_names: Sequence[str] | None = None,
        update_names: Sequence[str] | None = None,
        aggregation: str = "add",
        message_temperature: float = 0.5,
        update_temperature: float = 0.3,
        calibration_scales: Sequence[float] | None = None,
        update_calibration_scales: Sequence[float] | None = None,
        config: TaskKernelConfig | None = None,
    ) -> None:
        super().__init__()
        if not message_kernels or not update_kernels:
            raise ValueError("message and update kernel banks must be nonempty")
        if config is None:
            self.config = TaskKernelConfig(
                aggregation=aggregation,
                message_gate=GateConfig(end_temperature=message_temperature),
                update_gate=GateConfig(end_temperature=update_temperature),
            ).validate()
        else:
            self.config = copy.deepcopy(config).validate()
            aggregation = self.config.aggregation
            if self.config.kernels.message_names:
                message_names = self.config.kernels.message_names
            if self.config.kernels.update_names:
                update_names = self.config.kernels.update_names
        self.message_kernels = nn.ModuleList(message_kernels)
        self.update_kernels = nn.ModuleList(update_kernels)
        self.message_names = tuple(message_names or [f"message_{i}" for i in range(len(message_kernels))])
        self.update_names = tuple(update_names or [f"update_{i}" for i in range(len(update_kernels))])
        if len(self.message_names) != len(message_kernels) or len(self.update_names) != len(update_kernels):
            raise ValueError("kernel names must match their banks")
        self.aggregation = aggregation
        calibration = self.config.calibration
        self.message_calibrator = FixedRMSCalibrator(
            self.message_names, calibration_scales, minimum=calibration.minimum_rms,
            minimum_scale=calibration.minimum_scale, maximum_scale=calibration.maximum_scale,
        )
        self.update_calibrator = FixedRMSCalibrator(
            self.update_names, update_calibration_scales, minimum=calibration.minimum_rms,
            minimum_scale=calibration.minimum_scale, maximum_scale=calibration.maximum_scale,
        )
        message_logits = self.config.message_gate.initial_logits or None
        update_logits = self.config.update_gate.initial_logits or None
        self.message_gate = TemperatureSoftmaxGate(
            len(message_kernels),
            initial_logits=None if message_logits is None else torch.tensor(message_logits),
            **self.config.message_gate.gate_kwargs(),
        )
        self.update_gate = TemperatureSoftmaxGate(
            len(update_kernels),
            initial_logits=None if update_logits is None else torch.tensor(update_logits),
            **self.config.update_gate.gate_kwargs(),
        )

    def candidate_messages(self, node_state, edge_index, edge_attr=None, context=None) -> list[torch.Tensor]:
        return [kernel(node_state, edge_index, edge_attr, context) for kernel in self.message_kernels]

    @property
    def calibrator(self):
        """Backward-compatible alias for the message calibrator."""
        return self.message_calibrator

    def gate_regularization(self, progress: float = 1.0) -> torch.Tensor:
        """Return scheduled message/update entropy penalties for the training loss."""
        message = self.config.message_gate
        update = self.config.update_gate
        message_entropy = simplex_entropy(
            self.message_gate(progress),
            normalized=message.normalize_entropy,
            epsilon=message.entropy_epsilon,
        )
        update_entropy = simplex_entropy(
            self.update_gate(progress),
            normalized=update.normalize_entropy,
            epsilon=update.entropy_epsilon,
        )
        return (
            message.entropy_coefficient(progress) * message_entropy
            + update.entropy_coefficient(progress) * update_entropy
        )

    def forward(self, node_state, edge_index, edge_attr=None, context=None, *, progress: float = 1.0):
        raw_messages = self.candidate_messages(node_state, edge_index, edge_attr, context)
        messages = self.message_calibrator(raw_messages)
        message_weights = self.message_gate(progress)
        mixed = sum(message_weights[index] * message for index, message in enumerate(messages))
        aggregate = aggregate_messages(mixed, edge_index[1], len(node_state), self.aggregation)
        raw_updates = [kernel(node_state, aggregate, context) for kernel in self.update_kernels]
        updates = self.update_calibrator(raw_updates)
        update_weights = self.update_gate(progress)
        increment = sum(update_weights[index] * update for index, update in enumerate(updates))
        if increment.shape != node_state.shape:
            raise ValueError(
                f"mixed update increment shape {tuple(increment.shape)} must match node state {tuple(node_state.shape)}"
            )
        output = node_state + increment
        diagnostics = {
            "message_weights": dict(zip(self.message_names, message_weights)),
            "update_weights": dict(zip(self.update_names, update_weights)),
            "message_calibration": self.message_calibrator.as_dict(),
            "update_calibration": self.update_calibrator.as_dict(),
            "gate_regularization": self.gate_regularization(progress),
        }
        return output, diagnostics
