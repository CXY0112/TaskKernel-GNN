"""Dataset-specific step interfaces assembled from the reusable TaskKernel core."""

from __future__ import annotations

import torch
from torch import nn

from ..config import TaskKernelConfig
from ..core.aggregation import aggregate_messages
from ..data.graph import GraphSample
from ..kernels.message.fluids import compact_density
from ..kernels.update.fluids import SoftBoxWallAcceleration, UniformGravityAcceleration
from .taskkernel import TaskKernelGNN


def graph_edge_features(graph: GraphSample) -> torch.Tensor:
    relative = graph.coordinates[graph.source] - graph.coordinates[graph.target]
    return torch.cat((relative, graph.distance[:, None], graph.weight[:, None]), -1)


def _fit_analytic_calibrator(calibrator, responses: dict[str, torch.Tensor]) -> None:
    """Calibrate analytic candidates while leaving bounded MLP residuals at eta scale."""
    calibrator.fit(responses)
    for index, name in enumerate(calibrator.names):
        if "mlp" in name.lower():
            calibrator.scales[index] = 1.0


class GraphTaskKernelStepper(nn.Module):
    def __init__(self, model: TaskKernelGNN) -> None:
        super().__init__()
        self.model = model

    def step(self, state: torch.Tensor, graph: GraphSample, time_index: int | None = None,
             *, progress: float = 1.0, substeps: int = 1) -> torch.Tensor:
        if substeps < 1:
            raise ValueError("substeps must be positive")
        context = dict(graph.context)
        context.setdefault("position", graph.coordinates)
        context.setdefault("edge_weight", graph.weight)
        edge_features = graph_edge_features(graph)
        current = state
        for _ in range(substeps):
            full_step, _ = self.model(
                current, graph.edge_index, edge_features, context, progress=progress
            )
            current = current + (full_step - current) / substeps
        return current

    def gate_regularization(self, progress: float = 1.0):
        return self.model.gate_regularization(progress)

    @torch.no_grad()
    def fit_calibration(self, examples: list[tuple[torch.Tensor, GraphSample]]) -> dict[str, dict[str, float]]:
        """Fit fixed message/update RMS multipliers from training-only examples."""
        if len(self.model.layers) != 1:
            raise ValueError("external calibration currently expects a one-layer TaskKernel model")
        layer = self.model.layers[0]
        message_groups = [[] for _ in layer.message_kernels]
        for state, graph in examples:
            context = dict(graph.context)
            context.setdefault("position", graph.coordinates)
            context.setdefault("edge_weight", graph.weight)
            raw_messages = layer.candidate_messages(
                state, graph.edge_index, graph_edge_features(graph), context
            )
            for group, value in zip(message_groups, raw_messages):
                group.append(value)
        _fit_analytic_calibrator(layer.message_calibrator, {
            name: torch.cat(values) for name, values in zip(layer.message_names, message_groups)
        })
        update_groups = [[] for _ in layer.update_kernels]
        for state, graph in examples:
            context = dict(graph.context)
            context.setdefault("position", graph.coordinates)
            context.setdefault("edge_weight", graph.weight)
            raw_messages = layer.candidate_messages(
                state, graph.edge_index, graph_edge_features(graph), context
            )
            calibrated = layer.message_calibrator(raw_messages)
            weights = layer.message_gate(0.0)
            mixed = sum(weights[index] * value for index, value in enumerate(calibrated))
            aggregate = aggregate_messages(mixed, graph.target, len(state), layer.aggregation)
            raw_updates = [kernel(state, aggregate, context) for kernel in layer.update_kernels]
            for group, value in zip(update_groups, raw_updates):
                group.append(value)
        _fit_analytic_calibrator(layer.update_calibrator, {
            name: torch.cat(values) for name, values in zip(layer.update_names, update_groups)
        })
        return {
            "message": layer.message_calibrator.as_dict(),
            "update": layer.update_calibrator.as_dict(),
        }


class ParticleTaskKernelStepper(nn.Module):
    def __init__(self, model: TaskKernelGNN) -> None:
        super().__init__()
        self.model = model
        # Uniform acceleration is a node-side physical configuration, estimated
        # from training trajectories and then fine-tuned with the model.
        self.external_acceleration = nn.Parameter(torch.zeros(2))

    @torch.no_grad()
    def initialize_external_acceleration(self, value: torch.Tensor) -> None:
        if value.shape != self.external_acceleration.shape:
            raise ValueError("external acceleration has the wrong dimension")
        self.external_acceleration.copy_(value.to(self.external_acceleration))

    def predict_next(self, history: torch.Tensor, particle_type: torch.Tensor, graph: GraphSample,
                     *, progress: float = 1.0) -> torch.Tensor:
        state, edge_attr, context = self._inputs(history, particle_type, graph)
        result, _ = self.model(state, graph.edge_index, edge_attr, context, progress=progress)
        dt = float(graph.context.get("dt", 1.0))
        return (result[:, : history.shape[-1]] + dt * state[:, history.shape[-1]:]
                + dt * dt * self.external_acceleration)

    @staticmethod
    def _inputs(history: torch.Tensor, particle_type: torch.Tensor, graph: GraphSample):
        position = history[-1]
        dt = float(graph.context.get("dt", 1.0))
        velocity = (history[-1] - history[-2]) / dt
        state = torch.cat((position, velocity), -1)
        relative = position[graph.source] - position[graph.target]
        edge_attr = torch.cat((relative, graph.distance[:, None]), -1)
        context = dict(graph.context)
        context.update({
            "position": position,
            "velocity": velocity,
            "mass": torch.ones(len(position), device=position.device, dtype=position.dtype),
            "spring_k": torch.ones(len(graph.source), device=position.device, dtype=position.dtype),
            "rest_length": torch.full_like(graph.distance, 0.5 * max(float(graph.distance.detach().mean()), 1e-6)),
            "dashpot_gamma": torch.as_tensor(1.0, device=position.device, dtype=position.dtype),
            "particle_type": particle_type,
        })
        context.setdefault("gravity_g", torch.as_tensor(1.0e-6, device=position.device, dtype=position.dtype))
        return state, edge_attr, context

    def gate_regularization(self, progress: float = 1.0):
        return self.model.gate_regularization(progress)

    @torch.no_grad()
    def fit_calibration(
        self,
        examples: list[tuple[torch.Tensor, torch.Tensor, GraphSample]],
    ) -> dict[str, dict[str, float]]:
        if len(self.model.layers) != 1:
            raise ValueError("external calibration currently expects a one-layer TaskKernel model")
        layer = self.model.layers[0]
        message_groups = [[] for _ in layer.message_kernels]
        for history, particle_type, graph in examples:
            state, edge_attr, context = self._inputs(history, particle_type, graph)
            raw_messages = layer.candidate_messages(state, graph.edge_index, edge_attr, context)
            for group, value in zip(message_groups, raw_messages):
                group.append(value)
        _fit_analytic_calibrator(layer.message_calibrator, {
            name: torch.cat(values) for name, values in zip(layer.message_names, message_groups)
        })
        update_groups = [[] for _ in layer.update_kernels]
        for history, particle_type, graph in examples:
            state, edge_attr, context = self._inputs(history, particle_type, graph)
            raw_messages = layer.candidate_messages(state, graph.edge_index, edge_attr, context)
            calibrated = layer.message_calibrator(raw_messages)
            weights = layer.message_gate(0.0)
            mixed = sum(weights[index] * value for index, value in enumerate(calibrated))
            aggregate = aggregate_messages(mixed, graph.target, len(state), layer.aggregation)
            raw_updates = [kernel(state, aggregate, context) for kernel in layer.update_kernels]
            for group, value in zip(update_groups, raw_updates):
                group.append(value)
        _fit_analytic_calibrator(layer.update_calibrator, {
            name: torch.cat(values) for name, values in zip(layer.update_names, update_groups)
        })
        return {
            "message": layer.message_calibrator.as_dict(),
            "update": layer.update_calibrator.as_dict(),
        }


class ParticleKinematicResidual(nn.Module):
    """Remove the common kinematic displacement before update-kernel mixing."""

    def __init__(self, kernel: nn.Module, spatial_dims: int = 2) -> None:
        super().__init__()
        self.kernel = kernel
        self.spatial_dims = spatial_dims

    def forward(self, node_state, aggregate, context=None):
        increment = self.kernel(node_state, aggregate, context)
        dt = float(context.get("dt", 1.0))
        dimensions = self.spatial_dims
        kinematic = torch.cat((dt * node_state[:, dimensions:2 * dimensions],
                               torch.zeros_like(node_state[:, dimensions:2 * dimensions])), -1)
        return increment - kinematic


class FluidParticleTaskKernelStepper(ParticleTaskKernelStepper):
    """Two-pass fluid messages with common gravity and wall accelerations."""

    def __init__(self, model: TaskKernelGNN, *, density_radius: float = 0.015,
                 wall_options: dict | None = None, gravity_trainable: bool = True) -> None:
        super().__init__(model)
        del self.external_acceleration
        self.density_radius = float(density_radius)
        self.gravity = UniformGravityAcceleration(dimensions=2)
        self.gravity.acceleration.requires_grad_(gravity_trainable)
        self.wall = SoftBoxWallAcceleration(dimensions=2, **(wall_options or {}))

    @torch.no_grad()
    def initialize_external_acceleration(self, value: torch.Tensor) -> None:
        if value.shape != self.gravity.acceleration.shape:
            raise ValueError("external acceleration has the wrong dimension")
        self.gravity.acceleration.copy_(value.to(self.gravity.acceleration))

    def _inputs(self, history: torch.Tensor, particle_type: torch.Tensor, graph: GraphSample):
        state, edge_attr, context = super()._inputs(history, particle_type, graph)
        position = state[:, :2]
        velocity = state[:, 2:4]
        bounds = torch.as_tensor(context["bounds"], device=position.device, dtype=position.dtype)
        lower_distance, upper_distance = self.wall.wall_distances(position, bounds)
        context.update({
            "density": compact_density(position, graph.edge_index, self.density_radius),
            "wall_features": torch.cat((lower_distance, upper_distance), dim=-1)
                             .div(self.density_radius).clamp(-1.0, 1.0),
            "gravity_acceleration": self.gravity(position),
            "wall_acceleration": self.wall(position, velocity, bounds),
        })
        return state, edge_attr, context

    def predict_next(self, history: torch.Tensor, particle_type: torch.Tensor, graph: GraphSample,
                     *, progress: float = 1.0) -> torch.Tensor:
        state, edge_attr, context = self._inputs(history, particle_type, graph)
        result, _ = self.model(state, graph.edge_index, edge_attr, context, progress=progress)
        return result[:, : history.shape[-1]]


def build_external_taskkernel(dataset: str, config: TaskKernelConfig,
                              *, fluid_options: dict | None = None) -> nn.Module:
    model = TaskKernelGNN.from_config(config)
    if dataset in {"pdebench", "opencarp"}:
        return GraphTaskKernelStepper(model)
    if dataset == "waterdrop":
        if any(spec.get("type", "") == "fluid_semi_implicit"
               for spec in config.kernels.update_specs):
            options = fluid_options or {}
            return FluidParticleTaskKernelStepper(
                model, density_radius=float(options.get("density_radius", 0.015)),
                wall_options=dict(options.get("wall", {})),
                gravity_trainable=bool(options.get("gravity_trainable", True)),
            )
        layer = model.layers[0]
        layer.update_kernels = nn.ModuleList(
            ParticleKinematicResidual(kernel) for kernel in layer.update_kernels
        )
        return ParticleTaskKernelStepper(model)
    raise ValueError(f"unknown external dataset: {dataset}")
