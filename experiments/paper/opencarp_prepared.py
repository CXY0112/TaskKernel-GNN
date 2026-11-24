"""Shared history/forcing interface for E2; independent of older scalar trainer."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from taskkernel_gnn.config import TaskKernelConfig, GateConfig
from taskkernel_gnn.core.layer import TaskKernelLayer
from taskkernel_gnn.data.cardiac_fem import conditioned_history
from taskkernel_gnn.kernels.message.cardiac import AdapterDiffusionKernel
from taskkernel_gnn.kernels.message.conditioned import ConditionedDiffusionMessage, GaussianAdapterDiffusionMessage
from taskkernel_gnn.kernels.update.reaction import LinearReactionUpdate, PolynomialReactionUpdate
from taskkernel_gnn.kernels.update.conditioned import ConditionedMLPUpdate


def initial_history(trajectory, frame=0, frames=8, offset=85., scale=100.):
    indices = torch.arange(frame - frames + 1, frame + 1).clamp_min(0)
    return ((trajectory.voltage[indices].transpose(0, 1) + offset) / scale)


class E2Predictor(nn.Module):
    def __init__(self, name, config):
        super().__init__()
        if name != "taskkernel":
            raise ValueError("This experiment only supports TaskKernel-GNN")
        self.config = config
        self.history_frames = int(config["dataset"]["history_frames"])
        self.voltage_offset = float(config["dataset"]["voltage_offset_mv"])
        self.voltage_scale = float(config["dataset"]["voltage_scale_mv"])
        if self.history_frames < 1 or self.voltage_scale <= 0:
            raise ValueError("positive history_frames and voltage_scale required")
        features = self.history_frames + 8
        p = config["taskkernel"]
        gate = lambda: GateConfig(start_temperature=p["temperature"], end_temperature=p["temperature"],
                                  entropy_weight=p["entropy_weight"])
        core = TaskKernelConfig(aggregation=p["aggregation"], message_gate=gate(), update_gate=gate())
        messages = [AdapterDiffusionKernel("edge_weight"), GaussianAdapterDiffusionMessage(),
                    ConditionedDiffusionMessage(features, p["hidden"], p["eta"], p["symmetric"])]
        updates = [LinearReactionUpdate(rate=-.1), PolynomialReactionUpdate([0., .1, -.1]),
                   ConditionedMLPUpdate(features, p["hidden"], p["eta"])]
        self.model = TaskKernelLayer(messages, updates,
                                    message_names=["anisotropic_fem", "gaussian_fem", "continuous_mlp"],
                                    update_names=["linear", "polynomial", "history_mlp"], config=core)

    def internal_steps(self, graph, interval_ms):
        row_bound = graph.weight.new_zeros(graph.nodes)
        row_bound.index_add_(0, graph.target, graph.weight.abs())
        layer = self.model
        scales = layer.message_calibrator.scales.detach()
        amplitudes = torch.stack([layer.message_kernels[0].log_scale.exp().detach(),
                                  layer.message_kernels[1].log_scale.exp().detach(),
                                  row_bound.new_tensor(self.config["taskkernel"]["eta"])])
        # Includes the update calibration factors. This is a diffusion-oriented
        # step-size bound, not a guarantee on the nonlinear ionic closure.
        strength = (layer.message_gate().detach() * scales * amplitudes).sum()
        update_strength = layer.update_calibrator.scales.detach().max()
        steps = max(1, math.ceil(float(row_bound.max() * strength * update_strength) * interval_ms /
                                 self.config["taskkernel"]["internal_step_safety"]))
        if steps > self.config["taskkernel"]["max_internal_steps"]:
            raise RuntimeError(f"E2 needs {steps} substeps; configured cap exceeded")
        return steps

    def step(self, history, graph, time_ms, interval_ms=1.):
        current = history[:, -1:]
        steps = self.internal_steps(graph, interval_ms)
        dt = interval_ms / steps
        for index in range(steps):
            time = time_ms + index * dt
            def advance(state, time=time):
                causal = torch.cat([history[:, :-1], state], -1)
                context = dict(graph.context, node_features=conditioned_history(causal, graph, time, dt))
                output, _ = self.model(state, graph.edge_index, context=context)
                return state + dt * (output - state)
            if torch.is_grad_enabled() and self.config["taskkernel"]["checkpoint_substeps"]:
                current = checkpoint(advance, current, use_reentrant=False)
            else:
                current = advance(current)
        return torch.cat([history[:, 1:], current], -1)

    @torch.no_grad()
    def fit_calibration(self, samples):
        if not self.config["taskkernel"]["calibration"]:
            return
        message_energy = torch.zeros(3)
        count = 0
        for trajectory, graph in samples:
            for fraction in (.2, .4, .6, .8):
                frame = int(fraction * (len(trajectory.times) - 1))
                history = initial_history(trajectory, frame, self.history_frames, self.voltage_offset, self.voltage_scale)
                context = dict(graph.context, node_features=conditioned_history(history, graph, float(trajectory.times[frame])))
                candidates = self.model.candidate_messages(history[:, -1:], graph.edge_index, context=context)
                message_energy += torch.stack([x.square().mean() for x in candidates]).cpu()
                count += 1
        rms = (message_energy / count).sqrt().clamp_min(1e-8)
        scales = (rms[0] / rms).clamp(.1, 10.)
        scales[2] = 1. # MLP eta remains explicit, not cancelled by calibration.
        self.model.message_calibrator.scales.copy_(scales)
        self.model.message_calibrator.response_rms.copy_(rms)
        # Updates share the same aggregate; calibrate only their local reaction.
        update_energy = torch.zeros(3)
        count = 0
        for trajectory, graph in samples:
            for fraction in (.2, .4, .6, .8):
                frame = int(fraction * (len(trajectory.times) - 1))
                history = initial_history(trajectory, frame, self.history_frames, self.voltage_offset, self.voltage_scale)
                context = dict(graph.context, node_features=conditioned_history(history, graph, float(trajectory.times[frame])))
                zero = torch.zeros_like(history[:, -1:])
                candidates = [kernel(history[:, -1:], zero, context) for kernel in self.model.update_kernels]
                update_energy += torch.stack([x.square().mean() for x in candidates]).cpu()
                count += 1
        rms = (update_energy / count).sqrt().clamp_min(1e-8)
        # Keep update transport identical across candidates. These legacy
        # kernels include aggregate internally; scaling whole outputs would
        # also change diffusion. Record their RMS but retain unit multipliers.
        self.model.update_calibrator.response_rms.copy_(rms)
        self.model.update_calibrator.scales.fill_(1.)
