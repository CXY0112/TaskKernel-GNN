"""Central, serializable hyperparameter configuration for TaskKernel-GNN.

The defaults encode the current general starting point.  Task-specific kernel
names and configuration bounds are intentionally empty until supplied by an
experiment.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _fraction(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")


@dataclass
class GateConfig:
    """Temperature schedule, entropy penalty, and initialization for one gate."""

    start_temperature: float = 1.0
    end_temperature: float = 0.5
    hold_fraction: float = 0.30
    entropy_weight: float = 0.0
    entropy_start_fraction: float = 0.30
    entropy_end_fraction: float = 0.80
    normalize_entropy: bool = True
    entropy_epsilon: float = 1e-8
    initial_logits: tuple[float, ...] = ()

    def validate(self, name: str = "gate") -> None:
        if self.start_temperature <= 0 or self.end_temperature <= 0:
            raise ValueError(f"{name} temperatures must be positive")
        _fraction(f"{name}.hold_fraction", self.hold_fraction)
        _fraction(f"{name}.entropy_start_fraction", self.entropy_start_fraction)
        _fraction(f"{name}.entropy_end_fraction", self.entropy_end_fraction)
        if self.entropy_start_fraction > self.entropy_end_fraction:
            raise ValueError(f"{name} entropy start must not exceed entropy end")
        if self.entropy_weight < 0:
            raise ValueError(f"{name}.entropy_weight must be nonnegative")
        if self.entropy_epsilon <= 0:
            raise ValueError(f"{name}.entropy_epsilon must be positive")

    def temperature(self, progress: float) -> float:
        """Return the linearly annealed temperature at training progress [0, 1]."""
        _fraction("progress", progress)
        if progress <= self.hold_fraction:
            return float(self.start_temperature)
        ratio = (progress - self.hold_fraction) / max(1.0 - self.hold_fraction, 1e-12)
        return float(self.start_temperature + min(ratio, 1.0) * (self.end_temperature - self.start_temperature))

    def entropy_coefficient(self, progress: float) -> float:
        """Return the linearly ramped entropy coefficient at training progress."""
        _fraction("progress", progress)
        if progress <= self.entropy_start_fraction:
            return 0.0
        width = self.entropy_end_fraction - self.entropy_start_fraction
        if width <= 0:
            return float(self.entropy_weight)
        ratio = min((progress - self.entropy_start_fraction) / width, 1.0)
        return float(self.entropy_weight * ratio)

    def gate_kwargs(self) -> dict[str, Any]:
        """Arguments accepted by :class:`TemperatureSoftmaxGate`."""
        return {
            "start_temperature": self.start_temperature,
            "end_temperature": self.end_temperature,
            "hold_fraction": self.hold_fraction,
        }


@dataclass
class MLPKernelConfig:
    """Controls for the mesh-independent MLP completion kernel."""

    eta: float = 0.1
    symmetric: bool = False

    def validate(self) -> None:
        if self.eta < 0:
            raise ValueError("mlp.eta must be nonnegative")


@dataclass
class CalibrationConfig:
    """Fixed candidate-response calibration settings."""

    enabled: bool = True
    method: str = "rms"
    minimum_rms: float = 1e-8
    minimum_scale: float = 0.1
    maximum_scale: float = 10.0
    fixed_after_fit: bool = True
    sample_fractions: tuple[float, ...] = (0.20, 0.40, 0.60, 0.80)

    def validate(self) -> None:
        if self.method != "rms":
            raise ValueError("calibration.method currently supports only 'rms'")
        if self.minimum_rms <= 0:
            raise ValueError("calibration.minimum_rms must be positive")
        if self.minimum_scale <= 0 or self.maximum_scale < self.minimum_scale:
            raise ValueError("calibration scale range is invalid")
        if not self.sample_fractions:
            raise ValueError("calibration.sample_fractions cannot be empty")
        for value in self.sample_fractions:
            _fraction("calibration sample fraction", value)


@dataclass
class KernelBankConfig:
    """Candidate names and optional bounds for learned kernel configurations c."""

    message_names: tuple[str, ...] = ()
    update_names: tuple[str, ...] = ()
    message_specs: tuple[dict[str, Any], ...] = ()
    update_specs: tuple[dict[str, Any], ...] = ()
    positive_configurations: tuple[str, ...] = ()
    configuration_bounds: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)

    def validate(self) -> None:
        for label, names in (("message", self.message_names), ("update", self.update_names)):
            if len(names) != len(set(names)):
                raise ValueError(f"{label} kernel names must be unique")
        known = set(self.message_names) | set(self.update_names)
        for stage, specs in (("message", self.message_specs), ("update", self.update_specs)):
            for spec in specs:
                if not isinstance(spec, dict) or "type" not in spec:
                    raise ValueError(f"each {stage} kernel specification requires a type")
                known.add(spec.get("name", spec["type"]))
        unknown_positive = set(self.positive_configurations) - known
        if unknown_positive:
            raise ValueError(f"positive configurations reference unknown kernels: {sorted(unknown_positive)}")
        for name, bounds in self.configuration_bounds.items():
            if name not in known:
                raise ValueError(f"configuration bound references unknown kernel: {name}")
            if len(bounds) != 2:
                raise ValueError(f"configuration bound for {name} must contain (lower, upper)")
            lower, upper = bounds
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(f"configuration bound for {name} has lower > upper")


@dataclass
class OptimizerConfig:
    """Training controls shared by the core experiments."""

    learning_rate: float = 3e-3
    gate_learning_rate_scale: float = 1.0
    weight_decay: float = 1e-6
    max_gradient_norm: float = 2.0
    training_steps: int = 400
    rollout_window: int = 6
    batch_size: int = 96
    seed: int = 0

    def validate(self) -> None:
        if self.learning_rate <= 0 or self.gate_learning_rate_scale <= 0:
            raise ValueError("optimizer learning rates must be positive")
        if self.weight_decay < 0 or self.max_gradient_norm <= 0:
            raise ValueError("optimizer decay must be nonnegative and gradient norm positive")
        if self.training_steps < 1 or self.rollout_window < 1 or self.batch_size < 1:
            raise ValueError("optimizer step, rollout, and batch counts must be positive")


def _message_gate() -> GateConfig:
    return GateConfig(end_temperature=0.5)


def _update_gate() -> GateConfig:
    return GateConfig(end_temperature=0.3)


@dataclass
class TaskKernelConfig:
    """Complete reusable TaskKernel-GNN hyperparameter configuration."""

    aggregation: str = "add"
    message_gate: GateConfig = field(default_factory=_message_gate)
    update_gate: GateConfig = field(default_factory=_update_gate)
    mlp: MLPKernelConfig = field(default_factory=MLPKernelConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    kernels: KernelBankConfig = field(default_factory=KernelBankConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    def validate(self) -> "TaskKernelConfig":
        if self.aggregation not in {"add", "mean"}:
            raise ValueError("aggregation must be 'add' or 'mean'")
        self.message_gate.validate("message_gate")
        self.update_gate.validate("update_gate")
        self.mlp.validate()
        self.calibration.validate()
        self.kernels.validate()
        self.optimizer.validate()
        message_count = len(self.kernels.message_names) or len(self.kernels.message_specs)
        update_count = len(self.kernels.update_names) or len(self.kernels.update_specs)
        if self.message_gate.initial_logits and message_count:
            if len(self.message_gate.initial_logits) != message_count:
                raise ValueError("message initial logits must match message kernel count")
        if self.update_gate.initial_logits and update_count:
            if len(self.update_gate.initial_logits) != update_count:
                raise ValueError("update initial logits must match update kernel count")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TaskKernelConfig":
        data = dict(values)
        message = dict(data.pop("message_gate", {}))
        update = dict(data.pop("update_gate", {}))
        mlp = dict(data.pop("mlp", {}))
        calibration = dict(data.pop("calibration", {}))
        kernels = dict(data.pop("kernels", {}))
        optimizer = dict(data.pop("optimizer", {}))

        for gate in (message, update):
            if "initial_logits" in gate:
                gate["initial_logits"] = tuple(gate["initial_logits"])
        if "sample_fractions" in calibration:
            calibration["sample_fractions"] = tuple(calibration["sample_fractions"])
        for key in ("message_names", "update_names", "positive_configurations"):
            if key in kernels:
                kernels[key] = tuple(kernels[key])
        for key in ("message_specs", "update_specs"):
            if key in kernels:
                kernels[key] = tuple(dict(spec) for spec in kernels[key])
        if "configuration_bounds" in kernels:
            kernels["configuration_bounds"] = {
                name: tuple(bounds) for name, bounds in kernels["configuration_bounds"].items()
            }

        config = cls(
            message_gate=GateConfig(**message),
            update_gate=GateConfig(**update),
            mlp=MLPKernelConfig(**mlp),
            calibration=CalibrationConfig(**calibration),
            kernels=KernelBankConfig(**kernels),
            optimizer=OptimizerConfig(**optimizer),
            **data,
        )
        return config.validate()

    @classmethod
    def from_json(cls, path: str | Path) -> "TaskKernelConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TaskKernelConfig":
        """Load the editable YAML configuration page."""
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - dependency error path
            raise ImportError("YAML configuration requires PyYAML") from error
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("YAML configuration root must be a mapping")
        return cls.from_dict(values)


def default_config() -> TaskKernelConfig:
    """Return a fresh validated copy of the recommended starting config."""
    return TaskKernelConfig().validate()


__all__ = [
    "CalibrationConfig",
    "GateConfig",
    "KernelBankConfig",
    "MLPKernelConfig",
    "OptimizerConfig",
    "TaskKernelConfig",
    "default_config",
]
