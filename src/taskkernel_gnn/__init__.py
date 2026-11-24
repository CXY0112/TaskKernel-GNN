"""Public API for TaskKernel-GNN."""

from .config import (
    CalibrationConfig,
    GateConfig,
    KernelBankConfig,
    MLPKernelConfig,
    OptimizerConfig,
    TaskKernelConfig,
    default_config,
)
from .core.calibration import FixedRMSCalibrator, rms_scales
from .core.gating import TemperatureSoftmaxGate, annealed_temperature, simplex_entropy, temperature_softmax
from .core.layer import TaskKernelLayer
from .kernels.registry import (
    MESSAGE_KERNELS,
    UPDATE_KERNELS,
    build_kernel_bank,
    create_message_kernel,
    create_update_kernel,
    register_message_kernel,
    register_update_kernel,
)
from .models.taskkernel import TaskKernelGNN

__all__ = [
    "CalibrationConfig",
    "FixedRMSCalibrator",
    "GateConfig",
    "KernelBankConfig",
    "MESSAGE_KERNELS",
    "MLPKernelConfig",
    "OptimizerConfig",
    "TaskKernelGNN",
    "TaskKernelConfig",
    "TaskKernelLayer",
    "TemperatureSoftmaxGate",
    "UPDATE_KERNELS",
    "annealed_temperature",
    "build_kernel_bank",
    "create_message_kernel",
    "create_update_kernel",
    "default_config",
    "rms_scales",
    "register_message_kernel",
    "register_update_kernel",
    "simplex_entropy",
    "temperature_softmax",
]
