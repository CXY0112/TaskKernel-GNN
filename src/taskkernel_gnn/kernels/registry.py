"""Named kernel registry and small factory used by experiment assembly code."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import nn

from .message import (
    AdapterDiffusionKernel, DifferenceMessageKernel, DirectedTransportKernel, EdgeGatedMessageKernel,
    ExponentialDiffusionKernel, GaussianDiffusionKernel, GraphDiffusionKernel,
    InversePowerDistanceKernel, MLPMessageKernel, ViscousDampingKernel,
    PairwiseDashpotKernel, ShortRangeRepulsionKernel, SoftenedGravityKernel, SpringKernel, WavePropagationKernel,
    DensityPressureKernel, FluidViscosityKernel, FluidResidualMessageKernel,
    MolecularAngleEnergyKernel, MolecularBondEnergyKernel, MolecularNonbondedEnergyKernel,
    MolecularPairMLPEnergyKernel, MolecularTripletMLPEnergyKernel,
    MolecularAngleBendKernel, MolecularBondStretchKernel,
    MolecularEquivariantMLPKernel, MolecularNonbondedKernel,
    MolecularContinuousFilterMessage, ScalarEnergyFeatureMessage,
    ConditionedDiffusionMessage, GaussianAdapterDiffusionMessage,
)
from .update import (
    AlievPanfilovUpdate, CardiacDecayUpdate, FitzHughNagumoUpdate,
    DampedUpdate, DynamicsEulerUpdate, ExplicitEulerUpdate, GRUUpdate,
    IdentityMessageUpdate, LinearReactionUpdate, LinearResidualUpdate,
    LogisticReactionUpdate, MLPUpdate, ParticleResidualMLPUpdate, PolynomialReactionUpdate,
    CoupledPolynomialReactionUpdate, SaturatingReactionUpdate,
    BoundaryContactUpdate, SemiImplicitEulerUpdate, StateMessageLinearUpdate, WaveStateUpdate,
    FluidSemiImplicitUpdate, FluidResidualAccelerationUpdate,
    MolecularSaturatingEnergyUpdate,
    MolecularSaturatingForceUpdate,
    MolecularFeatureLinearUpdate, MolecularFeatureMLPUpdate,
    ConditionedMLPUpdate,
)


MESSAGE_KERNELS: dict[str, type[nn.Module]] = {
    "conditioned_diffusion_mlp": ConditionedDiffusionMessage,
    "gaussian_adapter_diffusion": GaussianAdapterDiffusionMessage,
    "molecular_continuous_filter": MolecularContinuousFilterMessage,
    "scalar_energy_feature": ScalarEnergyFeatureMessage,
    "difference": DifferenceMessageKernel,
    "graph_diffusion": GraphDiffusionKernel,
    "gaussian_diffusion": GaussianDiffusionKernel,
    "exponential_diffusion": ExponentialDiffusionKernel,
    "inverse_power": InversePowerDistanceKernel,
    "wave": WavePropagationKernel,
    "directed_transport": DirectedTransportKernel,
    "edge_gated": EdgeGatedMessageKernel,
    "mlp": MLPMessageKernel,
    "gravity": SoftenedGravityKernel,
    "spring": SpringKernel,
    "dashpot": PairwiseDashpotKernel,
    "short_range_repulsion": ShortRangeRepulsionKernel,
    "viscous_damping": ViscousDampingKernel,
    "adapter_diffusion": AdapterDiffusionKernel,
    "density_pressure": DensityPressureKernel,
    "fluid_viscosity": FluidViscosityKernel,
    "fluid_residual_mlp": FluidResidualMessageKernel,
    "molecular_bond_energy": MolecularBondEnergyKernel,
    "molecular_angle_energy": MolecularAngleEnergyKernel,
    "molecular_nonbonded_energy": MolecularNonbondedEnergyKernel,
    "molecular_pair_mlp_energy": MolecularPairMLPEnergyKernel,
    "molecular_triplet_mlp_energy": MolecularTripletMLPEnergyKernel,
    "molecular_bond_stretch": MolecularBondStretchKernel,
    "molecular_angle_bend": MolecularAngleBendKernel,
    "molecular_nonbonded_force": MolecularNonbondedKernel,
    "molecular_equivariant_mlp_force": MolecularEquivariantMLPKernel,
}

UPDATE_KERNELS: dict[str, type[nn.Module]] = {
    "conditioned_mlp": ConditionedMLPUpdate,
    "molecular_feature_linear": MolecularFeatureLinearUpdate,
    "molecular_feature_mlp": MolecularFeatureMLPUpdate,
    "identity_message": IdentityMessageUpdate,
    "linear_residual": LinearResidualUpdate,
    "state_message_linear": StateMessageLinearUpdate,
    "euler": DynamicsEulerUpdate,
    "linear_reaction": LinearReactionUpdate,
    "polynomial_reaction": PolynomialReactionUpdate,
    "coupled_polynomial_reaction": CoupledPolynomialReactionUpdate,
    "saturating_reaction": SaturatingReactionUpdate,
    "logistic_reaction": LogisticReactionUpdate,
    "damped": DampedUpdate,
    "second_order_explicit": ExplicitEulerUpdate,
    "second_order_semi_implicit": SemiImplicitEulerUpdate,
    "boundary_contact": BoundaryContactUpdate,
    "wave_state": WaveStateUpdate,
    "gru": GRUUpdate,
    "mlp": MLPUpdate,
    "particle_residual_mlp": ParticleResidualMLPUpdate,
    "aliev_panfilov": AlievPanfilovUpdate,
    "fitzhugh_nagumo": FitzHughNagumoUpdate,
    "cardiac_decay": CardiacDecayUpdate,
    "fluid_semi_implicit": FluidSemiImplicitUpdate,
    "fluid_residual_acceleration": FluidResidualAccelerationUpdate,
    "molecular_saturating_energy": MolecularSaturatingEnergyUpdate,
    "molecular_saturating_force": MolecularSaturatingForceUpdate,
}


def register_message_kernel(name: str, kernel: type[nn.Module], *, replace: bool = False) -> None:
    if name in MESSAGE_KERNELS and not replace:
        raise KeyError(f"message kernel already registered: {name}")
    MESSAGE_KERNELS[name] = kernel


def register_update_kernel(name: str, kernel: type[nn.Module], *, replace: bool = False) -> None:
    if name in UPDATE_KERNELS and not replace:
        raise KeyError(f"update kernel already registered: {name}")
    UPDATE_KERNELS[name] = kernel


def create_message_kernel(kind: str, **parameters: Any) -> nn.Module:
    try: kernel = MESSAGE_KERNELS[kind]
    except KeyError as error: raise KeyError(f"unknown message kernel {kind!r}; available={sorted(MESSAGE_KERNELS)}") from error
    return kernel(**parameters)


def create_update_kernel(kind: str, **parameters: Any) -> nn.Module:
    try: kernel = UPDATE_KERNELS[kind]
    except KeyError as error: raise KeyError(f"unknown update kernel {kind!r}; available={sorted(UPDATE_KERNELS)}") from error
    return kernel(**parameters)


def build_kernel_bank(specifications: Sequence[str | Mapping[str, Any]], *, stage: str) -> tuple[list[nn.Module], list[str]]:
    """Build kernels from names or ``{type, name, parameters}`` mappings."""
    factory = create_message_kernel if stage == "message" else create_update_kernel if stage == "update" else None
    if factory is None: raise ValueError("stage must be 'message' or 'update'")
    modules, names = [], []
    for spec in specifications:
        if isinstance(spec, str): kind, name, parameters = spec, spec, {}
        else:
            values = dict(spec); kind = values.pop("type")
            name = values.pop("name", kind); parameters = values.pop("parameters", {})
            if values: raise ValueError(f"unknown kernel specification fields: {sorted(values)}")
        modules.append(factory(kind, **dict(parameters))); names.append(name)
    if len(names) != len(set(names)): raise ValueError("kernel instance names must be unique")
    return modules, names
