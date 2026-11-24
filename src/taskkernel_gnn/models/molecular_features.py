"""Multi-kernel molecular feature interaction followed by conservative readout."""

import torch
from torch import nn

from ..config import GateConfig, TaskKernelConfig
from ..core.layer import TaskKernelLayer
from ..kernels.message.molecular_energy import (
    MolecularBondEnergyKernel, MolecularAngleEnergyKernel,
    MolecularNonbondedEnergyKernel, MolecularPairMLPEnergyKernel,
    MolecularTripletMLPEnergyKernel,
)
from ..kernels.message.molecular_features import (
    ScalarEnergyFeatureMessage, MolecularContinuousFilterMessage,
)
from ..kernels.update.molecular_features import (
    MolecularFeatureLinearUpdate, MolecularFeatureMLPUpdate,
)


class MolecularFeatureEnergyModel(nn.Module):
    def __init__(self, variant="C", features=32, rounds=2, pair_hidden=64,
                 triplet_hidden=32, mlp_amplitude=0.2, radial_basis=32,
                 filter_hidden=32, update_hidden=64, cutoff=8.0,
                 end_temperature=0.5):
        super().__init__()
        if variant not in ("A", "B", "C") or features < 1 or rounds < 1:
            raise ValueError("variant A/B/C and positive features/rounds required")
        self.variant, self.rounds = variant, rounds
        self.embedding = nn.Embedding(9, features)
        # Construct all banks in the same order so shared modules start identically
        # for the same seed; unused local modules are not registered or counted.
        descriptors = [MolecularBondEnergyKernel(), MolecularAngleEnergyKernel(),
                       MolecularNonbondedEnergyKernel(),
                       MolecularPairMLPEnergyKernel(pair_hidden, mlp_amplitude),
                       MolecularTripletMLPEnergyKernel(triplet_hidden, mlp_amplitude)]
        lifted = [ScalarEnergyFeatureMessage(kernel, features) for kernel in descriptors]
        names = ["bond_feature", "angle_feature", "nonbonded_feature",
                 "pair_mlp_feature", "triplet_mlp_feature"]
        continuous = MolecularContinuousFilterMessage(features, radial_basis, filter_hidden, cutoff)
        updates = [MolecularFeatureLinearUpdate(features),
                   MolecularFeatureMLPUpdate(features, update_hidden)]
        self.readout = nn.Sequential(nn.Linear(features, features), nn.SiLU(), nn.Linear(features, 1))
        if variant == "A":
            messages = lifted
        elif variant == "B":
            messages, names = [continuous], ["continuous_filter"]
        else:
            messages, names = lifted + [continuous], names + ["continuous_filter"]
        gate = GateConfig(start_temperature=1.0, end_temperature=end_temperature, hold_fraction=0.30)
        self.layer = TaskKernelLayer(messages, updates, message_names=names,
                                     update_names=["linear_feature", "mlp_feature"],
                                     config=TaskKernelConfig(aggregation="add", message_gate=gate,
                                                             update_gate=gate))
        self.energy_bias = nn.Parameter(torch.zeros(()))

    def energy(self, edge_index, context, progress=1.0):
        state = self.embedding(context["atomic_numbers"])
        for _ in range(self.rounds):
            state, info = self.layer(state, edge_index, context=context, progress=progress)
        atomic_energy = self.readout(state).squeeze(-1)
        molecule = context["molecule_index"]
        count = int(molecule.max()) + 1
        energy = atomic_energy.new_zeros(count).index_add_(0, molecule, atomic_energy)
        return energy + self.energy_bias, info

    def forward(self, edge_index, context, *, progress=1.0, create_graph=False):
        position = context["position"]
        if not position.requires_grad:
            raise ValueError("position must require gradients")
        energy, info = self.energy(edge_index, context, progress)
        force = -torch.autograd.grad(energy.sum(), position, create_graph=create_graph)[0]
        return energy, force.reshape(len(energy), -1, 3), info
