"""External neural baseline architectures used by comparison experiments."""

from .graph_pde import GraphNeuralOperator, MeshGraphNet, MessagePassingPDESolver, build_graph_pde_baseline
from .cardiac import CardiacStimulusAdapter, build_cardiac_baseline
from .gino import GeometryInformedNeuralOperator
from .particles import EGNNParticleSimulator, GraphNetworkSimulator, ParticleMeshGraphNet
from .registry import build_external_baseline
from .structured import FourierNeuralOperator2d, UNet2d

__all__ = [
    "GraphNeuralOperator",
    "MeshGraphNet",
    "MessagePassingPDESolver",
    "build_graph_pde_baseline",
    "CardiacStimulusAdapter",
    "build_cardiac_baseline",
    "build_external_baseline",
    "FourierNeuralOperator2d",
    "UNet2d",
    "GeometryInformedNeuralOperator",
    "GraphNetworkSimulator",
    "EGNNParticleSimulator",
    "ParticleMeshGraphNet",
]
