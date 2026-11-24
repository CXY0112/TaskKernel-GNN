[中文版](PROJECT_STRUCTURE.zh-CN.md)

# Project structure

## Source code

```text
src/taskkernel_gnn/
├── core/
│   ├── aggregation.py     ADD/MEAN neighborhood aggregation
│   ├── calibration.py     fixed RMS candidate calibration
│   ├── gating.py          annealed temperature softmax
│   └── layer.py           message → aggregation → update layer
├── kernels/
│   ├── message/           message kernels, then grouped by domain
│   │   ├── common.py      differences, graph diffusion, and distance kernels
│   │   ├── transport.py   wave propagation and directed transport
│   │   ├── mechanics.py   gravity, springs, damping, and particle helpers
│   │   ├── cardiac.py     cardiac FEM discretization adapter
│   │   └── neural.py      edge-gated and MLP messages
│   ├── update/            node-update kernels, then grouped by domain
│   │   ├── common.py      identity, linear, and generic Euler updates
│   │   ├── reaction.py    generic local reaction and damping
│   │   ├── mechanics.py   second-order mechanics integration
│   │   ├── wave.py        augmented-state wave update
│   │   ├── cardiac.py     AP, FHN, and cardiac decay
│   │   └── neural.py      GRU and MLP updates
│   ├── base.py            shared tensor and parameter helpers
│   ├── registry.py        named kernel factory for experiment assembly
│   └── README.zh-CN.md    kernel-library structure and interfaces
├── models/
│   ├── taskkernel.py      stackable TaskKernel-GNN
│   └── variants.py        Learned-Single, MLP-only, Physics+Residual
└── data/
    ├── io.py              NPZ and metric-table loading
    ├── cardiac.py         typed tetrahedral cardiac mesh loader
    └── particle.py        particle-system save/load interface
```

`src/` owns reusable
kernels, gates, layers, factories, models, and data interfaces. Experiment
directories contain dataset construction, training loops, and evaluation only.
The mirrored Chinese kernel manual is under `docs/kernel-manual/`.

## Experiments

```text
experiments/
├── paper/                    PDEBench, openCARP, ISO17
├── d1_scalar/                controlled mechanism studies
├── d2_reaction_diffusion/    Ring, Grid, and RGG ablations
├── d3_particle_dynamics/     mixed-force particle studies
└── README.md                 runnable entry points
```

## Outputs and media

Experiment outputs are written locally under `results/`. The three examples in
the repository README are stored in `visualizations/examples/`.
