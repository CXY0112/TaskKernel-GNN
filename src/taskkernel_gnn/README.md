[中文版](README.zh-CN.md)

# TaskKernel-GNN source package

This directory contains the reusable method implementation. Experiment scripts
under `experiments/` contain only task construction, training schedules, and
evaluation logic.

- `core/`: fixed RMS calibration, temperature-softmax gates, ADD/MEAN
  aggregation, and the three-stage TaskKernel layer;
- `kernels/`: first split into `message/` and `update/`, then grouped by
  common, neural, or physical domain;
- `models/`: the stacked TaskKernel-GNN and controlled model variants;
- `data/`: typed loaders for particle systems, tetrahedral cardiac meshes, and
  result tables.

The public API is exported by `taskkernel_gnn/__init__.py`. Install the package
in editable mode with `python -m pip install -e .`, or add `src/` to
`PYTHONPATH` when running without installation.

## Central hyperparameter configuration

`config.yaml` is the user-editable configuration page. It groups message and
update gates, MLP controls, calibration, kernel banks, and optimizer settings,
with Chinese comments giving each parameter's role, range, and suggested
values. Its defaults reproduce the current general starting point (`ADD`,
`eta=0.1`, message temperature 1.0 to 0.5, update temperature 1.0 to 0.3, and
no entropy penalty):

```python
from taskkernel_gnn import TaskKernelConfig

config = TaskKernelConfig.from_yaml("src/taskkernel_gnn/config.yaml")
```

`config.py` provides the YAML reader and validation schema. Use
`message_gate.temperature(progress)` and
`message_gate.entropy_coefficient(progress)` to obtain the scheduled values at
training progress from zero to one.

Pass the configuration to `TaskKernelLayer(..., config=config)`. During
training, add `layer.gate_regularization(progress)` (or the stacked model's
`model.gate_regularization(progress)`) to the prediction loss. The returned
term already contains the scheduled message and update entropy coefficients.

## Registry-based assembly

`kernels/registry.py` exposes candidates implemented under `kernels/message/`
and `kernels/update/`. Experiments select a small
compatible subset with `TaskKernelGNN.from_specs(...)`; new domain kernels can
be added with `register_message_kernel` or `register_update_kernel` without
changing the core layer. Every message kernel returns one source-to-target edge
message. Every update kernel returns a state increment, and `TaskKernelLayer`
adds the old state exactly once.

The mirrored Chinese formula and API manual is in `docs/kernel-manual/`.
