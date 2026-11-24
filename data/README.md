# 数据准备 / Dataset setup

D1, D2, and D3 generate synthetic data in their experiment scripts. Prepare the following inputs for the three main paper experiments. Run the commands from the repository root.

- **PDEBench.** Download `2D_diff-react_NA_NA.h5` to `data/external/pdebench/2d_diffusion_reaction/`. Follow the [verification and split instructions](external/pdebench/2d_diffusion_reaction/README.zh-CN.md) and use the file URL and checksum in the [manifest](external/pdebench/2d_diffusion_reaction/manifest.json).
- **openCARP.** Obtain the official input at the commit and path in the [source manifest](external/opencarp/manifest.json), under `data/external/opencarp/experiments/`. Set up the solver using the [runtime package manifest](../tools/downloads/E2_RUNTIME_PACKAGES.json). Run `python tools/prepare_opencarp_run_plan.py` and `python tools/prepare_e2_protocol.py`. See the [E2 protocol](../results/SOTA_Baseline/external/opencarp/preparation/PROTOCOL.zh-CN.md) for the frozen simulation settings.
- **ISO17.** Download `iso17.tar.gz` from the [dataset provider](https://www.quantum-machine.org/datasets/), extract the ASE databases and ID files to `data/external/iso17/raw/`, and install `python -m pip install -e ".[experiments,iso17]"`. Follow the [E3 experiment instructions](../experiments/README.md#main-paper-experiments) to build the local test caches before evaluation.
