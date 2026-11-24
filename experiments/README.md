[中文版](README.zh-CN.md)

# Experiments

## Main paper experiments

Prepare external inputs using the [dataset guide](../data/README.md). Run commands from the repository root. The [repository README](../README.md#main-paper-experiments) shows the three qualitative examples.

- **E1 · PDEBench.** [Training](paper/external_train.py), [shared diffusion](paper/configs/external/pdebench.yaml) and [channel-wise diffusion](paper/configs/external/pdebench_channel_diffusion.yaml) configurations, [test evaluation](paper/evaluate_e1_channel_diffusion_test.py), and [plotting](paper/plot_e1_channel_diffusion.py).
- **E2 · openCARP.** Follow the [simulation protocol](../results/SOTA_Baseline/external/opencarp/preparation/PROTOCOL.zh-CN.md) and run the [training, evaluation, and rendering entry point](paper/run_e2_taskkernel.py).
- **E3 · ISO17.** [Training](paper/train_iso17_feature_kernels.py), [full test evaluation of variant C](paper/evaluate_iso17_feature_full.py), and [molecular rollout rendering](paper/visualize_iso17_feature_default.py). Variants A and B are the kernel ablations from §3.5.

E1 example with the channel-wise configuration and seed 0:

```bash
python experiments/paper/external_train.py --config experiments/paper/configs/external/pdebench_channel_diffusion.yaml --output-name taskkernel_channel_diffusion --seed 0 --execute-training
python experiments/paper/evaluate_e1_channel_diffusion_test.py --variant channel-wise --seed 0
```

For shared diffusion, train with the default configuration and output name, then evaluate with `--variant shared`.

E3 example with the main model and seed 0:

```bash
python experiments/paper/train_iso17_feature_kernels.py --variant C --seed 0
python experiments/paper/cache_iso17_full_tests.py
python experiments/paper/cache_iso17_energy_labels.py
python experiments/paper/evaluate_iso17_feature_full.py --seed 0
```

## Controlled ablations (§3.5)

- **D1 · Scalar dynamics:** [single mechanism](d1_scalar/d1_single.py), [mixed mechanisms](d1_scalar/d1_mixture.py), and [missing priors](d1_scalar/d1_missing.py).
- **D2 · Graph reaction–diffusion:** [full model and MLP ablation](d2_reaction_diffusion/d2_fixed_task.py) and [message/update kernel ablations](d2_reaction_diffusion/d2_ablation.py).
- **D3 · Mixed-force particle dynamics:** [TaskKernel-GNN training and rollout evaluation](d3_particle_dynamics/d3_particle_formal.py).

D3 seeds model initialization directly. Initialization in the original comparison depended on the order in which the comparison models ran, so newly trained checkpoints may differ.
