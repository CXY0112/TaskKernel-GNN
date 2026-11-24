[English](README.md)

# 实验

## 论文主实验

按[数据准备说明](../data/README.md)获取外部输入，并在仓库根目录运行命令。[仓库首页](../README.zh-CN.md#论文主实验)展示三个主实验的图例。

- **E1 · PDEBench**。[训练](paper/external_train.py)、[共享扩散配置](paper/configs/external/pdebench.yaml)、[分通道扩散配置](paper/configs/external/pdebench_channel_diffusion.yaml)、[测试集评估](paper/evaluate_e1_channel_diffusion_test.py)与[示例绘图](paper/plot_e1_channel_diffusion.py)。
- **E2 · openCARP**。按[仿真协议](../results/SOTA_Baseline/external/opencarp/preparation/PROTOCOL.zh-CN.md)准备数据，再运行[训练、评估和绘图入口](paper/run_e2_taskkernel.py)。
- **E3 · ISO17**。[训练](paper/train_iso17_feature_kernels.py)、[组合模型 C 的完整测试集评估](paper/evaluate_iso17_feature_full.py)与[分子轨迹绘图](paper/visualize_iso17_feature_default.py)。A 和 B 是论文第 3.5 节的核配置消融。

E1 分通道扩散配置的 seed 0 示例：

```bash
python experiments/paper/external_train.py --config experiments/paper/configs/external/pdebench_channel_diffusion.yaml --output-name taskkernel_channel_diffusion --seed 0 --execute-training
python experiments/paper/evaluate_e1_channel_diffusion_test.py --variant channel-wise --seed 0
```

运行共享扩散配置时，训练使用默认配置和输出名，评估使用 `--variant shared`。

E3 组合模型 C 的 seed 0 示例：

```bash
python experiments/paper/train_iso17_feature_kernels.py --variant C --seed 0
python experiments/paper/cache_iso17_full_tests.py
python experiments/paper/cache_iso17_energy_labels.py
python experiments/paper/evaluate_iso17_feature_full.py --seed 0
```

## 受控消融（第 3.5 节）

- **D1 · 标量动力学**。[单一机制](d1_scalar/d1_single.py)、[混合机制](d1_scalar/d1_mixture.py)与[先验缺失](d1_scalar/d1_missing.py)。
- **D2 · 图上的反应–扩散**。[完整模型与 MLP 消融](d2_reaction_diffusion/d2_fixed_task.py)，以及[消息核与更新核消融](d2_reaction_diffusion/d2_ablation.py)。
- **D3 · 混合力粒子动力学**。[TaskKernel-GNN 的训练与轨迹推进评估](d3_particle_dynamics/d3_particle_formal.py)。

D3 现在直接为模型初始化设置随机种子。原对比实验的初始化受各模型运行顺序影响，因此重新训练得到的权重可能与原实验不同。
