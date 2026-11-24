[English](README.md)

# TaskKernel-GNN 源码包

本目录包含可复用的方法实现。`experiments/` 只保存任务构造、训练日程与评估逻辑。

- `core/`：固定 RMS 校准、温度 softmax、ADD/MEAN 聚合以及三阶段 TaskKernel 层；
- `kernels/`：第一层划分 `message/` 与 `update/`，第二层再按通用、神经或物理领域划分；
- `models/`：堆叠式 TaskKernel-GNN 和受控模型变体；
- `data/`：粒子系统、四面体心脏网格和结果表的类型化加载接口。

公开 API 由 `taskkernel_gnn/__init__.py` 导出。可以运行 `python -m pip install -e .` 进行可编辑安装；不安装时，也可以把 `src/` 加入 `PYTHONPATH`。

## 统一超参数配置

面向用户编辑的配置页是 `config.yaml`。文件按照消息门控、节点门控、MLP、尺度校准、核库和优化器分组，每个参数都带有中文作用说明、取值范围和建议值。默认配置对应当前通用起点：`ADD`、`eta=0.1`、消息温度从 1.0 降至 0.5、节点温度从 1.0 降至 0.3，并且暂不启用熵惩罚。

```python
from taskkernel_gnn import TaskKernelConfig

config = TaskKernelConfig.from_yaml("src/taskkernel_gnn/config.yaml")
```

`config.py` 负责读取和校验 YAML。训练过程中，调用 `message_gate.temperature(progress)` 和 `message_gate.entropy_coefficient(progress)`，即可获得进度从 0 到 1 时对应的温度与熵惩罚系数。

构造核心层时传入 `TaskKernelLayer(..., config=config)`。训练时，将 `layer.gate_regularization(progress)`（多层模型则使用 `model.gate_regularization(progress)`）加入预测损失；返回值已经包含按训练进度调度后的消息门控和节点门控熵系数。

## 核注册表与实验组装

`kernels/registry.py` 统一暴露 `kernels/message/` 与 `kernels/update/` 下的实现。当前消息核覆盖差分、图扩散、Gaussian/Exponential 扩散、逆幂距离、波传播、有向输运、边属性门控、全相对速度阻尼、引力、弹簧、径向阻尼、FEM适配器和MLP；节点核覆盖恒等消息、线性残差、状态—消息线性、通用Euler、线性/多项式/Logistic反应、阻尼、显式/半隐式二阶力学、波动状态、GRU、MLP、Aliev–Panfilov、FitzHugh–Nagumo和心脏衰减。每个模块的公式、形参与调用方式见 `docs/kernel-manual/`。

实验可以只声明任务需要的小型核库：

```python
from taskkernel_gnn import TaskKernelConfig, TaskKernelGNN

config = TaskKernelConfig.from_yaml("src/taskkernel_gnn/config.yaml")
model = TaskKernelGNN.from_specs(
    message_specs=[
        {"type": "graph_diffusion", "parameters": {"diffusion": 0.1}},
        {"type": "gaussian_diffusion", "parameters": {"diffusion": 0.1, "beta": 2.0}},
        {"type": "mlp", "parameters": {
            "node_features": 1, "edge_features": 2, "output_features": 1,
        }},
    ],
    update_specs=[
        {"type": "polynomial_reaction", "parameters": {
            "coefficients": [0.5, 0.0, -0.7], "dt": 0.02,
        }},
        {"type": "logistic_reaction", "parameters": {"dt": 0.02}},
        {"type": "mlp", "parameters": {
            "state_features": 1, "message_features": 1, "output_features": 1,
        }},
    ],
    config=config,
)
```

所有消息核统一输出从源节点传向目标节点的边消息；所有节点核统一输出状态增量，旧状态只由 `TaskKernelLayer` 加一次。新任务若需要额外领域核，可通过 `register_message_kernel` 或 `register_update_kernel` 注册，而不需要修改核心层。

## 分子多维特征与保守受力

`models/molecular_features.py` 将原子类型嵌入、多核消息、ADD、残差节点更新和原子能量读出组装起来。`variant="A"` 使用旧能量描述符投影，`"B"` 仅使用连续滤波，`"C"` 组合两者。默认两轮共享参数，不人为配平三组参数量。

ISO17 场景后续默认采用 `variant="C"`（原核＋连续滤波核，23,210 参数），A/B 保留为消融变体，旧标量模型保留历史记录。这是该场景的默认组装方案，不替代其他场景的核配置。冻结配置与完整测试记录见 `results/SOTA_Baseline/external/iso17/energy_force/feature_kernels/`。

```python
from taskkernel_gnn.models import MolecularFeatureEnergyModel
model = MolecularFeatureEnergyModel(variant="C", features=32, rounds=2)
context["position"].requires_grad_(True)
energy, force, diagnostics = model(edge_index, context, create_graph=True)
```

`context` 包含位置、逐原子 `atomic_numbers`、逐原子 `molecule_index`；A/C 还需已有分子能量核的元素对、连接掩码和角度三元组。类型表沿用 H/C/O。当前批量受力输出假设每个分子原子数相同。消息投影后代表描述特征，最终读出才是标量能量，力由同一能量对位置求负梯度。消息核和节点核详见中文手册中的 `molecular_features.md`。
