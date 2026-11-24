# TaskKernel-GNN 核库

本目录是实验共享的候选算子库。消息核的统一接口为：

```python
message = kernel(node_state, edge_index, edge_attr, context)
```

`edge_index[0]` 是源节点，`edge_index[1]` 是目标节点，输出第一维必须与边数相同。节点核的统一接口为：

```python
increment = kernel(node_state, aggregate, context)
```

节点核只输出状态增量，不直接重复加入旧状态；`TaskKernelLayer` 在混合所有节点核后统一执行一次 `next_state = node_state + increment`。

## 目录职责

第一层严格按 TaskKernel-GNN 的两个可学习阶段划分，第二层再按领域划分：

```text
kernels/
├── message/                 消息生成核
│   ├── common.py            差分、图扩散与距离核
│   ├── transport.py         波传播与有向输运
│   ├── mechanics.py         引力、弹簧、阻尼及粒子辅助算子
│   ├── fluids.py            密度、压力、黏性与流体残差消息
│   ├── cardiac.py           心脏 FEM 离散适配器
│   ├── molecular.py         分子受力消息核
│   ├── molecular_energy.py  分子能量消息核
│   ├── molecular_features.py 分子多维特征与连续滤波消息
│   └── neural.py            边门控与 MLP 消息
├── update/                  节点更新核
│   ├── common.py            恒等、线性与通用 Euler 更新
│   ├── reaction.py          通用局部反应与阻尼
│   ├── mechanics.py         二阶力学积分
│   ├── fluids.py            统一重力、墙面接触与流体积分
│   ├── wave.py              波动增广状态更新
│   ├── cardiac.py           AP、FHN 与心脏衰减
│   ├── molecular.py         分子受力更新核
│   ├── molecular_energy.py  分子能量更新核
│   ├── molecular_features.py 分子隐特征线性与 MLP 更新
│   └── neural.py            GRU 与 MLP 更新
├── base.py                  两类核共用的张量与参数辅助函数
└── registry.py              具名注册表、构造函数与核库组装
```

`base.py` 和 `registry.py` 是基础设施，不定义新的消息核或节点核，因此保留在根目录。详细的公式、形参和调用示例见[中文核函数说明手册](../../../docs/kernel-manual/README.md)。

## 组装原则

核注册表是完整候选目录，但单个实验只选择状态维度、物理单位和输出语义兼容的子集。新增领域核时实现相同接口并调用：

```python
register_message_kernel("my_message", MyMessageKernel)
register_update_kernel("my_update", MyUpdateKernel)
```

之后即可在YAML或`TaskKernelGNN.from_specs`中按名称构造，无需修改门控、校准、聚合或主模型代码。
