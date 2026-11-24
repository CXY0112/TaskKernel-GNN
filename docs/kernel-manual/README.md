# TaskKernel-GNN 中文核函数手册

本手册对应 `src/taskkernel_gnn/kernels/`，用于查询每个候选核的用途、数学定义、构造形参、运行时输入和调用方法。目录与源码严格镜像：第一层区分**消息核**与**节点更新核**，第二层按领域分类。

## 阅读约定

每个核都按以下顺序说明：

1. **标题**：类名或函数名以及完整构造形参；
2. **简述**：该核描述的作用及适用场景；
3. **公式**：消息或状态增量的数学定义，并解释符号；
4. **调用方式**：可运行的最小示例、形参含义和取值范围。

参数表中的“范围”同时说明源码约束和物理建议。若写作“源码接受任意实数，通常取正数”，表示当前实现不会主动拒绝负值，但正值才符合常见物理语义。

各页示例默认已从统一公开入口导入对应符号，例如：

```python
from taskkernel_gnn.kernels import LinearResidualUpdate
```

## 统一接口

消息核生成从源节点 \(j\) 指向目标节点 \(i\) 的逐边消息：

```python
message = kernel(node_state, edge_index, edge_attr=None, context=None)
```

| 输入 | 典型形状 | 含义 |
|---|---:|---|
| `node_state` | `[N, F_state]` | 所有节点的当前状态 |
| `edge_index` | `[2, E]` | `edge_index[0]` 为源节点，`edge_index[1]` 为目标节点 |
| `edge_attr` | `[E, F_edge]` 或 `None` | 可选逐边特征 |
| `context` | `dict` 或 `None` | 位置、速度、质量、时间等附加物理量 |
| 返回值 | `[E, F_message]` | 每条有向边的消息 |

节点更新核根据当前状态和聚合消息返回**状态增量**：

```python
increment = kernel(node_state, aggregate, context=None)
```

`TaskKernelLayer` 在混合所有节点核后统一执行

\[
h_i^{\ell+1}=h_i^\ell+\Delta h_i.
\]

因此节点核不应自行再次加入旧状态。

## 文档目录

```text
kernel-manual/
├── message/
│   ├── common.md       基础差分、图扩散与距离核
│   ├── transport.md    波传播与有向输运
│   ├── mechanics.md    引力、弹簧、阻尼与粒子辅助算子
│   ├── fluids.md       流体密度、压力、黏性与补全
│   ├── cardiac.md      心脏 FEM 离散适配器
│   ├── conditioned.md  条件输入与连续系数修正
│   ├── molecular_energy.md  分子键、角、非键及可学习能量消息
│   ├── molecular_features.md  分子多维特征与连续滤波消息
│   └── neural.md       边门控与 MLP 消息
└── update/
    ├── common.md       恒等、线性与通用 Euler 更新
    ├── reaction.md     通用局部反应与阻尼
    ├── mechanics.md    二阶力学积分
    ├── fluids.md       流体重力、墙面与半隐式更新
    ├── wave.md         波动增广状态更新
    ├── cardiac.md      AP、FHN 与心脏衰减
    ├── conditioned.md  历史与条件输入的局部动力学补全
    ├── molecular_energy.md  分子标量能量更新
    ├── molecular_features.md  分子隐特征线性与 MLP 更新
    └── neural.md       GRU 与 MLP 更新
```

| 阶段 | 领域 | 文档 |
|---|---|---|
| 消息生成 | 通用扩散与距离 | [message/common.md](message/common.md) |
| 消息生成 | 传播与输运 | [message/transport.md](message/transport.md) |
| 消息生成 | 力学 | [message/mechanics.md](message/mechanics.md) |
| 消息生成 | 流体 | [message/fluids.md](message/fluids.md) |
| 消息生成 | 心脏电传导 | [message/cardiac.md](message/cardiac.md) |
| 消息生成 | 分子能量 | [message/molecular_energy.md](message/molecular_energy.md) |
| 消息生成 | 分子多维特征 | [message/molecular_features.md](message/molecular_features.md) |
| 消息生成 | 神经补全 | [message/neural.md](message/neural.md) |
| 节点更新 | 通用更新 | [update/common.md](update/common.md) |
| 节点更新 | 局部反应 | [update/reaction.md](update/reaction.md) |
| 节点更新 | 力学积分 | [update/mechanics.md](update/mechanics.md) |
| 节点更新 | 流体 | [update/fluids.md](update/fluids.md) |
| 节点更新 | 波动状态 | [update/wave.md](update/wave.md) |
| 节点更新 | 心脏电生理 | [update/cardiac.md](update/cardiac.md) |
| 节点更新 | 分子能量 | [update/molecular_energy.md](update/molecular_energy.md) |
| 节点更新 | 分子多维特征 | [update/molecular_features.md](update/molecular_features.md) |
| 节点更新 | 神经更新 | [update/neural.md](update/neural.md) |

源码入口：[TaskKernel-GNN 核库](../../src/taskkernel_gnn/kernels/README.zh-CN.md)。
