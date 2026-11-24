# 带历史与条件输入的连续消息核

对应源码：`src/taskkernel_gnn/kernels/message/conditioned.py`。这些核可用于只有部分状态可观测、同时具有已知外部驱动的扩散系统；不局限于某个心脏数据集。

## ConditionedDiffusionMessage(context_features, hidden=64, eta=0.1, symmetric=False)

**简述**：利用节点历史与已知条件学习连续扩散系数修正，随后通过离散适配器生成标量消息。

**公式**：

$$m_{j\to i}=\eta\tanh f_\theta(z_j,z_i,\widehat{r_j-r_i})\,w_{ij}(h_j-h_i).$$

其中 $z_i$ 是 `context["node_features"]`，$w_{ij}$ 是已完成物理单位转换与空间离散的 `edge_weight`。MLP 不接收 $1/h^2$；网格尺度由适配器处理。此核仅输出修正项，基础扩散通过其他候选核提供。`symmetric=True` 时先将两个方向的 MLP 原始输出平均，再取 tanh；默认关闭。对称系数不意味着质量归一化后的消息权重相等。

**调用方式**：

```python
from taskkernel_gnn.kernels.message import ConditionedDiffusionMessage
kernel = ConditionedDiffusionMessage(context_features=16, hidden=64, eta=0.1)
message = kernel(state, edge_index, context=context)
```

| 参数 | 含义与范围 |
|---|---|
| `context_features` | 每节点条件特征维数，正整数，须与 `node_features.shape[-1]` 一致 |
| `hidden` | 两个隐藏层的宽度，正整数 |
| `eta` | 连续系数修正幅度，非负数；默认 0.1；不是门控权重 |
| `symmetric` | 布尔值，是否平均交换源、目标后的系数 |
| `state` | `[N,1]` 的当前标量状态 |
| `edge_index` | `[2,E]` 的源、目标节点索引 |
| `context` | `node_features:[N,F]`、`position:[N,3]`、`edge_weight:[E]` |

输出 `[E,1]`；空间单位由 `position` 和适配器共同约定。

## GaussianAdapterDiffusionMessage(scale=1.0, beta=1.0)

**简述**：在已有扩散适配器上施加可学习的高斯距离调制。

**公式**：

$$m_{j\to i}=s\exp(-\beta\|r_j-r_i\|^2)w_{ij}(h_j-h_i).$$

`scale` 与 `beta` 均用对数参数化保持正值。它是相对于基础离散扩散的调制核，不是另一个独立有限元求解器。

**调用方式**：

```python
from taskkernel_gnn.kernels.message import GaussianAdapterDiffusionMessage
kernel = GaussianAdapterDiffusionMessage(scale=1.0, beta=1.0)
message = kernel(state, edge_index, context=context)
```

| 参数 | 含义与范围 |
|---|---|
| `scale` | 正的初始强度倍率，训练可学习 |
| `beta` | 正的距离衰减系数，训练可学习；若坐标为 mm，单位为 mm⁻² |
| `state`、`edge_index` | 同上 |
| `context` | `position:[N,3]`、`edge_weight:[E]` |

输出 `[E,1]`。
