# 神经消息核

对应源码：[message/neural.py](../../../src/taskkernel_gnn/kernels/message/neural.py)。神经核用于补充解析核未覆盖的作用机制。

## `EdgeGatedMessageKernel(node_features, edge_features, output_features, hidden=32)`

### 简述

根据边属性生成逐通道门值，再对源节点的线性投影进行调制。它适合边类型、材料属性或方向特征决定信息传递强弱的场景。

### 公式

\[
g_{ij}=\sigma\!\left(f_\theta(e_{ij})\right),
\qquad
m_{j\to i}=g_{ij}\odot W x_j.
\]

\(g_{ij}\in(0,1)^{F_{out}}\) 是逐通道门值，\(\odot\) 表示逐元素乘法。

### 调用方式

```python
kernel = EdgeGatedMessageKernel(
    node_features=4,
    edge_features=3,
    output_features=8,
    hidden=32,
)
message = kernel(node_state, edge_index, edge_attr=edge_attr)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `node_features` | 正整数 | 节点状态维数 `F_state` |
| `edge_features` | 正整数 | 边特征维数 `F_edge` |
| `output_features` | 正整数 | 消息输出维数 `F_message` |
| `hidden` | 正整数，常用 `16`～`256` | 门控 MLP 的隐藏宽度 |

运行时必须提供形状 `[E, edge_features]` 的 `edge_attr`，否则抛出 `ValueError`。

## `MLPMessageKernel(node_features, edge_features, output_features, hidden=64, eta=0.1)`

### 简述

拼接源状态、目标状态和边特征，再用 MLP 学习无法由解析核表达的消息。末层零初始化，使该残差核从零贡献开始训练。

### 公式

\[
z_{ij}=[x_j,x_i,e_{ij}],
\qquad
m_{j\to i}=\eta\tanh\!\left(f_\theta(z_{ij})\right).
\]

\(\eta\) 控制神经消息的最大绝对强度，是此前实验确定的尺度安全参数，默认值为 `0.1`。

### 调用方式

```python
kernel = MLPMessageKernel(
    node_features=4,
    edge_features=3,
    output_features=4,
    hidden=64,
    eta=0.1,
)
message = kernel(node_state, edge_index, edge_attr=edge_attr)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `node_features` | 正整数 | 源、目标节点状态的维数 |
| `edge_features` | 非负整数 | 边特征维数；为 `0` 时可省略 `edge_attr` |
| `output_features` | 正整数 | 消息输出维数 |
| `hidden` | 正整数，常用 `32`～`256` | 两层 MLP 的隐藏宽度 |
| `eta` | 源码接受任意浮点数；建议 `0 < eta <= 1`，默认 `0.1` | `tanh` 输出的强度倍率 |

若 `edge_attr=None`，实现会创建 `[E, 0]` 空特征；因此只有在构造时 `edge_features=0` 才能省略边特征。
