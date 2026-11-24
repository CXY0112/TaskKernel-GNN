# 神经节点更新核

对应源码：[update/neural.py](../../../src/taskkernel_gnn/kernels/update/neural.py)。两种核都返回增量，与解析节点核共享同一套权重混合和外层状态更新。

## `GRUUpdate(state_features, message_features)`

### 简述

使用 GRUCell 将聚合消息作为输入、当前节点状态作为隐状态，适合具有记忆效应或难以显式写出局部更新方程的任务。

### 公式

\[
\tilde h_i=\operatorname{GRUCell}(\bar m_i,h_i),
\qquad
\Delta h_i=\tilde h_i-h_i.
\]

减去旧状态是为了把 GRU 的“下一状态”转换成 TaskKernel-GNN 统一要求的“状态增量”。

### 调用方式

```python
kernel = GRUUpdate(state_features=16, message_features=8)
increment = kernel(node_state, aggregate)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `state_features` | 正整数 | 节点状态及 GRU 隐状态维数 |
| `message_features` | 正整数 | 聚合消息维数 |

运行时 `node_state` 为 `[N, state_features]`，`aggregate` 为 `[N, message_features]`，输出与 `node_state` 同形。

## `MLPUpdate(state_features, message_features, output_features, hidden=64, eta=0.1)`

### 简述

拼接当前状态和聚合消息，用 MLP 学习解析节点核未覆盖的局部状态增量。末层零初始化，使训练初始阶段的神经更新为零。

### 公式

\[
z_i=[h_i,\bar m_i],
\qquad
\Delta h_i=\eta\tanh\!\left(f_\theta(z_i)\right).
\]

### 调用方式

```python
kernel = MLPUpdate(
    state_features=4,
    message_features=8,
    output_features=4,
    hidden=64,
    eta=0.1,
)
increment = kernel(node_state, aggregate)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `state_features` | 正整数 | 当前节点状态维数 |
| `message_features` | 正整数 | 聚合消息维数 |
| `output_features` | 正整数；通常等于 `state_features` | 输出增量维数 |
| `hidden` | 正整数，常用 `32`～`256` | 两层 MLP 的隐藏宽度 |
| `eta` | 源码接受任意浮点数；建议 `0 < eta <= 1`，默认 `0.1` | 神经增量的强度上限 |

若 `output_features != state_features`，外层 `TaskKernelLayer` 会因增量无法与状态相加而报错。
