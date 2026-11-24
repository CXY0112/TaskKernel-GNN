# 通用节点更新核

对应源码：[update/common.py](../../../src/taskkernel_gnn/kernels/update/common.py)。本页所有类都返回状态增量 \(\Delta h\)，外层统一执行 \(h^+=h+\Delta h\)。

## `IdentityMessageUpdate()`

### 简述

直接把聚合消息作为节点状态增量。适合消息和状态维数相同、聚合结果已经具有“一步增量”语义的场景。`ResidualUpdate` 是该类的兼容别名。

### 公式

\[
\Delta h_i=\bar m_i,
\]

其中 \(\bar m_i\) 是节点 \(i\) 收到的聚合消息。

### 调用方式

```python
kernel = IdentityMessageUpdate()
increment = kernel(node_state, aggregate)
```

该类没有构造形参。运行时要求 `aggregate.shape == node_state.shape`，否则外层状态更新无法相加。

## `LinearResidualUpdate(message_features=None, output_features=None, step_size=1.0)`

### 简述

先把聚合消息线性投影到状态空间，再乘固定步长。它适合消息维数与节点状态维数不同，或需要学习消息到状态增量映射的场景。

### 公式

\[
\Delta h_i=\Delta t\,P\bar m_i.
\]

当输入输出维数相同，或任一维数未指定时，\(P=I\)；否则 \(P\) 是无偏置可学习线性层。

### 调用方式

```python
kernel = LinearResidualUpdate(
    message_features=8,
    output_features=4,
    step_size=0.01,
)
increment = kernel(node_state, aggregate)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `message_features` | 正整数或 `None` | 聚合消息维数；`None` 时不创建投影 |
| `output_features` | 正整数或 `None` | 输出增量维数；应等于节点状态维数 |
| `step_size` | 浮点数；正向积分通常取 `> 0` | 固定倍率 \(\Delta t\) |

若只填写一个维数，当前实现仍使用恒等映射，因此建议两个维数同时填写或同时省略。

## `StateMessageLinearUpdate(state_features, message_features, output_features, step_size=1.0)`

### 简述

分别线性投影当前节点状态和聚合消息，再将两部分相加。它可表达线性自演化与邻居驱动的共同作用。

### 公式

\[
\Delta h_i=\Delta t\left(W_hh_i+W_m\bar m_i\right).
\]

### 调用方式

```python
kernel = StateMessageLinearUpdate(
    state_features=4,
    message_features=8,
    output_features=4,
    step_size=0.01,
)
increment = kernel(node_state, aggregate)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `state_features` | 正整数 | 当前状态维数 |
| `message_features` | 正整数 | 聚合消息维数 |
| `output_features` | 正整数 | 输出增量维数；通常等于 `state_features` |
| `step_size` | 浮点数；正向积分通常取 `> 0` | 整体更新步长 |

## `DynamicsEulerUpdate(dynamics, dt)`

### 简述

把任意连续时间动力学函数包装成显式 Euler 状态增量。适合快速接入新的解析节点动力学，而无需新建专用类。

### 公式

\[
\frac{dh_i}{dt}=f(h_i,\bar m_i,c),
\qquad
\Delta h_i=\Delta t\,f(h_i,\bar m_i,c).
\]

\(c\) 表示 `context` 中的额外参数。

### 调用方式

```python
def dynamics(state, message, context):
    return message - context["decay"] * state

kernel = DynamicsEulerUpdate(dynamics=dynamics, dt=0.01)
increment = kernel(
    node_state,
    aggregate,
    context={"decay": 0.2},
)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `dynamics` | `nn.Module` 或可调用对象 | 签名必须为 `(node_state, aggregate, context)`，返回导数张量 |
| `dt` | 浮点数；正向积分必须 `> 0` | Euler 时间步长 |

动力学返回值必须与 `node_state` 同形。当前类不检查稳定性，`dt` 上限应由具体方程的数值稳定条件确定。
