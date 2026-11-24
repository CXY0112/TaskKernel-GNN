# 反应节点更新核

对应源码：[update/reaction.py](../../../src/taskkernel_gnn/kernels/update/reaction.py)。这些核将聚合得到的空间传播项 \(\bar m_i\) 与节点局部反应项相加，再乘时间步长。

## `LinearReactionUpdate(rate=0.0, dt=1.0, learnable=True)`

### 简述

描述与当前状态成正比的局部增长或衰减。正反应率表示增长，负反应率表示衰减。

### 公式

\[
\Delta h_i=\Delta t(\bar m_i+r h_i).
\]

### 调用方式

```python
kernel = LinearReactionUpdate(rate=-0.2, dt=0.01, learnable=True)
increment = kernel(node_state, aggregate)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `rate` | 浮点数，任意实数 | 线性反应率 \(r\) 的初值 |
| `dt` | 浮点数；正向积分通常取 `> 0` | 时间步长 |
| `learnable` | `bool` | 是否训练反应率 |

`aggregate` 必须能与 `node_state` 逐元素相加。

## `PolynomialReactionUpdate(coefficients: Sequence[float], dt=1.0)`

### 简述

使用可学习多项式表示局部非线性反应，可覆盖二次、三次以及更高阶反应动力学。

### 公式

\[
R(h_i)=\sum_{q=1}^{Q}c_qh_i^q,
\qquad
\Delta h_i=\Delta t[\bar m_i+R(h_i)].
\]

系数列表从一次项开始，不包含常数项。

### 调用方式

```python
kernel = PolynomialReactionUpdate(
    coefficients=[0.5, 0.0, -0.7],
    dt=0.02,
)
increment = kernel(node_state, aggregate)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `coefficients` | 非空实数序列 | `[c1, c2, ..., cQ]`，全部注册为可训练参数 |
| `dt` | 浮点数；正向积分通常取 `> 0` | 时间步长 |

空系数列表会抛出 `ValueError`。高阶项可能导致大状态下数值爆炸，应结合状态归一化和步长选择。

## `CoupledPolynomialReactionUpdate(state_features, output_features=None, degree=3, dt=1.0, initial_scale=0.02)`

### 简述

使用完整的多变量多项式基表示局部反应。它同时包含常数项、各状态通道的自作用项以及跨通道乘积，适用于多组分 Reaction–Diffusion，而不绑定某一种具体方程。

### 公式

令节点状态为 \(x_i\in\mathbb R^C\)，\(\phi_{\le d}(x_i)\) 包含从零次到 \(d\) 次的全部不重复单项式：

\[
R(x_i)=W\phi_{\le d}(x_i),
\qquad
\Delta x_i=\Delta t[\bar m_i+R(x_i)].
\]

当 \(C=2,d=3\) 时：

\[
\phi_{\le3}(u,v)=
[1,u,v,u^2,uv,v^2,u^3,u^2v,uv^2,v^3].
\]

### 调用方式

```python
kernel = CoupledPolynomialReactionUpdate(
    state_features=2,
    output_features=2,
    degree=3,
    dt=0.05,
    initial_scale=0.02,
)
increment = kernel(node_state, aggregate)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `state_features` | 正整数 | 输入状态通道数 \(C\) |
| `output_features` | 正整数或 `None` | 输出通道数；`None` 时等于 `state_features` |
| `degree` | 正整数；通常建议 `1–3` | 多项式最高次数 |
| `dt` | 浮点数；正向积分通常取 `> 0` | 时间步长 |
| `initial_scale` | 非负浮点数 | 系数正态初始化的标准差 |

三阶以上的基数量和数值增长速度都会快速增加，默认通用候选使用三阶。

## `SaturatingReactionUpdate(state_features, output_features=None, half_saturation=1.0, dt=1.0, initial_scale=0.02)`

### 简述

使用线性状态与有界饱和特征的耦合组合表示局部反应。它适合近似具有有限响应幅度的激活、抑制和生化饱和动力学。

### 公式

每个状态通道构造有界特征：

\[
s_c(x_c)=\frac{x_c}{K_c+|x_c|},
\qquad K_c>0.
\]

反应与一步增量为：

\[
R(x_i)=W[1,x_i,s(x_i)],
\qquad
\Delta x_i=\Delta t[\bar m_i+R(x_i)].
\]

### 调用方式

```python
kernel = SaturatingReactionUpdate(
    state_features=2,
    output_features=2,
    half_saturation=1.0,
    dt=0.05,
    initial_scale=0.02,
)
increment = kernel(node_state, aggregate)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `state_features` | 正整数 | 输入状态通道数 |
| `output_features` | 正整数或 `None` | 输出通道数；`None` 时等于输入通道数 |
| `half_saturation` | 正浮点数 | 所有通道的初始半饱和尺度 \(K_c\) |
| `dt` | 浮点数；正向积分通常取 `> 0` | 时间步长 |
| `initial_scale` | 非负浮点数 | 反应组合系数初始化标准差 |

`half_saturation` 在内部采用正值参数化，训练过程中始终保持为正。

## `LogisticReactionUpdate(rate=1.0, capacity=1.0, dt=1.0)`

### 简述

实现 Logistic 增长：低状态时近似指数增长，接近承载量时增长逐渐停止。

### 公式

\[
\Delta h_i=\Delta t\left[
\bar m_i+r h_i\left(1-\frac{h_i}{K}\right)
\right],
\qquad r,K>0.
\]

### 调用方式

```python
kernel = LogisticReactionUpdate(rate=1.0, capacity=1.0, dt=0.01)
increment = kernel(node_state, aggregate)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `rate` | 浮点数，建议 `> 0` | 可学习的正增长率初值 |
| `capacity` | 浮点数，建议 `> 0` | 可学习的正承载量初值 |
| `dt` | 浮点数；正向积分通常取 `> 0` | 时间步长 |

非正的 `rate` 或 `capacity` 初值会被截到 `1e-8` 后再进行正值参数化。

## `DampedUpdate(damping=0.1, dt=1.0)`

### 简述

在聚合消息驱动之外加入一阶线性衰减，用于耗散系统和稳定性控制。

### 公式

\[
\Delta h_i=\Delta t(\bar m_i-\gamma h_i),
\qquad \gamma>0.
\]

### 调用方式

```python
kernel = DampedUpdate(damping=0.1, dt=0.01)
increment = kernel(node_state, aggregate)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `damping` | 浮点数，建议 `> 0` | 可学习的正衰减率初值 |
| `dt` | 浮点数；正向积分通常取 `> 0` | 时间步长 |

非正阻尼初值会被截到 `1e-8`。
