# 心脏电生理节点更新核

对应源码：[update/cardiac.py](../../../src/taskkernel_gnn/kernels/update/cardiac.py)。状态统一写为 \(h=[u,w]\)：\(u\) 是归一化跨膜电位，\(w\) 是恢复变量。聚合消息只加入电位方程。

## `aliev_panfilov_reaction(state, k, a, b, epsilon, mu1, mu2)`

### 简述

计算 Aliev–Panfilov 双变量模型的局部离子反应导数，不包含空间扩散和时间积分。该函数供数据生成器、独立积分器与 `AlievPanfilovUpdate` 复用。

### 公式

\[
\dot u=-ku(u-a)(u-1)-uw,
\]

\[
\dot w=\left(\epsilon+\frac{\mu_1w}{u+\mu_2}\right)
[-w-ku(u-b-1)].
\]

### 调用方式

```python
derivative = aliev_panfilov_reaction(
    state,
    k=7.8,
    a=0.085,
    b=0.105,
    epsilon=0.0105,
    mu1=0.115,
    mu2=0.31,
)
```

| 形参 | 类型与范围 | 说明 |
|---|---|---|
| `state` | `[..., 2]` 张量 | 最后一维依次为 `[u, w]` |
| `k` | 正标量或可广播张量 | 动作电位非线性强度 |
| `a` | 正标量，典型范围 `(0, 1)` | 激发阈值参数 |
| `b` | 正标量，典型范围 `(0, 1)` | 恢复动力学参数 |
| `epsilon` | 正标量，通常较小 | 基础恢复时间尺度 |
| `mu1` | 正标量 | 电位相关恢复强度 |
| `mu2` | 正标量 | 分母稳定项，应保证 `u + mu2` 不接近零 |

返回张量与 `state` 同形，依次为 `[du/dt, dw/dt]`。

## `fitzhugh_nagumo_reaction(state, epsilon=0.02, offset=0.10, recovery_scale=0.80)`

### 简述

计算 FitzHugh–Nagumo 双变量模型的局部反应导数，不包含扩散和时间积分。

### 公式

\[
\dot u=u-\frac{u^3}{3}-w,
\qquad
\dot w=\epsilon(u+s-\rho w),
\]

其中 \(s\) 对应 `offset`，\(\rho\) 对应 `recovery_scale`。

### 调用方式

```python
derivative = fitzhugh_nagumo_reaction(
    state,
    epsilon=0.02,
    offset=0.10,
    recovery_scale=0.80,
)
```

| 形参 | 类型与范围 | 说明 |
|---|---|---|
| `state` | `[..., 2]` 张量 | 最后一维依次为 `[u, w]` |
| `epsilon` | 浮点数，建议 `> 0` | 快慢变量时间尺度比 |
| `offset` | 浮点数，任意实数 | 恢复方程偏置 \(s\) |
| `recovery_scale` | 浮点数，通常 `> 0` | 恢复变量反馈强度 \(\rho\) |

返回与 `state` 同形的局部导数。

## `AlievPanfilovUpdate(dt=1.0, k=7.8, a=0.085, b=0.105, epsilon=0.0105, mu1=0.115, mu2=0.31)`

### 简述

将 Aliev–Panfilov 局部离子反应与消息聚合得到的电传导项组合，并返回一步 Euler 状态增量。

### 公式

\[
\Delta h_i=\Delta t
\begin{bmatrix}
R_u(u_i,w_i)+\bar m_i^{(u)}\\
R_w(u_i,w_i)
\end{bmatrix},
\]

其中 \(R_u,R_w\) 由上面的 Aliev–Panfilov 公式给出，只有电位方程接收空间聚合消息。

### 调用方式

```python
kernel = AlievPanfilovUpdate(
    dt=0.02,
    k=7.8,
    a=0.085,
    b=0.105,
    epsilon=0.0105,
    mu1=0.115,
    mu2=0.31,
)
increment = kernel(cardiac_state, diffusion_aggregate)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `dt` | 浮点数，正向积分必须 `> 0` | 时间步长 |
| `k` | 浮点数，建议 `> 0` | 非线性反应强度初值 |
| `a` | 浮点数，建议 `(0, 1)` | 激发阈值初值 |
| `b` | 浮点数，建议 `(0, 1)` | 恢复参数初值 |
| `epsilon` | 浮点数，建议 `> 0` | 基础恢复速率初值 |
| `mu1` | 浮点数，建议 `> 0` | 电位相关恢复参数初值 |
| `mu2` | 浮点数，建议 `> 0` | 分母稳定参数初值 |

六个反应参数都采用正值参数化并共同训练。`cardiac_state` 必须为 `[N, 2]`；`diffusion_aggregate` 至少有一个通道。

## `FitzHughNagumoUpdate(dt=1.0, epsilon=0.02)`

### 简述

将 FitzHugh–Nagumo 局部反应与电位扩散消息组合，返回一步 Euler 增量。当前类固定使用辅助函数的 `offset=0.10` 和 `recovery_scale=0.80`。

### 公式

\[
\Delta h_i=\Delta t
\begin{bmatrix}
u_i-u_i^3/3-w_i+\bar m_i^{(u)}\\
\epsilon(u_i+0.10-0.80w_i)
\end{bmatrix}.
\]

### 调用方式

```python
kernel = FitzHughNagumoUpdate(dt=0.02, epsilon=0.02)
increment = kernel(cardiac_state, diffusion_aggregate)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `dt` | 浮点数，正向积分必须 `> 0` | 时间步长 |
| `epsilon` | 浮点数，建议 `> 0` | 可学习的正快慢时间尺度比初值 |

状态必须为 `[N, 2]`，聚合消息至少有一个通道。

## `CardiacDecayUpdate(dt=1.0, rate=0.2)`

### 简述

用于控制实验的简化电位传播与全状态衰减核。它能隔离扩散和线性恢复机制，但不能替代完整离子模型。

### 公式

\[
\Delta h_i=\Delta t\left(-r h_i+
\begin{bmatrix}\bar m_i^{(u)}\\0\end{bmatrix}\right),
\qquad r>0.
\]

### 调用方式

```python
kernel = CardiacDecayUpdate(dt=0.02, rate=0.2)
increment = kernel(cardiac_state, diffusion_aggregate)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `dt` | 浮点数，正向积分必须 `> 0` | 时间步长 |
| `rate` | 浮点数，建议 `> 0` | 可学习的正衰减率初值 |

非正 `rate` 初值会被截到 `1e-8`。聚合消息只加入状态的第 `0` 通道。
