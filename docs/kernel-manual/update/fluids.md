# 流体节点更新核与共用加速度

对应源码：[update/fluids.py](../../../src/taskkernel_gnn/kernels/update/fluids.py)。重力、墙面接触在两种节点更新候选之前计算，并传入相同的 `context`；因此它们不会被更新核的 (b)-softmax 缩小。

## `UniformGravityAcceleration(dimensions=2)`

### 简述

提供作用于所有粒子的同一重力加速度，与粒子间反平方引力不同。

### 公式

\[g_i=g,\qquad i=1,\dots,N.\]

### 调用方式

```python
from taskkernel_gnn.kernels.update.fluids import UniformGravityAcceleration
gravity = UniformGravityAcceleration(dimensions=2)
acceleration = gravity(position)
```

`dimensions` 为正整数，输出与 `position` 同形。向量 `gravity.acceleration` 可从训练轨迹初始化，随后参与学习；二维下典型方向为负 (y)。

## `SoftBoxWallAcceleration(dimensions=2, contact_radius=0.015, strength=200.0, damping=20.0, maximum_offset=0.02, initial_lower_offset=-0.005, initial_upper_offset=0.005)`

### 简述

用可学习的接触位置、法向软排斥和仅在靠近墙时生效的法向速度阻尼，描述轴对齐盒形边界。切向速度不会直接被墙面阻尼削弱。

### 公式

对任一坐标轴的下壁，令有效墙面 (L=L_0+\delta_L)，距离 (d_L=x-L)，接触量 (q_L=\max(1-d_L/h,0))，则

\[
a_L=kq_L+\gamma q_L\max(-v,0).
\]

上壁 (U=U_0+\delta_U) 给出方向相反的同型作用。墙面偏移限制在 ((-s,s))，其中 (s=\texttt{maximum_offset})。

### 调用方式

```python
from taskkernel_gnn.kernels.update.fluids import SoftBoxWallAcceleration
wall = SoftBoxWallAcceleration(contact_radius=0.015, strength=200.0, damping=20.0)
acceleration = wall(position, velocity, bounds)
```

| 构造形参 | 范围 | 作用 |
|---|---|---|
| `dimensions` | 正整数 | 空间维度 |
| `contact_radius` | `>0` | 距墙多近开始接触 |
| `strength` | 初值建议 `>0` | 可学习的正排斥强度 |
| `damping` | 初值建议 `>0` | 可学习的正法向接近阻尼 |
| `maximum_offset` | `>0` | 可学习墙面偏移的绝对上限 |
| `initial_lower_offset`、`initial_upper_offset` | 绝对值小于 `maximum_offset` | 相对元数据边界的初始偏移 |

## `FluidSemiImplicitUpdate(dt=0.0025, spatial_dims=2)`

### 简述

把聚合消息、统一重力和墙面作用相加为总加速度，再按半隐式 Euler 计算位置与速度增量。

### 公式

\[
A_i=\bar m_i+g_i+a_i^{wall},\quad
\Delta v_i=\Delta t A_i,\quad
\Delta x_i=\Delta t(v_i+\Delta v_i).
\]

### 调用方式

```python
from taskkernel_gnn.kernels.update.fluids import FluidSemiImplicitUpdate
kernel = FluidSemiImplicitUpdate(dt=0.0025, spatial_dims=2)
increment = kernel(state, aggregate, context={
    "gravity_acceleration": gravity_acceleration,
    "wall_acceleration": wall_acceleration,
})
```

`dt` 必须为正，`spatial_dims` 为正整数；`state` 是 `[N,2*spatial_dims]`，`aggregate` 与两项共用加速度均是 `[N,spatial_dims]`。

## `FluidResidualAccelerationUpdate(dt=0.0025, spatial_dims=2, hidden=64, eta=0.1, acceleration_scale=100.0)`

### 简述

保留与上一个核完全相同的重力、墙面和解析消息推进，再增加有界的 MLP 加速度修正。与物理更新核混合时，(b) 只控制这项额外修正。

### 公式

\[
R_i=\eta A_0\tanh\bigl(\operatorname{MLP}(s_i,\bar m_i,\rho_i,d_i^{wall})\bigr),
\qquad A_i=\bar m_i+g_i+a_i^{wall}+R_i.
\]

位置/速度增量仍由半隐式积分公式给出。

### 调用方式

```python
from taskkernel_gnn.kernels.update.fluids import FluidResidualAccelerationUpdate
kernel = FluidResidualAccelerationUpdate(dt=0.0025, hidden=64, eta=0.1)
increment = kernel(state, aggregate, context={
    "gravity_acceleration": gravity_acceleration,
    "wall_acceleration": wall_acceleration,
    "density": density,
    "wall_features": wall_features,
})
```

`hidden` 为正整数，`eta` 建议在 `[0,1]`，`acceleration_scale` 为正数；末层零初始化。
