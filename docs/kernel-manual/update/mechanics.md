# 力学节点更新核

对应源码：[update/mechanics.py](../../../src/taskkernel_gnn/kernels/update/mechanics.py)。状态布局固定为 \([r,v]\)：前 `spatial_dims` 维是位置，后 `spatial_dims` 维是速度；聚合消息 `force` 表示合力。

## `SecondOrderMechanicsUpdate(dt, spatial_dims=2, damping=0.0)`

### 简述

二阶牛顿动力学的显式 Euler 更新基类。它根据合力、质量和速度阻尼计算位置与速度增量。

### 公式

\[
a_i=\frac{F_i}{m_i}-\gamma v_i,
\]

\[
\Delta r_i=\Delta t\,v_i,
\qquad
\Delta v_i=\Delta t\,a_i.
\]

### 调用方式

```python
kernel = SecondOrderMechanicsUpdate(
    dt=0.005,
    spatial_dims=3,
    damping=0.01,
)
increment = kernel(
    particle_state,
    force,
    context={"mass": mass},
)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `dt` | 浮点数，正向积分必须 `> 0` | 时间步长 |
| `spatial_dims` | 正整数，通常为 `1`、`2` 或 `3` | 空间维数 \(D\) |
| `damping` | 浮点数，建议 `>= 0` | 可学习的正速度阻尼初值；`0` 会以 `1e-8` 初始化 |

运行时要求 `particle_state` 为 `[N, 2D]`、`force` 为 `[N, D]`、`context["mass"]` 为 `[N]` 且质量应 `> 0`。

## `ExplicitEulerUpdate(dt, spatial_dims=2, damping=0.0)`

### 简述

`SecondOrderMechanicsUpdate` 的语义化子类，公式、形参和运行时要求完全相同。实验配置中注册名为 `second_order_explicit`。

### 公式

\[
(r^{n+1},v^{n+1})=(r^n,v^n)+
\Delta t\left(v^n,\frac{F^n}{m}-\gamma v^n\right).
\]

### 调用方式

```python
kernel = ExplicitEulerUpdate(dt=0.005, spatial_dims=3, damping=0.01)
increment = kernel(particle_state, force, context={"mass": mass})
```

形参范围与 `SecondOrderMechanicsUpdate` 相同。

## `SemiImplicitEulerUpdate(dt, spatial_dims=2, damping=0.0)`

### 简述

先更新速度，再使用新速度更新位置。对弹簧等较刚性的系统，它通常比显式位置更新更稳定。

### 公式

\[
a_i^n=\frac{F_i^n}{m_i}-\gamma v_i^n,
\qquad
v_i^{n+1}=v_i^n+\Delta t\,a_i^n,
\]

\[
\Delta r_i=\Delta t\,v_i^{n+1},
\qquad
\Delta v_i=\Delta t\,a_i^n.
\]

### 调用方式

```python
kernel = SemiImplicitEulerUpdate(
    dt=0.005,
    spatial_dims=3,
    damping=0.01,
)
increment = kernel(particle_state, force, context={"mass": mass})
```

形参范围、状态布局和上下文要求与 `SecondOrderMechanicsUpdate` 相同。实验配置中的注册名为 `second_order_semi_implicit`。
