# 流体消息核

对应源码：[message/fluids.py](../../../src/taskkernel_gnn/kernels/message/fluids.py)。这些核是可复用的粒子流体近似，不等同于特定 MPM 求解器。`edge_index[0]` 为源粒子 (j)，`edge_index[1]` 为接收粒子 (i)；逐边输出均为二维加速度消息。

## `compact_density(position, edge_index, radius)`

### 简述

先按邻居距离估计每个粒子的标量局部密度。它是压力核和残差 MLP 的输入，不经过消息门控。

### 公式

\[
\rho_i=1+\sum_{j\in\mathcal N(i)}\bigl[\max(1-r_{ij}/h,0)\bigr]^2.
\]

其中 (r_{ij}=\|x_j-x_i\|)，(h) 为作用半径，常数 1 是粒子自身的贡献。

### 调用方式

```python
from taskkernel_gnn.kernels.message.fluids import compact_density
density = compact_density(position, edge_index, radius=0.015)
```

| 形参 | 形状与范围 | 作用 |
|---|---|---|
| `position` | `[N, 2]` | 粒子坐标 |
| `edge_index` | `[2, E]` | 有向邻接边 |
| `radius` | 正数 | 密度核支撑半径，宜与建图半径一致 |

返回 `[N]`。将结果写入 `context["density"]` 供以下消息核和更新核使用。

## `DensityPressureKernel(radius=0.015, strength=100.0, reference_density=1.45)`

### 简述

局部密度超过参考值时，沿粒子连线产生向外压力作用；两端粒子获得方向相反的消息。

### 公式

\[
p_i=\max(\rho_i-\rho_0,0),\quad
m_{j\to i}=k\frac{p_i+p_j}{2}\max(1-r_{ij}/h,0)
\frac{x_i-x_j}{r_{ij}}.
\]

(k>0) 为可学习强度，(ho_0>0) 为可学习参考密度，(h>0) 为固定支撑半径。这个公式是通用的近似压力核，不宣称还原 MPM 的完整应力计算。

### 调用方式

```python
from taskkernel_gnn.kernels.message.fluids import DensityPressureKernel
kernel = DensityPressureKernel(radius=0.015, strength=100.0, reference_density=1.45)
message = kernel(state, edge_index, context={"position": position, "density": density})
```

| 构造形参 | 范围 | 作用 |
|---|---|---|
| `radius` | `>0` | 压力相互作用距离 |
| `strength` | 初值建议 `>0` | 可学习的正压力强度 |
| `reference_density` | 初值建议 `>0` | 可学习的无量纲参考密度 |

## `FluidViscosityKernel(radius=0.015, damping=1.0)`

### 简述

按距离衰减相邻粒子的相对速度，近似局部黏性阻尼。

### 公式

\[
m_{j\to i}=\nu\max(1-r_{ij}/h,0)(v_j-v_i),\quad\nu>0.
\]

### 调用方式

```python
from taskkernel_gnn.kernels.message.fluids import FluidViscosityKernel
kernel = FluidViscosityKernel(radius=0.015, damping=1.0)
message = kernel(state, edge_index, context={"position": position, "velocity": velocity})
```

| 构造形参 | 范围 | 作用 |
|---|---|---|
| `radius` | `>0` | 相互作用距离 |
| `damping` | 初值建议 `>0` | 可学习的正阻尼强度 |

## `FluidResidualMessageKernel(hidden=64, eta=0.1, acceleration_scale=100.0)`

### 简述

以位置、速度、相对几何、两端密度和目标粒子的墙面距离学习未被解析核覆盖的逐边修正。末层零初始化。

### 公式

\[
m_{j\to i}=\eta A_0\tanh\bigl(\operatorname{MLP}(s_j,s_i,\Delta x_{ij},r_{ij},\rho_j,\rho_i,d_i^{wall})\bigr).
\]

### 调用方式

```python
from taskkernel_gnn.kernels.message.fluids import FluidResidualMessageKernel
kernel = FluidResidualMessageKernel(hidden=64, eta=0.1, acceleration_scale=100.0)
message = kernel(state, edge_index, edge_attr, context={
    "density": density, "wall_features": wall_features,
})
```

`state` 为 `[N,4]` 的位置/速度，`edge_attr` 为 `[E,3]` 的相对位置/距离，`wall_features` 为 `[N,4]` 的归一化墙面距离。

| 构造形参 | 范围 | 作用 |
|---|---|---|
| `hidden` | 正整数 | 隐藏层宽度 |
| `eta` | 建议 `[0,1]` | 残差相对强度，默认 0.1 |
| `acceleration_scale` | `>0` | 把无量纲网络输出换算到加速度尺度 |
