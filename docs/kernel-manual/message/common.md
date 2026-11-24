# 通用消息核

对应源码：[message/common.py](../../../src/taskkernel_gnn/kernels/message/common.py)。以下公式统一使用 \(j\to i\) 表示从源节点到目标节点的有向边，\(x_i\) 表示节点状态。

## `DifferenceMessageKernel(scale: float = 1.0, learnable: bool = False)`

### 简述

最基础的邻居差分核。它只比较相邻节点的状态，不依赖位置或边属性，常用于图扩散、平滑和受控基线。

### 公式

\[
m_{j\to i}=s(x_j-x_i),
\]

其中 \(s\) 是整体尺度。若图中同时包含 \(i\to j\) 与 \(j\to i\)，两条边消息方向相反。

### 调用方式

```python
kernel = DifferenceMessageKernel(scale=0.1, learnable=True)
message = kernel(node_state, edge_index)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `scale` | 浮点数，任意实数；通常取 `> 0` | 差分消息的初始倍率 |
| `learnable` | `bool` | `True` 时将 `scale` 注册为可训练参数 |

运行时只要求 `node_state` 与 `edge_index`；输出形状为 `[E, F_state]`。

## `GraphDiffusionKernel(diffusion: float = 1.0, weight_key: str = "edge_weight", learnable: bool = True)`

### 简述

使用预先离散好的图边权进行扩散。它适合有限差分、有限元或图 Laplacian 已经给出逐边权重的情况。

### 公式

\[
m_{j\to i}=D\,w_{ij}(x_j-x_i), \qquad D>0,
\]

其中 \(D\) 是扩散系数，\(w_{ij}\) 是离散图权。聚合后得到离散扩散算子的一部分。

### 调用方式

```python
kernel = GraphDiffusionKernel(diffusion=0.05, weight_key="edge_weight")
message = kernel(
    node_state,
    edge_index,
    context={"edge_weight": edge_weight},
)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `diffusion` | 浮点数，建议 `> 0`；非正初值会被截到 `1e-8` | 扩散系数初值 |
| `weight_key` | 非空字符串 | `context` 中逐边权重的键名 |
| `learnable` | `bool` | 是否训练扩散系数 |

`context[weight_key]` 的形状可为 `[E]` 或 `[E, 1]`。边权通常非负；若允许负边权，核会产生局部反扩散贡献。

## `GaussianDiffusionKernel(diffusion=1.0, beta=1.0, position_key="position")`

### 简述

按节点距离施加 Gaussian 衰减的差分扩散核。近邻影响较强，远邻影响快速减弱。

### 公式

\[
m_{j\to i}=D\exp(-\beta d_{ij}^{2})(x_j-x_i),
\qquad d_{ij}=\lVert r_j-r_i\rVert.
\]

\(D>0\) 控制总强度，\(\beta>0\) 控制空间衰减速度；特征长度约为 \(1/\sqrt{\beta}\)。

### 调用方式

```python
kernel = GaussianDiffusionKernel(diffusion=0.1, beta=2.0)
message = kernel(node_state, edge_index, context={"position": position})
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `diffusion` | 浮点数，建议 `> 0` | 可学习的正扩散系数初值 |
| `beta` | 浮点数，建议 `> 0` | 可学习的正距离衰减率 |
| `position_key` | 非空字符串 | `context` 中节点坐标的键名 |

`context[position_key]` 的形状为 `[N, D]`，其中 `D` 是空间维数。

## `GaussianDistanceKernel(sigma=1.0, diffusion=1.0, position_key="position")`

### 简述

`GaussianDiffusionKernel` 的标准差参数化版本。它用更直观的长度尺度 \(\sigma\) 代替 \(\beta\)。

### 公式

\[
m_{j\to i}=D\exp\!\left(-\frac{d_{ij}^{2}}{2\sigma^{2}}\right)(x_j-x_i),
\qquad \beta=\frac{1}{2\sigma^2}.
\]

### 调用方式

```python
kernel = GaussianDistanceKernel(sigma=0.5, diffusion=0.1)
message = kernel(node_state, edge_index, context={"position": position})
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `sigma` | 浮点数，必须非零，物理上建议 `> 0` | Gaussian 空间尺度；当前实现不单独训练 `sigma` |
| `diffusion` | 浮点数，建议 `> 0` | 可学习的正扩散强度初值 |
| `position_key` | 非空字符串 | 坐标在 `context` 中的键名 |

## `ExponentialDiffusionKernel(amplitude=1.0, beta=1.0, position_key="position")`

### 简述

使用一次距离指数衰减的差分传播核。相较 Gaussian 核，它在远距离处衰减更慢。

### 公式

\[
m_{j\to i}=A\exp(-\beta d_{ij})(x_j-x_i),
\qquad A,\beta>0.
\]

### 调用方式

```python
kernel = ExponentialDiffusionKernel(amplitude=0.2, beta=1.5)
message = kernel(node_state, edge_index, context={"position": position})
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `amplitude` | 浮点数，建议 `> 0` | 可学习的正消息强度初值 |
| `beta` | 浮点数，建议 `> 0` | 可学习的正距离衰减率 |
| `position_key` | 非空字符串 | 坐标在 `context` 中的键名 |

## `InversePowerDistanceKernel(amplitude=1.0, power=1.0, epsilon=0.1, position_key="position")`

### 简述

按距离逆幂衰减并直接传递源节点状态。它适合长程影响候选，但不是差分型守恒核。

### 公式

\[
m_{j\to i}=A(d_{ij}+\varepsilon)^{-p}x_j,
\qquad A,p>0.
\]

\(\varepsilon\) 防止距离接近零时出现奇点。因为公式使用 \(x_j\) 而不是 \(x_j-x_i\)，是否守恒取决于图结构与后续更新。

### 调用方式

```python
kernel = InversePowerDistanceKernel(
    amplitude=1.0,
    power=2.0,
    epsilon=1e-2,
)
message = kernel(node_state, edge_index, context={"position": position})
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `amplitude` | 浮点数，建议 `> 0` | 可学习的正强度初值 |
| `power` | 浮点数，建议 `> 0` | 可学习的正幂指数 |
| `epsilon` | 浮点数，建议 `> 0` | 固定的距离软化量 |
| `position_key` | 非空字符串 | 坐标在 `context` 中的键名 |
