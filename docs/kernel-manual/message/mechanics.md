# 力学消息核

对应源码：[message/mechanics.py](../../../src/taskkernel_gnn/kernels/message/mechanics.py)。稀疏消息核使用 `edge_index[0]=j`、`edge_index[1]=i`，输出逐边力；稠密辅助函数直接输出每个粒子的合力。

## `ViscousDampingKernel(damping=1.0, velocity_key="velocity")`

### 简述

根据相邻节点的完整相对速度生成黏性阻尼消息。它会同时作用于切向和法向速度分量。

### 公式

\[
m_{j\to i}=\gamma(v_j-v_i), \qquad \gamma>0.
\]

### 调用方式

```python
kernel = ViscousDampingKernel(damping=0.2, velocity_key="velocity")
message = kernel(
    node_state,
    edge_index,
    context={"velocity": velocity},
)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `damping` | 浮点数，建议 `> 0` | 可学习的正阻尼强度初值 |
| `velocity_key` | 非空字符串 | 速度在 `context` 中的键名 |

`context[velocity_key]` 应为 `[N, D]`，输出为 `[E, D]`。

## `SoftenedGravityKernel(epsilon=0.1)`

### 简述

计算带软化长度的成对引力。软化项消除粒子非常接近时的奇点，提高数值积分稳定性。

### 公式

\[
m_{j\to i}=Gm_im_j
\frac{r_j-r_i}{(\lVert r_j-r_i\rVert^2+\varepsilon^2)^{3/2}}.
\]

\(G\) 是引力常量，\(m_i,m_j\) 是质量，\(\varepsilon\) 是固定软化长度。

### 调用方式

```python
kernel = SoftenedGravityKernel(epsilon=0.1)
message = kernel(
    node_state,
    edge_index,
    context={
        "position": position,
        "mass": mass,
        "gravity_g": gravity_g,
    },
)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `epsilon` | 浮点数，建议 `> 0` | 距离软化长度；越大，近距离引力越平缓 |

`position` 为 `[N, D]`，`mass` 为 `[N]`，`gravity_g` 为标量。当前类不直接学习 \(G\) 或质量。

## `SpringKernel()`

### 简述

实现 Hooke 弹簧的逐边作用力。刚度和静止长度由每条边的上下文给出，因此同一图中可混合不同弹簧。

### 公式

\[
m_{j\to i}=k_{ij}(d_{ij}-\ell_{ij})\hat r_{ij},
\qquad
\hat r_{ij}=\frac{r_j-r_i}{d_{ij}}.
\]

当 \(d_{ij}>\ell_{ij}\) 时弹簧拉伸，当 \(d_{ij}<\ell_{ij}\) 时弹簧压缩。

### 调用方式

```python
kernel = SpringKernel()
message = kernel(
    node_state,
    edge_index,
    context={
        "position": position,
        "spring_k": spring_k,
        "rest_length": rest_length,
    },
)
```

该类没有构造形参。`spring_k` 和 `rest_length` 均为 `[E]`；物理上建议 `spring_k >= 0`、`rest_length > 0`。

## `PairwiseDashpotKernel()`

### 简述

计算只沿粒子连线方向作用的径向阻尼。与 `ViscousDampingKernel` 相比，它不会直接阻尼切向相对运动。

### 公式

\[
m_{j\to i}=\gamma_{ij}
\big[(v_j-v_i)\cdot\hat r_{ij}\big]\hat r_{ij}.
\]

### 调用方式

```python
kernel = PairwiseDashpotKernel()
message = kernel(
    node_state,
    edge_index,
    context={
        "position": position,
        "velocity": velocity,
        "dashpot_gamma": dashpot_gamma,
    },
)
```

该类没有构造形参。`dashpot_gamma` 为 `[E]`，物理上建议 `>= 0`；位置和速度均为 `[N, D]`。

## `dense_particle_components(batch, epsilon=0.1, gaussian_sigma=0.55)`

### 简述

面向批量全连接粒子系统的辅助函数，一次计算引力、弹簧力、径向阻尼和 Gaussian 候选力。它用于 Mixed-Force Particle Dynamics 实验，不经过稀疏 `edge_index` 接口。

### 公式

函数返回

\[
\{F^{gravity},F^{spring},F^{dashpot},F^{gaussian}\},
\]

每一项都已经对邻居 \(j\) 求和，形状为 `[B, N, D]`。Gaussian 候选使用固定幅值 `0.02`：

\[
F_i^{gaussian}=\sum_{j\ne i}
0.02\exp\!\left(-\frac{d_{ij}^2}{2\sigma^2}\right)\hat r_{ij}.
\]

### 调用方式

```python
components = dense_particle_components(
    batch,
    epsilon=0.1,
    gaussian_sigma=0.55,
)
force = sum(weights[name] * value for name, value in components.items())
```

| 形参 | 类型与范围 | 说明 |
|---|---|---|
| `batch` | 字典 | 必须包含下表列出的批量张量 |
| `epsilon` | 浮点数，建议 `> 0` | 引力软化长度 |
| `gaussian_sigma` | 浮点数，必须非零，建议 `> 0` | Gaussian 候选的空间尺度 |

`batch` 必须包含 `position [B,N,D]`、`velocity [B,N,D]`、`mass [B,N]`、`gravity_g [B]`、`bond [B,N,N]`、`spring [B,N,N]`、`rest [B,N,N]` 和 `dashpot [B,N,N]`。

## `EquivariantPairMLP(input_features=8, hidden=48, output_scale=0.06)`

### 简述

让 MLP 只预测成对作用的标量强度，再乘粒子间单位方向。这样，平移不会改变输出，旋转坐标会使输出力同步旋转。

### 公式

\[
s_{ij}=\eta\tanh\!\left(f_\theta(z_{ij})\right)M_{ij},
\qquad
F_i^{MLP}=\sum_j s_{ij}\hat r_{ij}.
\]

\(z_{ij}\) 是网格无关的成对连续特征，\(M_{ij}\) 是邻接掩码，\(\eta\) 对应 `output_scale`。

### 调用方式

```python
kernel = EquivariantPairMLP(
    input_features=8,
    hidden=48,
    output_scale=0.06,
)
force = kernel(pair_features, pair_direction, pair_mask)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `input_features` | 正整数 | 成对连续特征 `z_ij` 的维数 |
| `hidden` | 正整数，常用 `32`～`256` | 两层 MLP 的隐藏宽度 |
| `output_scale` | 源码接受任意浮点数；建议 `> 0` | 标量作用强度上限 |

运行时 `features` 为 `[B,N,N,input_features]`，`direction` 为 `[B,N,N,D]`，`mask` 为 `[B,N,N]`。末层零初始化，因此初始输出为零。
