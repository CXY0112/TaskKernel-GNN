# 传播与输运消息核

对应源码：[message/transport.py](../../../src/taskkernel_gnn/kernels/message/transport.py)。

## `WavePropagationKernel(amplitude=1.0, wavenumber=1.0, frequency=1.0, phase=0.0)`

### 简述

使用余弦相位描述随距离和时间变化的波动传播，用作周期传播场的解析候选核。

### 公式

\[
m_{j\to i}=A\cos(kd_{ij}-\omega t+\phi)x_j,
\]

其中 \(A>0\) 是振幅，\(k>0\) 是波数，\(\omega>0\) 是角频率，\(\phi\) 是相位。

### 调用方式

```python
kernel = WavePropagationKernel(
    amplitude=1.0,
    wavenumber=2.0,
    frequency=0.5,
    phase=0.0,
)
message = kernel(
    node_state,
    edge_index,
    context={"position": position, "time": time},
)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `amplitude` | 浮点数，建议 `> 0` | 可学习的正振幅初值 |
| `wavenumber` | 浮点数，建议 `> 0` | 可学习的正波数初值 |
| `frequency` | 浮点数，建议 `> 0` | 可学习的正角频率初值 |
| `phase` | 浮点数，任意实数 | 可学习的初始相位 |

运行时要求 `context["position"]` 为 `[N, D]`，`context["time"]` 为标量或可广播张量。

## `DirectedTransportKernel(flow_key="directed_flow")`

### 简述

使用带方向和符号的逐边流量输运节点差值，适合对流、定向传播和非对称图。

### 公式

\[
m_{j\to i}=q_{ij}(x_j-x_i),
\]

其中 \(q_{ij}\) 是有符号边流量。正负号决定沿当前边方向的增强或抑制。

### 调用方式

```python
kernel = DirectedTransportKernel(flow_key="directed_flow")
message = kernel(
    node_state,
    edge_index,
    context={"directed_flow": edge_flow},
)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `flow_key` | 非空字符串 | `context` 中逐边流量的键名 |

`context[flow_key]` 形状可为 `[E]` 或 `[E, 1]`，数值可正、可负，也可为零。
