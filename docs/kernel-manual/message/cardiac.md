# 心脏电传导消息核

对应源码：[message/cardiac.py](../../../src/taskkernel_gnn/kernels/message/cardiac.py)。

## `AdapterDiffusionKernel(adapter_key, scale=1.0, state_channel=0, learnable=True)`

### 简述

把外部离散适配器提供的逐边系数转换为扩散消息。它用于四面体有限元等场景：网格尺度、单元几何和各向异性导电率由适配器系数承担，核只学习整体强度修正。

### 公式

\[
m_{j\to i}=s\,A_{ij}\left(x_j^{(c)}-x_i^{(c)}\right),
\qquad s>0,
\]

其中 \(A_{ij}\) 是外部离散适配器系数，\(c\) 是使用的状态通道，\(s\) 是整体尺度。

### 调用方式

```python
kernel = AdapterDiffusionKernel(
    adapter_key="fem_edge_weight",
    scale=0.1,
    state_channel=0,
    learnable=True,
)
message = kernel(
    cardiac_state,
    edge_index,
    context={"fem_edge_weight": fem_edge_weight},
)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `adapter_key` | 非空字符串，无默认值 | `context` 中适配器系数的键名 |
| `scale` | 浮点数，建议 `> 0`；非正初值会被截到 `1e-8` | 整体扩散强度初值 |
| `state_channel` | 整数，`0 <= value < F_state` | 参与扩散的状态通道，心脏模型通常取电位通道 `0` |
| `learnable` | `bool` | 是否训练 `scale` |

`context[adapter_key]` 应为 `[E]`。输出为 `[E, 1]`，因此后续节点核应知道该消息只对应电位分量。
