# 波动节点更新核

对应源码：[update/wave.py](../../../src/taskkernel_gnn/kernels/update/wave.py)。

## `WaveStateUpdate(dt, damping=0.0)`

### 简述

把二阶时间波动方程写成一阶增广状态 \([u,v]\)，其中 \(u\) 是场值，\(v=\partial_tu\) 是时间速度。

### 公式

\[
\partial_tu=v,
\qquad
\partial_tv=\bar m-\gamma v,
\]

\[
\Delta h=\Delta t[v,\bar m-\gamma v].
\]

聚合消息 \(\bar m\) 的第一通道表示空间传播产生的加速度。

### 调用方式

```python
kernel = WaveStateUpdate(dt=0.01, damping=0.02)
increment = kernel(wave_state, aggregate)
```

| 构造形参 | 类型与范围 | 说明 |
|---|---|---|
| `dt` | 浮点数，正向积分必须 `> 0` | 时间步长 |
| `damping` | 浮点数，建议 `>= 0` | 可学习的正波动阻尼初值；`0` 会以 `1e-8` 初始化 |

`wave_state` 的最后一维必须严格为 `2`，布局为 `[u, v]`；否则抛出 `ValueError`。`aggregate` 至少需要一个通道。
