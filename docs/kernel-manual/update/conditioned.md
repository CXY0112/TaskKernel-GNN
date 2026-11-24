# 带历史与条件输入的节点更新核

对应源码：`src/taskkernel_gnn/kernels/update/conditioned.py`。

## ConditionedMLPUpdate(context_features, hidden=64, eta=0.1, dt=1.0)

**简述**：利用可观测历史和已知条件近似未观测的局部内部动力学，同时保留聚合消息的传输作用。例如，只有膜电位可观测时，它可以近似离子门控变量所产生的记忆效应；不等同于精确恢复全部离子状态。

**公式**：

$$\Delta h_i=\Delta t\,[\overline m_i+\eta f_\theta(z_i,\overline m_i)].$$

其中 $z_i$ 为历史与条件特征，$\overline m_i$ 为消息聚合。此处 `eta` 控制局部 MLP 项；不等于消息核 `eta` 的物理相对系数含义，也不是节点门控权重 $b$。

**调用方式**：

```python
from taskkernel_gnn.kernels.update import ConditionedMLPUpdate
kernel = ConditionedMLPUpdate(context_features=16, hidden=64, eta=0.1, dt=1.0)
increment = kernel(state, aggregate, context=context)
```

| 参数 | 含义与范围 |
|---|---|
| `context_features` | 正整数，与 `context["node_features"].shape[-1]` 一致 |
| `hidden` | 两个隐藏层的宽度，正整数 |
| `eta` | 局部 MLP 输出的非负强度倍率，默认 0.1；MLP 输出本身没有 tanh 限幅 |
| `dt` | 正的积分时间步；若聚合量为每 ms 变化率，单位为 ms |
| `state` | `[N,1]`；本核主要通过条件特征读取当前状态 |
| `aggregate` | `[N,1]` 的聚合变化率 |
| `context` | 至少含 `node_features:[N,F]`，必须由过去观测或自由预测构造 |

输出是 `[N,1]` 的状态增量。E2 适配器将本核 `dt` 固定为 1，再在外层统一乘内部子步长度，避免重复乘时间步。
