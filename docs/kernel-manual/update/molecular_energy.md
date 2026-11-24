# 分子能量节点更新核

对应源码：[update/molecular_energy.py](../../../src/taskkernel_gnn/kernels/update/molecular_energy.py)。输入 `aggregate` 是各有向边能量消息经 `add` 聚合得到的每原子标量，形状 `[N,1]`。模型随后对所有原子节点输出求和，得到分子总能量。

## MolecularSaturatingEnergyUpdate(scale: float = 1.0)

**简述：**将每原子聚合能量平滑限幅，避免异常大的局部消息直接主导总能量。与 `IdentityMessageUpdate` 并列作为节点更新候选。

**公式：**

\[
\Delta h_i=s\tanh(\bar m_i/s),\qquad s>0.
\]

`\bar m_i` 是聚合后的标量能量消息；`s` 为可学习正尺度，初值由 `scale>0` 给定，单位与能量一致。较小的 `s` 更早饱和；`s` 很大时近似恒等更新。

**调用方式：**

```python
from taskkernel_gnn.kernels import MolecularSaturatingEnergyUpdate
kernel = MolecularSaturatingEnergyUpdate(scale=1.0)
increment = kernel(node_state, aggregate)
```

`node_state` 与 `aggregate` 均为 `[N,1]`；`context` 可省略。当前构造函数只接受 `scale`，建议取正数；此实现通过正值变换保持训练中的尺度为正。该核输出状态增量，不自行加入旧状态。
