# 分子多维特征节点更新核

对应源码：[update/molecular_features.py](../../../src/taskkernel_gnn/kernels/update/molecular_features.py)。均输出 `[N,d]` 状态增量；TaskKernelLayer 在混合后统一执行 `h'=h+Δh`。

## MolecularFeatureLinearUpdate(features: int = 32)

**简述：**将聚合消息做可学习的无偏置线性投影，作为节点残差增量。

**公式：**

\[
\Delta h_i=P\bar m_i.
\]

`P` 为 `d×d` 可学习矩阵；不在核内部重复加入旧状态。

**调用方式：**

```python
from taskkernel_gnn.kernels import MolecularFeatureLinearUpdate
kernel = MolecularFeatureLinearUpdate(features=32)
increment = kernel(node_state, aggregate)
```

`features` 是正整数；`node_state` 与 `aggregate` 均为 `[N,features]`。此处不同于等维时采用恒等映射的 `LinearResidualUpdate`，本核始终学习投影矩阵。

## MolecularFeatureMLPUpdate(features: int = 32, hidden: int = 64)

**简述：**结合当前原子隐状态和聚合消息生成平滑残差更新。

**公式：**

\[
\Delta h_i=\operatorname{MLP}_\phi([h_i,\bar m_i]).
\]

网络为 `2d→hidden→hidden→d`，中间采用 SiLU。最后一层初始为零，输出不使用 tanh 限幅；这是隐特征更新，不是物理能量或受力的强度上限。

**调用方式：**

```python
from taskkernel_gnn.kernels import MolecularFeatureMLPUpdate
kernel = MolecularFeatureMLPUpdate(features=32, hidden=64)
increment = kernel(node_state, aggregate)
```

`features` 与 `hidden` 均为正整数；状态和聚合消息均为 `[N,features]`，输出同形状。`context` 可省略。
