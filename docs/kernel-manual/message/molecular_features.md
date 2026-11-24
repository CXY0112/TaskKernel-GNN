# 分子多维特征消息核

对应源码：[message/molecular_features.py](../../../src/taskkernel_gnn/kernels/message/molecular_features.py)。这些核输出 `[E,d]` 特征，不直接输出真实能量。最终原子特征经能量读出后求和，受力由总能量求负梯度。`edge_index[0]` 为源节点 j，第二行为目标节点 i。

## ScalarEnergyFeatureMessage(energy_kernel: nn.Module, features: int = 32)

**简述：**将已有标量能量核作为物理描述符，通过无偏置投影适配到统一消息维度；原能量核的结构和参数均保留。

**公式：**

\[
m_{ij}=P_k e_{ij}^{(k)},\quad P_k\in\mathbb R^{d\times1}.
\]

`e` 是原核输出，`P` 可学习。投影后不能继续把该向量解释为物理能量分解。

**调用方式：**

```python
from taskkernel_gnn.kernels import MolecularBondEnergyKernel, ScalarEnergyFeatureMessage
kernel = ScalarEnergyFeatureMessage(MolecularBondEnergyKernel(), features=32)
message = kernel(node_state, edge_index, context=context)
```

`energy_kernel` 必须遵守消息核接口并输出 `[E,1]`；`features` 是正整数。运行时 `context` 的需求与被包装核一致。该包装器不自动让原核依赖邻居隐状态。

## MolecularContinuousFilterMessage(features: int = 32, radial_basis: int = 32, hidden: int = 32, cutoff: float = 8.0)

**简述：**由原子距离生成连续滤波器，乘以源原子的当前隐特征，向目标原子传递多维消息。是消息核，不包含节点更新或能量读出。

**公式：**

\[
\rho_q(r)=\exp[-(r-\mu_q)^2/(2\sigma^2)],\qquad
m_{ij}=f_c(r_{ij}) W_\theta(\rho(r_{ij}))\odot Vh_j.
\]

径向中心均匀覆盖 `[0,cutoff]`，`σ=cutoff/(radial_basis−1)`；`W` 是 SiLU 滤波 MLP，`V` 是无偏置线性投影。令 `x=min(r/cutoff,1)`，截断包络为 `1−10x³＋15x⁴−6x⁵`，边界处一、二阶导数均为零。

**调用方式：**

```python
from taskkernel_gnn.kernels import MolecularContinuousFilterMessage
kernel = MolecularContinuousFilterMessage(features=32, radial_basis=32, hidden=32, cutoff=8.0)
message = kernel(node_state, edge_index, context={"position": position})
```

`features` 和 `hidden` 为正整数，`radial_basis≥2`，`cutoff>0`。`node_state` 为 `[N,features]`，`position` 为 `[N,3]`，距离单位与 `cutoff` 一致。用于 Å 坐标时 `cutoff` 的单位为 Å。输出 `[E,features]`。训练力损失时位置需可求导；核只使用距离，故消息的特征通道对整体旋转不变。
