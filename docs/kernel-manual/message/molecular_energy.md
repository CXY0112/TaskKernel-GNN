# 分子能量消息核

对应源码：[message/molecular_energy.py](../../../src/taskkernel_gnn/kernels/message/molecular_energy.py)。本页各核输出形状均为 `[E,1]`：每条**有向边**承担一份能量，`add` 聚合并对节点求和后得到分子标量能量。原子力由总能量对坐标求负梯度得到，而不是把能量消息直接当作力。所有核需要 `context["position"]`（`[N,3]`，训练时需 `requires_grad=True`）。

## MolecularBondEnergyKernel(pair_types: int = 6)

**简述：**按无序元素对区分刚度和平衡键长的谐振子键能核；仅对 `bond_edge_mask=1` 的边生效。

**公式：**每个无向键的势能为

\[
E_{ij}^{\mathrm{bond}}=\frac12 k_{z_i z_j}(r_{ij}-r^0_{z_i z_j})^2.
\]

`r_{ij}` 是两原子距离；`k>0` 和 `r^0` 可学习。两个有向边各返回该键能的一半。`pair_types` 当前必须等于 6，对应 H/C/O 的六种无序组合。

**调用方式：**

```python
from taskkernel_gnn.kernels import MolecularBondEnergyKernel
kernel = MolecularBondEnergyKernel(pair_types=6)
message = kernel(node_state, edge_index, context=context)
```

还需 `context["pair_type"]`（`[E]`，取值 `0..5`）与 `context["bond_edge_mask"]`（`[E]`，取值 `0` 或 `1`）。

## MolecularAngleEnergyKernel()

**简述：**对三原子键角施加谐势；按中心原子 H/C/O 类型设刚度和平衡余弦。

**公式：**对三元组 `(i,j,k)`，其中 `j` 是中心原子，

\[
E_{ijk}^{\mathrm{angle}}=\frac12 k_{z_j}(\cos\theta_{ijk}-\cos\theta^0_{z_j})^2.
\]

`k>0` 和平衡余弦可学习。实现将每个三体势能平均写入四条有向边，确保所有消息求和后每个角只计一次。

**调用方式：**

```python
from taskkernel_gnn.kernels import MolecularAngleEnergyKernel
kernel = MolecularAngleEnergyKernel()
message = kernel(node_state, edge_index, context=context)
```

除位置外，还需 `angle_triplets`（`[T,3]`）、`angle_edge_indices`（`[T,4]`）及 `angle_center_type`（`[T]`，取值 `0..2`）。构造函数没有形参。

## MolecularNonbondedEnergyKernel()

**简述：**在非近邻原子对上表示平滑短程排斥及吸引势。

**公式：**

\[
E_{ij}^{\mathrm{nonbond}}=\frac12 k_{z_i z_j}\max(0,(R_{ij}-r_{ij})/R_{ij})^2
-A_{z_i z_j}\exp[-((r_{ij}-R_{ij})/0.8)^2].
\]

`k>0` 和 `A>0` 可学习；`R` 是核内固定的元素对半径。两条有向边各返回该势能的一半。该简化势不是量子化学能量的精确分解。

**调用方式：**

```python
from taskkernel_gnn.kernels import MolecularNonbondedEnergyKernel
kernel = MolecularNonbondedEnergyKernel()
message = kernel(node_state, edge_index, context=context)
```

还需 `pair_type`（`[E]`）与 `nonbonded_edge_mask`（`[E]`，取值 `0` 或 `1`）。构造函数没有形参。

## MolecularPairMLPEnergyKernel(hidden: int = 64, amplitude: float = 0.2)

**简述：**用对称的两体特征学习解析核未覆盖的势能修正。输入含距离径向基、元素对、成键标志和两端局部密度摘要；交换两端原子不改变输出。

**公式：**

\[
E_{ij}^{\mathrm{pair\,MLP}}=A\tanh(f_\theta(\phi_{ij})),\qquad
m_{j\to i}=\tfrac12 E_{ij}^{\mathrm{pair\,MLP}}.
\]

`A=amplitude>0` 限制每个无向对的修正范围为 `[-A,A]`；`hidden>=1` 是两层 MLP 的隐藏宽度。`\phi_{ij}` 为 17 维对称特征。MLP 最后一层零初始化，因此初始修正为零。

**调用方式：**

```python
from taskkernel_gnn.kernels import MolecularPairMLPEnergyKernel
kernel = MolecularPairMLPEnergyKernel(hidden=64, amplitude=0.2)
message = kernel(node_state, edge_index, context=context)
```

还需 `pair_type`（`[E]`）与 `bond_edge_mask`（`[E]`）。

## MolecularTripletMLPEnergyKernel(hidden: int = 32, amplitude: float = 0.2)

**简述：**学习环境条件化的三体势能修正。它读取键角余弦、两条键长、中心元素及两条中心—端点边的元素对类型，且交换两端原子后输出不变。

**公式：**

\[
E_{ijk}^{\mathrm{triplet\,MLP}}=A\tanh(g_\theta(\psi_{ijk})),
\]

每个三体势能分配给四条有向边，四份相加为原值。`A=amplitude>0` 限制每个三元组的修正范围为 `[-A,A]`；`hidden>=1`，`\psi_{ijk}` 是 18 维交换对称特征。最后一层零初始化。

**调用方式：**

```python
from taskkernel_gnn.kernels import MolecularTripletMLPEnergyKernel
kernel = MolecularTripletMLPEnergyKernel(hidden=32, amplitude=0.2)
message = kernel(node_state, edge_index, context=context)
```

还需 `pair_type`、`angle_triplets`、`angle_edge_indices` 与 `angle_center_type`。该核只对已列出的三元组生效；三元组来自外部连接关系构造。

## 能量与力的组装

```python
position = context["position"].requires_grad_(True)
node_energy, _ = layer(torch.zeros(len(position), 1), edge_index, context=context)
energy = torch.zeros(batch_size).index_add_(0, context["molecule_index"], node_energy[:, 0])
force = -torch.autograd.grad(energy.sum(), position, create_graph=True)[0]
```

训练时 `create_graph=True` 使力损失可反向传播到核参数；仅评估时可用 `False`。若启用非线性节点更新，最终能量仍可求导，但“逐核势能直接相加”的解释不再严格成立。
