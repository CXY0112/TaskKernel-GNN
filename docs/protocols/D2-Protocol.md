# D2-v1：Graph Reaction–Diffusion 实施协议

## 1. 目标

D2 验证完整 TaskKernel-GNN 在图反应扩散系统上的预测、结构泛化和物理一致性。基础方程为

$$
\dot u_i
=D\sum_{j\in\mathcal N(i)}w_{ij}(u_j-u_i)
+\rho_1u_i-\rho_2u_i^3.
$$

消息生成表示空间扩散，默认使用 ADD 邻居聚合，节点更新表示局部反应和时间积分。D2-A 已完成一次 ADD/MEAN 选择实验，后续不再重复该聚合消融。

## 2. 参数协议

实验计划原稿要求每条轨迹随机采样 $D,\rho_1,\rho_2$，但当前 TaskKernel-GNN 的配置 $c_1,c_2$ 是任务级可学习参数，模型没有参数条件输入或任务编码器。若直接把不同参数轨迹混合训练，一个全局 $c$ 无法同时表示所有动力系统。

D2 因此拆成三个可解释阶段：

1. **D2-A Fixed-Task**：固定 $D=0.10,\rho_1=0.50,\rho_2=0.65$，训练图规模为 32、64。首先验证完整三阶段结构、核选择和 200 步稳定性。
2. **D2-B Known-Parameter**：每条轨迹采样参数，并把参数作为已知任务描述输入解析核和各基线。用于结构、初值、噪声和参数范围 OOD，不声称从轨迹恢复参数。
3. **D2-C Few-Shot Adaptation**：一个参数组合定义一个任务；模型在每个新任务上使用 1、4、16 条支持轨迹学习或微调 $a,b,c_1,c_2$。用于验证未知物理参数的少样本恢复。
4. **D2-D 3D Cardiac Reaction–Diffusion**：在真实或公开的三维心室网格上使用 monodomain 方程、各向异性传导张量和心肌细胞离子反应模型，作为三维高保真验证。

本轮先执行 D2-A，避免在验证基本 PDE 能力前引入任务编码问题。

PDEBench 外部验证与自生成实验分开报告。PDEBench 的 diffusion-reaction 数据为 1D/2D，因此只使用其 2D diffusion-reaction 数据作为 D2 外部验证，不开展 PDEBench 3D compressible Navier–Stokes。

## 3. 图与数据

| 项目 | D2-A 设置 |
|---|---|
| 图类型 | 1D ring、2D grid、Random Geometric Graph |
| 训练规模 | $N\in\{32,64\}$ |
| ID 测试 | 与训练同规模、不同图和初值随机种子 |
| 规模 OOD | $N\in\{128,256\}$ |
| 初值 | Gaussian random field 与局部脉冲的随机混合 |
| 时间步长 | $\Delta t=0.02$ |
| 训练窗口 | 20 步 |
| 测试 rollout | 20、50、100、200 步 |
| 聚合 | 固定为 ADD |

D2 的四个主场景为：

| 场景 | 空间/图 | 状态与方程 | 主要目的 |
|---|---|---|---|
| D2-Ring | 1D 周期环图 | 标量 cubic reaction–diffusion | 周期边界和最简单拓扑 |
| D2-Grid | 2D 规则网格图 | 标量 cubic reaction–diffusion | 规则 PDE 与 FNO/U-Net 比较 |
| D2-RGG | 2D 随机几何图 | 标量 cubic reaction–diffusion | 不规则拓扑、度数和结构 OOD |
| D2-Cardiac-3D | 三维心室体网格 | monodomain + 离子反应 | 各向异性、多状态、高保真三维验证 |

D2-Cardiac-3D 使用稀疏 `edge_index` 或四面体有限元矩阵，禁止构造 $N\times N$ 稠密距离矩阵。可视化使用跨膜电位波前和激活时间图；评价基于完整节点场。

D2-Cardiac-3D 采用三维心肌 monodomain reaction–diffusion：

$$
\chi C_m\frac{\partial V}{\partial t}
=\nabla\cdot(\boldsymbol\sigma\nabla V)
-\chi I_{\mathrm{ion}}(V,\mathbf z)+I_{\mathrm{stim}},
$$

$$
\frac{\partial\mathbf z}{\partial t}=G(V,\mathbf z).
$$

其中 $V$ 为跨膜电位，$\mathbf z$ 为离子门控状态，$\boldsymbol\sigma$ 为沿心肌纤维方向变化的各向异性传导张量。图节点来自三维四面体心室网格；消息核表示各向异性电扩散，节点更新核表示离子电流和门控反应。参考解由公开心脏电生理求解器或经过验证的数值实现生成。

D2-Cardiac-3D 分两档：理想化心室壁与简化 Aliev–Panfilov 反应用于代码和数值预检；具有真实心室几何、纤维方向、无通量边界、刺激协议和文献参数的细胞离子模型用于正式高保真结果。只有后一档可以表述为高保真三维验证。

正式网格首选 openCARP Modeling Resources 列出的 [whole-heart meshes](https://doi.org/10.5281/zenodo.3890033)：该资源包含基于心衰患者影像构建的体积网格、区域标签、规则法纤维角和心室坐标。若完整心脏规模超过第一轮计算预算，则使用同一资源中的心室子网格；备选为 [biventricular statistical shape model instances](https://doi.org/10.5281/zenodo.4419783)。下载后必须记录具体版本、许可证、节点/单元数量和预处理步骤。

参考轨迹使用 openCARP monodomain 求解，首轮采用 Aliev–Panfilov 做结构调试，正式结果采用 ten Tusscher–Panfilov 人体心室离子模型。必须先在 1D cable/2D sheet 上校准纵向与横向传导速度，再固定参数生成三维训练、验证和测试轨迹。

无向边使用对称非负权重

$$
w_{ij}=\exp[-6d_{ij}^2].
$$

Random Geometric Graph 首先按距离半径连边；若产生多个连通分量，则用分量间最近节点补边，保证每张图连通。

训练、验证、ID 测试和 OOD 测试使用互不重叠的图、初值和轨迹随机种子。超参数只依据验证集选择。

## 4. D2-A 候选核

消息核库：

$$
\left\{
D_1w_{ij}(u_j-u_i),
D_2e^{-\beta d_{ij}^2}(u_j-u_i),
q_{ij}(u_j-u_i),
K_{\mathrm{msg}}^{\mathrm{MLP}}
\right\}.
$$

节点更新核库：

$$
\left\{
\Delta t(M_i+\rho_1u_i-\rho_2u_i^3),
\Delta t[M_i+\rho_3u_i(1-u_i/C)],
\Delta t(M_i-\gamma u_i),
K_{\mathrm{upd}}^{\mathrm{MLP}}
\right\}.
$$

所有候选核先进行固定参考分布上的 RMS 尺度校准；报告 raw softmax 权重和尺度校准后的实际贡献。该处理回应 D1 中发现的核幅值不可比问题。

解析核完成尺度校准后，默认使用温度 softmax。训练前 30% 取 $\tau_a=\tau_b=1$，随后消息门控退火至 $\tau_a=0.5$，节点门控退火至 $\tau_b=0.3$。RGG-ID/OOD 对照中该设置优于标准 softmax 和 entmax-1.5。entmax-1.5 允许精确零权重，但当前实验未形成零支持，因此只保留为可选方案。

## 5. 精简后的比较模型

后续 D2 主实验固定使用以下五项；Oracle 不训练，因此每个设置最多训练四个模型。

| 报告名称 | 历史编号 | 主要作用 |
|---|---|---|
| Oracle | B0 | 数据生成器一致性上界，仅用于合成数据 |
| Learned-Single | B2L | 正确或验证集最优单核，并学习核配置 |
| MLP-only | B6 | 纯数据驱动基线 |
| Physics+Residual | B7 | 解析模型加非门控 MLP 残差 |
| TaskKernel-GNN | B8 | 多解析核与 MLP 自适应组合 |

B3 Multi-Msg、B4 Multi-Upd 和 B5 NoMLP 已完成一次性结构诊断，不再随每个 OOD 条件重复。NoMLP 只在缺失核实验中按需启用。所有模型默认使用 ADD，后续名称省略 `-ADD`。

## 6. 指标与通过标准

预测指标：one-step NRMSE，20/50/100/200 步 NRMSE，rollout 平均与最终 NRMSE，有效预测时间。

结构指标：图规模和拓扑 OOD NRMSE，以及按平均度分组的误差。

物理指标：

1. 对称扩散项的总和残差 $|\sum_i M_i|$；
2. 真实和预测总质量变化的误差；
3. 非有限值与状态幅值爆炸率；
4. 若初值限定为非负，则报告负状态违反率。

D2-A 数据预检必须满足：

- 所有图连通；
- 所有 200 步轨迹无 NaN/Inf；
- $\max|u|<3$；
- 对称扩散的每节点平均守恒残差低于 $10^{-6}$。

模型通过 D2-A 的最低条件为：TaskKernel-GNN 在 ID rollout 上优于 MLP-only，并在至少一个结构 OOD 条件上优于 MLP-only；同时报告 Learned-Single，防止把已知正确核的参数学习收益误判成多核收益。

## 7. 实施顺序

1. 数据生成与数值稳定性预检；
2. 已完成原始基线的 ID 小样本曲线；
3. 已完成 B3、B4、B7 一次性结构诊断；
4. 已完成 ADD/MEAN 选择和图规模 OOD，后续固定 ADD；
5. 已完成冻结版“尺度校准 + 温度 softmax”在 Ring/Grid ID 与新规模 OOD 上的确认，并使用新种子、新偏移强度完成 RGG 度数、密度和区域几何 OOD 独立复验；初值、时间、噪声和边缺失 OOD 尚待执行；
6. D2-B 已知参数条件化；
7. D2-C 每任务少样本参数适配。
8. D2-D1 理想化三维心脏正式训练已完成：ID、纤维和刺激测试保持有限，但 TaskKernel-GNN 在高分辨率网格 OOD 上发生 100% 爆炸；冻结 checkpoint 的受控干预已确认 MLP 消息输出随 $h^{-2}$ 放大是主要原因，输入尺度偏移是次要因素；
9. D2-D2 已完成 MLP 网格无关化验证：只移除 MLP 输入中的绝对网格尺度不足以解决 1248/2880 节点 Mesh-OOD；可选 $\eta$ 限制在当前任务取 0.1 时，3 个训练种子和全新留出轨迹在 464/768/1248/2880 节点上均无爆炸，Mesh-1248/ID NRMSE 比为 0.894。对称化显著增加误差，因此保留为可选项；$\eta=0.1$ 仍需 Ring/Grid/RGG 回归验证，不作为通用默认值；
10. D2-D3 已完成关闭对称性后的 MLP 强度多数据集敏感性验证：Ring、Grid、RGG 对 $\eta\in[0.05,1]$ 整体不敏感，Cardiac 明显偏好 0.05–0.1，且 $\eta=1$ 在 1248/2880 节点上爆炸。按四个数据集族等权的相对误差折损选择 $\eta_0=0.1$ 作为通用初始值；Cardiac 固定网格与非网格 OOD 可改用 0.05，Mesh-OOD 使用 0.1；
11. D2-D4 冻结最终方法的理想化三维心脏基线比较已完成：固定 $\eta=0.1$、关闭对称化，并使用未参与参数选择的新数据种子。TaskKernel-GNN 在 ID、三档 Mesh-OOD 和 Fiber-OOD 上优于公平修正后的 MLP-only 与 Physics+Residual，全部条件爆炸率为 0%，Mesh-1248/ID 与 Mesh-2880/ID NRMSE 比分别为 0.898 和 0.855；Stimulus-OOD 差于 Physics+Residual，是进入高保真阶段后需要保留的压力测试；
12. D2-D5a 解剖几何桥接已完成：使用公开平均双心室表面生成 6/4.5/3.5 mm 三档四面体网格，以规则法纤维、四面体 FEM monodomain 和 Aliev–Panfilov 建立 190 时间单位轨迹。三档参考解全心激活且 APD90 覆盖 99.86%–100%，TaskKernel 在 ID 和两档 Mesh-OOD 均无爆炸；但其 ID NRMSE 为 0.652，差于 Physics+Residual 的 0.207。节点 AP 核权重达到 0.993，而简单图核与消息 MLP 干扰项仍占 0.379，说明当前瓶颈是消息核选择，不是网格尺度适配。该阶段只算部分通过，不表述为最终高保真验证；
13. D2-D5b 暂缓扩大到 ten Tusscher–Panfilov。先在 D2-D5a 固定数据上只改变消息门控机制，预注册准确度和 Mesh-OOD 稳定性门槛；通过后再加入患者特异纤维、openCARP 参考轨迹、刺激位置和导电参数变化；
14. PDEBench 2D diffusion-reaction 外部验证。

### 7.1 RGG 拓扑 OOD 定义

RGG 的主要拓扑 OOD 不采用“Ring/Grid 训练，RGG 测试”。后者属于跨图族零样本迁移，只能作为额外压力测试。正式设置保持训练和测试都属于随机几何图：

| 划分 | 节点坐标分布 | 目标平均度数 | 区域几何 |
|---|---|---:|---|
| 训练/ID | 单位正方形均匀分布 | 约 10–13 | 正方形 |
| Degree-OOD-Low | 单位正方形均匀分布 | 约 5–7，并保持连通 | 正方形 |
| Degree-OOD-High | 单位正方形均匀分布 | 约 18–22 | 正方形 |
| Density-OOD | 混合 Gaussian 或空间非均匀密度 | 与训练集匹配 | 正方形 |
| Geometry-OOD | 均匀分布 | 与训练集匹配 | L 形或带孔区域 |

只更换随机种子而保持生成分布不变属于 RGG-ID，不称为 OOD。节点数从 32/64 变为 128/256 属于 Size-OOD，已经单独完成。每一种 RGG 分布分别列出结果，不合并成唯一总体数值。

## 8. PDEBench 外部验证边界

当前预检数据没有使用 PDEBench。正式外部验证采用 PDEBench 官方数据和官方 train/test 划分信息，并保留原边界条件：

- **D2-External-RD**：2D diffusion-reaction，$128\times128$ 网格，转换为四邻域图；与 FNO、U-Net 及 TaskKernel-GNN 比较。

PDEBench 3D compressible Navier–Stokes 不在本项目当前实验范围内。三维高保真结果由 D2-Cardiac-3D 承担，从而保持“空间扩散消息 + 局部反应更新”的方法对应关系。

