# TaskKernel-GNN 实验设置

## 1. 实验目标

实验需要验证 TaskKernel-GNN 的四个核心设计：

1. 消息生成阶段的多核组合是否能够从少量数据中识别有效交互机制；
2. 节点更新阶段的多核组合是否能够表达不同的局部状态转移；
3. 解析核与 MLP 核的组合是否能够在核库不完整时补足未知机制；
4. 机制核是否能够改善小样本学习、长时间预测和分布外泛化。

实验按照“可控机制恢复 → 完整物理系统 → 核库缺失 → 外部数据验证”的顺序开展。前两个阶段使用可控合成数据，以获得真实核权重和配置参数；外部数据只用于验证预测能力，不用于声称真实机制恢复。

## 2. Research Questions

| RQ | 研究问题 | 主要实验 | 核心比较 | 主要指标 |
|---|---|---|---|---|
| RQ1 | 多消息核是否优于单消息核？ | 可控核恢复、反应扩散、多力场粒子 | Full、Single-Msg、Multi-Msg、MLP-only | 消息误差、rollout NRMSE、$a$ 恢复误差 |
| RQ2 | 多节点更新核是否优于单更新核？ | 可控核恢复、反应扩散 | Full、Single-Upd、Multi-Upd、MLP-only | 状态增量误差、rollout NRMSE、$b$ 恢复误差 |
| RQ3 | TaskKernel-GNN 是否具有小样本优势？ | 三个核心数据集的数据量曲线 | Full、MPNN、机制模型、Physics+Residual | 不同训练轨迹数下的 NRMSE、误差曲线面积 |
| RQ4 | MLP 核能否补足候选核库之外的机制？ | 缺失核、错误核、局部缺失实验 | Full、No-MLP、MLP-only、Oracle-Library | 缺失区域误差、MLP 实际贡献、rollout NRMSE |
| RQ5 | 学到的核权重和配置是否与真实生成机制一致？ | 可控核恢复、参数范围外推 | 参数固定/分开学习/联合学习 | $a,b,c_1,c_2$ 误差、有效贡献误差、权重稳定性 |
| RQ6 | 模型是否具有更好的结构和时间泛化？ | 图规模、图族内部拓扑分布、参数、初值、时间 OOD | 精简后的主基线 | OOD NRMSE、有效预测时间、守恒/能量误差 |
| RQ7 | ADD 是否能保持叠加型物理消息和规模泛化？ | 反应扩散、多力场粒子 | 默认 ADD；已有一次性 MEAN 对照 | OOD NRMSE、物理量误差、度数敏感性 |
| RQ8 | 人工指定、数据驱动和混合模式的性能如何变化？ | 所有可控任务 | Manual、Learned、Hybrid、Oracle | 参数误差、预测误差、所需数据量 |

## 3. 实验数据

### 3.1 D1：Controlled Kernel Identification

该数据专门验证核选择、参数恢复和三阶段模型是否按预期工作。数据直接由 TaskKernel 形式生成：

$$
m_{ij}
=\sum_{p=1}^{P}a_{ij}^{(p),\star}
K_{\mathrm{msg}}^{(p)}
(x_i,x_j,e_{ij};c_{1}^{(p),\star}),
$$

$$
M_i=\operatorname{ADD}_{j\in\mathcal N(i)}m_{ij},
$$

$$
x_i'
=x_i+
\sum_{q=1}^{Q}b_i^{(q),\star}
K_{\mathrm{upd}}^{(q)}
(x_i,M_i;c_{2}^{(q),\star}).
$$

默认设置：

| 项目 | 设置 |
|---|---|
| 图类型 | Random Geometric Graph、Erdős–Rényi graph |
| 训练图规模 | $N\in[20,40]$ |
| ID 测试规模 | $N\in[20,40]$ |
| OOD 测试规模 | $N\in[80,120]$ |
| 节点状态 | 2–4 维连续状态 |
| 消息核数 | 3 个真实候选核 + 2 个干扰核 + 1 个 MLP 核 |
| 更新核数 | 3 个真实候选核 + 2 个干扰核 + 1 个 MLP 核 |
| 轨迹长度 | 100–200 步 |
| 训练池 | 64 条独立轨迹 |
| 验证集 | 64 条独立轨迹 |
| ID 测试集 | 128 条独立轨迹 |
| OOD 测试集 | 128 条独立轨迹 |

构造五个难度等级：

| 设置 | 数据生成方式 | 验证目的 |
|---|---|---|
| D1-Single | 一个消息核和一个更新核占主导 | 检查模型能否退化到单核 |
| D1-Mixture | 两至三个消息核、更新核同时激活 | 验证多核叠加 |
| D1-Distractor | 核库加入不参与生成的干扰核 | 验证核选择能力 |
| D1-Correlated | 加入响应形状相近的候选核 | 测试权重稳定性和可辨识难度 |
| D1-Missing | 删除一个真实核并保留 MLP | 验证未知机制补全 |

真实权重可以从具有不同稀疏度的 Dirichlet 分布采样：

$$
a^\star\sim\operatorname{Dirichlet}(\alpha_a),
\qquad
b^\star\sim\operatorname{Dirichlet}(\alpha_b).
$$

分别采用 $\alpha<1$、$\alpha=1$ 和 $\alpha>1$，产生稀疏、均匀和密集的多核组合。

### 3.2 D2：Graph Reaction–Diffusion

该任务作为完整 TaskKernel-GNN 的主要 PDE 实验：消息核表达空间扩散，ADD 表达邻居消息叠加，节点更新核表达反应动力学和时间更新。

基础方程采用

$$
\dot u_i
=D\sum_{j\in\mathcal N(i)}w_{ij}(u_j-u_i)
+\rho_1u_i-\rho_2u_i^3.
$$

消息核库：

$$
\mathcal B_{\mathrm{msg}}
=\left\{
D_1w_{ij}(u_j-u_i),
D_2e^{-\beta d_{ij}^2}(u_j-u_i),
q_{ij}(u_j-u_i),
K_{\mathrm{msg}}^{\mathrm{MLP}}
\right\}.
$$

节点更新核库：

$$
\mathcal B_{\mathrm{upd}}
=\left\{
\Delta t(M_i+\rho_1u_i-\rho_2u_i^3),
\Delta t[M_i+\rho_3u_i(1-u_i/C)],
\Delta t(M_i-\gamma u_i),
K_{\mathrm{upd}}^{\mathrm{MLP}}
\right\}.
$$

数据设置：

| 项目 | 设置 |
|---|---|
| 图结构 | 1D ring、2D grid、Random Geometric Graph |
| 训练图规模 | $N\in\{32,64\}$ |
| ID 测试规模 | $N\in\{32,64\}$ |
| OOD 测试规模 | $N\in\{128,256\}$ |
| 初值 | Gaussian random field 与局部脉冲混合 |
| 参数采样 | $D,\rho_1,\rho_2$ 从训练区间均匀采样 |
| 时间长度 | 训练窗口 20 步，测试 rollout 200 步 |
| 训练池 | 256 条轨迹 |
| 验证集 | 64 条轨迹 |
| ID 测试集 | 128 条轨迹 |
| OOD 测试集 | 每个 OOD 条件 128 条轨迹 |

OOD 设置：

- 未见图规模；
- grid 训练、随机几何图测试；
- 未见初始频率或脉冲数量；
- $D,\rho_1,\rho_2$ 超出训练区间 20%–50%；
- 训练 20 步、测试 50/100/200 步；
- 节点状态加入 1%、5%、10% Gaussian 噪声；
- 随机删除 5%、10%、20% 的边。

完成自生成图数据后，使用 [PDEBench](https://github.com/pdebench/PDEBench) 的 reaction-diffusion 数据进行外部验证。规则网格转为图时，将网格点作为节点，将有限差分邻接关系作为边，并固定相同的边界处理。

增加三维扩展时需要区分方程类型：

- **PDEBench 2D 外部验证**：PDEBench 的 diffusion-reaction 数据为 1D/2D，其中 2D 数据适合作为 D2 的外部规则网格测试。
- **D2-Cardiac-3D**：作为 ring、grid、随机几何图之外的第四个主场景，在三维心室四面体网格上采用 monodomain 方程。扩散项使用心肌纤维方向决定的各向异性传导张量，反应项使用心肌细胞离子电流与门控变量，边界采用组织表面的无通量条件，并给出明确刺激协议。参考轨迹由经过验证的心脏电生理求解器生成。该设置承担三维高保真验证。

不开展 PDEBench 3D compressible Navier–Stokes。三维场采用等值面或体渲染展示，但模型评价必须在完整体素或图节点上计算 NRMSE、激活时间误差、传导速度误差和波前形状误差。高保真结论必须同时具备真实几何、各向异性、合理边界条件、刺激协议和文献或实验标定参数，不能只依据三维可视化效果。

### 3.3 D3：Mixed-Force Particle Dynamics

该任务验证多个矢量消息核、add 聚合以及二阶节点更新。真实系统同时包含引力、弹簧和成对 dashpot 阻尼。dashpot 只作用于稀疏弹簧边，并按作用力与反作用力成对施加；普通介质速度阻尼属于节点更新，不作为消息核：

$$
F_{ij}
=F_{ij}^{\mathrm{gravity}}
+F_{ij}^{\mathrm{spring}}
+F_{ij}^{\mathrm{dashpot}},
$$

$$
F_i=\sum_jF_{ij}.
$$

消息核库：

$$
\mathcal B_{\mathrm{msg}}
=\left\{
K_{\mathrm{gravity}},
K_{\mathrm{spring}},
K_{\mathrm{dashpot}},
K_{\mathrm{gaussian}},
K_{\mathrm{msg}}^{\mathrm{MLP}}
\right\}.
$$

节点更新核库：

$$
\mathcal B_{\mathrm{upd}}
=\left\{
K_{\mathrm{Euler}},
K_{\mathrm{damped}},
K_{\mathrm{second\text{-}order}},
K_{\mathrm{upd}}^{\mathrm{MLP}}
\right\}.
$$

数据设置：

| 项目 | 设置 |
|---|---|
| 空间 | 2D 为主，3D 作为扩展 |
| 训练粒子数 | $N\in\{5,8\}$ |
| ID 测试粒子数 | $N\in\{5,8\}$ |
| OOD 测试粒子数 | $N\in\{10,20\}$ |
| 图结构 | 全连接引力边 + 稀疏弹簧边 |
| 状态 | 位置、速度、质量和粒子类型 |
| 轨迹长度 | 训练窗口 20 步，测试 rollout 500 步 |
| 训练池 | 256 条轨迹 |
| 验证集 | 64 条轨迹 |
| ID 测试集 | 128 条轨迹 |
| OOD 测试集 | 每个 OOD 条件 128 条轨迹 |

完整的冻结方法、力学方程、基线含义、OOD 划分和预注册判定标准见
[`docs/protocols/D3-Protocol.md`](protocols/D3-Protocol.md)。D3 开始前不再针对
粒子数据调整温度、MLP 强度或稀疏化方式；无关消息核抑制留到 D2-D5a 与 D3
均完成后统一优化。

OOD 设置：

- 未见粒子数量；
- 未见质量、弹簧刚度和阻尼范围；
- 未见弹簧拓扑；
- 全局旋转、平移和反射；
- 更长时间 rollout；
- 初始位置和速度分布变化。

几何 MLP 核必须采用旋转和平移等变参数化。主要领域基线采用 [EGNN](https://proceedings.mlr.press/v139/satorras21a)，纯神经图模拟基线参考 [Graph Network-based Simulator](https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html)。

### 3.4 D4：Missing-Kernel Stress Test

D4 不单独建立新数据集，而是在 D1–D3 上系统改变候选核库。

| 核库设置 | 操作 | 需要观察的行为 |
|---|---|---|
| Complete | 所有真实解析核均存在 | 解析核贡献较高，MLP 贡献较低 |
| Missing-One | 删除一个真实核 | MLP 补足缺失项，优于 No-MLP |
| Approximate | 用近似核替换真实核 | 近似核与 MLP 共同解释误差 |
| Distractor | 加入多个无关核 | 无关核实际贡献保持较低 |
| MLP-Only | 删除全部解析核 | 得到纯数据驱动性能下界 |
| Oracle | 提供真实核和真实参数 | 得到可达到的性能上界 |

局部缺失实验采用

$$
F(r)
=F_{\mathrm{known}}(r)
+\mathbf 1[r<r_0]F_{\mathrm{unknown}}(r).
$$

核库只包含 $F_{\mathrm{known}}$，检查 MLP 消息核的实际贡献是否集中在 $r<r_0$ 的区域。

## 4. 对比基线

### 4.1 精简后的统一基线

| 报告名称 | 历史编号 | 消息生成 | 聚合 | 节点更新 | 验证目的 |
|---|---|---|---|---|---|
| Oracle | B0 | 真实核与真实参数 | ADD | 真实核与真实参数 | 合成数据数值上界 |
| Learned-Single | B2L | 单个正确或验证集最优解析核，学习 $c_1$ | ADD | 单个解析核，学习 $c_2$ | 正确单核是否足够 |
| MLP-only | B6 | MLP 消息函数 | ADD | MLP 更新函数 | 纯数据驱动参考 |
| Physics+Residual | B7 | 解析核加非门控 MLP 残差 | ADD | 解析更新加 MLP 残差 | 与普通混合模型比较 |
| TaskKernel-GNN | B8 | 多解析核加 MLP，学习 $a,c_1$ | ADD | 多解析核加 MLP，学习 $b,c_2$ | 完整模型 |

Oracle 不训练，因此每个实验设置最多训练四个模型。B3 Multi-Msg、B4 Multi-Upd 和 B5 NoMLP 的既有结果只作为一次性结构诊断保留，不在后续每个数据集、数据预算和 OOD 条件上重复。NoMLP 仅在核库缺失实验中按需启用。人工指定和 Hybrid 作为 TaskKernel-GNN 的运行模式报告，不再作为所有实验的常规基线。

Physics+Residual 的训练形式可参考 [APHYNITY](https://openreview.net/pdf?id=kmG8vRXTFv)，但实现时应与 TaskKernel-GNN 使用相同的积分器、隐藏维度和数据。

### 4.2 领域基线

| 数据 | 领域基线 | 用途 |
|---|---|---|
| D1 可控恢复 | 稀疏回归/SINDy 风格基线 | 已知候选函数字典下的机制识别参考 |
| D2 反应扩散 | GRAND、FNO、U-Net、Physics+Residual | 图扩散、规则网格神经算子和物理补全比较 |
| D3 粒子动力学 | EGNN、GNS、标准 MPNN | 等变结构与纯数据驱动物理模拟比较 |

[GRAND](https://proceedings.mlr.press/v139/chamberlain21a.html) 将图网络解释为连续扩散过程，适合作为 D2 的图扩散基线。FNO 和 U-Net 只用于规则网格或 PDEBench 设置，不用于不规则图上的直接横向比较。

## 5. 消融实验

### 5.1 核组成消融

后续只保留两个直接消融：

| 消融 | 使用场景 | 对应问题 |
|---|---|---|
| Learned-Single vs TaskKernel-GNN | 所有主数据集 | 单核是否已经足够 |
| No-MLP vs TaskKernel-GNN | 仅核库缺失数据 | MLP 是否补足未知机制 |

消息端/节点端单独多核、均匀权重、干扰核和相关核不再作为常规消融；既有 D1、D2 结果保留，需要诊断具体问题时才重新启用。

### 5.2 参数学习消融

依次比较：

1. 固定 $c_1,c_2$，只学习 $a,b$；
2. 固定 $a,b$，只学习 $c_1,c_2$；
3. 联合学习 $a,b,c_1,c_2$；
4. 人工指定全部参数；
5. 固定已知参数，只学习未知参数；
6. 任务级共享参数与逐边/逐节点参数。

### 5.3 聚合设置

后续固定使用

$$
M_i=\operatorname{ADD}_{j\in\mathcal N(i)}m_{ij}.
$$

D2-A 已通过 ID、规模 OOD 和质量变化误差完成一次 ADD/MEAN 对照，结果支持叠加型物理消息使用 ADD。后续不再重复 ADD/MEAN 实验，也不在模型名称后添加 `-ADD`。只有新任务的物理定义明确要求邻域平均时才启用 MEAN。

### 5.4 Softmax 与核尺度消融

比较：

- 有无候选核尺度归一化；
- $\tau_a,\tau_b\in\{0.1,0.5,1.0,2.0\}$；
- 固定温度与退火温度；
- 有无 MLP 初始 logit 抑制；
- 有无 MLP 实际贡献正则；
- softmax 与 sparsemax/entmax 的可选扩展。

第一版主结果固定使用 softmax，其他归一化只作为附加分析。

## 6. 数据划分与小样本协议

### 6.1 小样本预算

从同一个训练池中构造嵌套子集：

$$
N_{\mathrm{train}}
\in\{1,2,4,8,16,32,64\}
$$

条独立轨迹。较大的训练预算可以增加 128 和 256 条轨迹。所有方法在同一预算下使用相同轨迹、相同时间点和相同观测变量。

数据效率的主指标为对数数据量坐标上的误差曲线面积：

$$
\operatorname{AUC}_{\mathrm{data}}
=\operatorname{Area}
\left(
\log N_{\mathrm{train}},
\operatorname{NRMSE}
\right).
$$

数值越小表示用更少数据达到更低误差。

### 6.2 划分规则

- 按完整轨迹划分训练、验证和测试集；
- 同一轨迹的相邻时间片不得跨集合；
- OOD 参数区间不得参与训练或超参数选择；
- 所有归一化统计量只由当前训练子集计算；
- 不同方法使用完全相同的数据索引和随机种子；
- 验证集只用于早停和超参数选择，测试集只在配置确定后评估。

## 7. 训练设置

### 7.1 默认训练策略

TaskKernel-GNN 采用三阶段训练：

1. 降低 MLP 核初始 logit，先学习解析核的 $a,b,c_1,c_2$；
2. 暂时固定解析分支，训练消息 MLP 核和更新 MLP 核拟合剩余误差；
3. 使用较小学习率联合微调全部参数。

同时增加端到端从头训练版本作为训练策略消融。

### 7.2 损失函数

基础损失为

$$
\mathcal L
=\lambda_1\mathcal L_{\mathrm{one\text{-}step}}
+\lambda_R\mathcal L_{\mathrm{rollout}}
+\lambda_M\mathcal R_{\mathrm{MLP}}
+\lambda_C\mathcal R_c
+\lambda_P\mathcal L_{\mathrm{physics}}.
$$

其中

$$
\mathcal L_{\mathrm{one\text{-}step}}
=\frac{1}{NT}
\sum_{i,t}
\|\widehat x_i^{t+1}-x_i^{t+1}\|_2^2,
$$

$\mathcal L_{\mathrm{rollout}}$ 为多步 rollout 误差，$\mathcal R_{\mathrm{MLP}}$ 限制神经核贡献，$\mathcal R_c$ 限制物理配置范围，$\mathcal L_{\mathrm{physics}}$ 根据任务选择守恒、对称或耗散约束。

### 7.3 公平比较

- 所有动力学模型使用相同的时间步长与积分器；
- 神经基线调整隐藏宽度，使参数量与完整模型处于同一量级；
- 每种方法使用相同的优化器、训练步数上限和早停规则；
- 每种方法获得相同数量级的超参数搜索预算；
- 主结果报告 5 个随机种子，可控恢复实验报告 10 个种子；
- 同时报告参数量、训练时间、单步推理时间和峰值显存。

## 8. 评价指标

### 8.1 预测指标

归一化均方根误差：

$$
\operatorname{NRMSE}(t)
=
\frac{
\|\widehat X_t-X_t\|_2
}{
\|X_t\|_2+\varepsilon
}.
$$

报告：

- one-step NRMSE；
- 10、20、50、100、200 步 rollout NRMSE；
- rollout 平均 NRMSE；
- 最终时刻 NRMSE；
- 误差首次超过给定阈值的有效预测时间。

### 8.2 核与参数恢复指标

当真实参数已知时，报告

$$
E_a=\|a-a^\star\|_1,
\qquad
E_b=\|b-b^\star\|_1,
$$

$$
E_c
=\frac{\|c-c^\star\|_2}{\|c^\star\|_2+\varepsilon}.
$$

核的实际贡献定义为

$$
C_{\mathrm{msg}}^{(p)}
=\mathbb E_{i,j,t}
\left[
\left\|
a_{ij}^{(p)}
K_{\mathrm{msg}}^{(p)}
\right\|_2
\right],
$$

$$
C_{\mathrm{upd}}^{(q)}
=\mathbb E_{i,t}
\left[
\left\|
b_i^{(q)}
K_{\mathrm{upd}}^{(q)}
\right\|_2
\right].
$$

同时报告：

- 正确核 Top-1/Top-2 识别率；
- 干扰核的总贡献；
- MLP 核的总贡献；
- 不同随机种子下权重的标准差；
- 局部缺失实验中，MLP 贡献在缺失区域内外的比值。

### 8.3 物理一致性指标

反应扩散任务根据方程设置报告总质量或浓度变化误差，以及非负状态违反率。若反应项本身包含源汇，则比较真实与预测的质量变化，而不是要求质量恒定。

粒子任务报告动量误差

$$
E_{\mathrm{momentum}}(t)
=\left\|
\sum_i m_iv_i(t)-
\sum_i m_iv_i(0)
\right\|_2,
$$

能量误差

$$
E_{\mathrm{energy}}(t)
=\frac{|E(t)-E(0)|}{|E(0)|+\varepsilon},
$$

以及等变误差

$$
E_{\mathrm{eq}}
=\frac{
\|f(T_gX)-S_gf(X)\|_2
}{
\|f(X)\|_2+\varepsilon
}.
$$

其中 $T_g$ 是施加在输入状态上的旋转、平移或反射，$S_g$ 是对应的输出变换。若模型输出位置，则平移作用于输出；若模型输出速度、力或状态增量，则只应用与该物理量对应的旋转或反射。

存在阻尼时，不使用能量守恒作为目标，而比较预测与真实能量衰减曲线。

## 9. RQ—数据—基线—结果表对应关系

| RQ | 数据 | 必须包含的基线/消融 | 主结果图表 |
|---|---|---|---|
| RQ1 | D1、D2、D3 | Learned-Single、MLP-only、TaskKernel-GNN | 多消息核与单核的 ID/OOD NRMSE；$a$ 恢复图 |
| RQ2 | D1、D2 | Learned-Single、MLP-only、TaskKernel-GNN | 状态增量与 rollout；$b$ 恢复图 |
| RQ3 | D1、D2、D3 | Learned-Single、MLP-only、Physics+Residual、TaskKernel-GNN | NRMSE–训练轨迹数曲线及 $\operatorname{AUC}_{\mathrm{data}}$ |
| RQ4 | D1-Missing、D4 | No-MLP、MLP-only、TaskKernel-GNN、Oracle | 完整/缺失核库表；MLP 局部贡献热图 |
| RQ5 | D1 | Learned-Single、TaskKernel-GNN 的参数控制 | 参数误差、实际贡献误差、种子稳定性 |
| RQ6 | D2、D3 | Learned-Single、MLP-only、Physics+Residual、TaskKernel-GNN | 图规模/参数/时间 OOD 表；长 rollout 曲线 |
| RQ7 | D2、D3 | TaskKernel-GNN（固定 ADD） | 度数变化下误差；守恒、动量和能量指标 |
| RQ8 | D1、D2、D3 | TaskKernel-GNN 的 Manual/Learned/Hybrid 三种运行模式 | 不同模式的精度与数据需求 |

## 10. 主论文结果组织

建议主文保留以下结果：

1. **主性能表**：D1、D2、D3 的 ID 与主要 OOD NRMSE；
2. **小样本曲线**：误差随训练轨迹数变化；
3. **核库完整性表**：Complete、Missing、Approximate、MLP-only；
4. **长期 rollout 图**：TaskKernel-GNN 与主要基线的时间误差；
5. **核贡献图**：解析消息核、解析更新核和 MLP 核的实际贡献；
6. **核心消融表**：Learned-Single、No-MLP（仅缺失核）、TaskKernel-GNN；
7. **物理一致性表**：守恒、动量、能量或等变误差。

其余温度、核尺度、噪声、边缺失、训练策略和完整参数范围放入附录。

## 11. 结论判定标准

不同 RQ 使用预先确定的主指标，避免只挑选有利结果：

- RQ1、RQ2：完整模型在多机制数据上的 rollout NRMSE 优于对应单核版本；
- RQ3：完整模型具有更低的 $\operatorname{AUC}_{\mathrm{data}}$；
- RQ4：缺失核时 Full 显著优于 No-MLP，完整核时 MLP 实际贡献保持较低；
- RQ5：在候选核可区分时，真实核贡献排名正确，并能恢复配置或有效系数；
- RQ6：完整模型在至少图规模、参数范围和时间长度三类 OOD 中保持优势；
- RQ7：默认 ADD 在总力/总通量任务上保持合理物理量和规模泛化；
- RQ8：Manual 接近 Oracle，Hybrid 用少于纯数据驱动模型的数据达到相近或更低误差。

所有主结果报告均值、标准差和 95% bootstrap 置信区间。模型间比较使用相同数据划分和随机种子进行配对统计；除统计显著性外，同时报告相对误差改善和物理指标变化。

## 12. 推荐实施顺序

按以下顺序实施可以最快发现模型设计问题：

1. D1-Single：验证前向计算、softmax 和参数恢复；
2. D1-Mixture/Distractor：验证多核选择；
3. D1-Missing：验证 MLP 补全；
4. D2 反应扩散：验证完整三阶段结构；
5. D3 多力场粒子：验证矢量叠加和几何泛化；
6. D4 系统压力测试；
7. PDEBench 与领域基线外部验证。

后续主实验只运行 Oracle（若可用）、Learned-Single、MLP-only、Physics+Residual 和 TaskKernel-GNN。逐数据集分别报告结果，总体平均只作为辅助信息。


