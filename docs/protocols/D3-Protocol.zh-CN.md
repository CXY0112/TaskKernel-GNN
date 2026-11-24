[English](D3-Protocol.md)

# D3 混合力粒子动力学实验协议

## 1. 目标

D3 用于检验 TaskKernel-GNN 在标量反应扩散动力学中得到的结论，能否迁移到矢量值、二阶粒子动力学。进入 D3 前冻结门控方案：解析核 RMS 校准、退火温度 softmax、`eta=0.1`、ADD 聚合，并且不针对 D3 调整门控。只有确认 D3 的通用性结果后，才研究如何进一步压低无关核。

`Physics+Residual` 被视为已经选出相关解析力集合的理想化 TaskKernel-GNN。完整 TaskKernel-GNN 应保持稳定，并优于纯 MLP 和最佳单力模型。若它仍落后于 `Physics+Residual` 或 Learned-True-Set，则差距可用于判断无关候选核的干扰。

## 2. 真实动力学

孤立二维系统在每一对无序粒子之间施加引力，并在稀疏无向连接图上施加弹簧力和阻尼力。定义
`r_ij = r_j-r_i`、`d_ij=||r_ij||`、`rhat_ij=r_ij/d_ij`：

$$
F_{i\leftarrow j}^{\mathrm{grav}}
=Gm_im_j\frac{r_{ij}}{(d_{ij}^2+\epsilon^2)^{3/2}},
$$

$$
F_{i\leftarrow j}^{\mathrm{spring}}
=\mathbf 1[(i,j)\in E_s]k_{ij}(d_{ij}-\ell_{ij}^0)\widehat r_{ij},
$$

$$
F_{i\leftarrow j}^{\mathrm{dashpot}}
=\mathbf 1[(i,j)\in E_s]\gamma_{ij}
\big[(v_j-v_i)\cdot\widehat r_{ij}\big]\widehat r_{ij}.
$$

每个成对作用都以相反符号施加到粒子 (j)。三种消息机制均保持总线性动量。成对阻尼会耗散机械能，因此能量应随时间下降，而不是保持恒定。

节点更新采用半隐式 Euler：

$$
v_i^{t+1}=v_i^t+\Delta t\frac{F_i^t}{m_i},
\qquad
r_i^{t+1}=r_i^t+\Delta t\,v_i^{t+1}.
$$

参考时间步由数值预检扫描确定，随后冻结。初始位置满足最小间距约束，初始总动量被移除，质心平移到原点。系统没有墙壁或外力。

## 3. 候选核

消息候选包括软化平方反比引力、Hooke 弹簧、成对阻尼、高斯径向干扰核和 E(2) 等变 MLP 力。MLP 从旋转平移不变量的成对特征预测标量系数，再乘以等变矢量基。成对输出进行反对称化，使 `F_ij=-F_ji`。该任务启用可选的作用力—反作用力对称约束；心脏实验中该约束保持关闭。

节点更新候选包括半隐式 Euler、显式 Euler、错误的节点阻尼更新，以及保持静止状态的 MLP 更新。实验只使用 ADD 聚合。

## 4. 模型

| 模型 | D3 中的含义 |
|---|---|
| Oracle | 使用真实力定律、参数、连接图和积分器 |
| Learned-True-Set | 使用三个正确解析消息核和正确更新核，学习配置修正 |
| Learned-Single | 由验证集选择一个力核，无法表达完整混合机制 |
| MLP-only | 不含解析力核的等变神经消息与神经更新 |
| Physics+Residual | 三种近似解析力加有界残差，是理想化精简 TaskKernel |
| TaskKernel-GNN | 正确力、干扰力和 MLP 候选通过温度 softmax 自适应组合 |

只有回答完受控核选择问题后，才加入外部 EGNN/GNS 基线，避免架构差异掩盖当前要验证的机制。

## 5. 数据划分

| 划分 | 粒子数 | 参数/拓扑 |
|---|---:|---|
| 训练集 | 6、8 | 标称质量、力、阻尼和稀疏连接范围 |
| 验证集 | 6、8 | 同分布、不重叠轨迹 |
| ID 测试 | 6、8 | 同分布、不重叠轨迹 |
| Size-OOD | 12、20 | 标称参数，匹配连接平均度分布 |
| Parameter-OOD | 8 | 未见质量、(G)、弹簧刚度、原长和阻尼 |
| Topology-OOD | 8 | 训练未出现的链、环和分支连接模式 |
| SE(2)-OOD | 8 | ID 状态的整体旋转、平移和反射 |
| Initial-state OOD | 8 | 更宽的位置与速度分布 |
| Long-rollout OOD | 8 | 至少两倍训练预测时域 |

生成器首先产生确定性的预检数据。正式实验采用不重叠随机种子、至少三个训练种子，并分别报告每一种 OOD 条件。

## 6. 指标

- 位置和速度 rollout NRMSE；
- 加速度/力误差；
- 稳定预测时域、爆炸率和碰撞率；
- 总动量漂移；
- 机械能曲线误差和错误能量增加率；
- 相连与不相连粒子对的距离误差；
- E(2) 等变误差；
- 最终门控权重、校准后的候选贡献，以及干扰核与 MLP 的合计贡献。

## 7. 预注册判断标准

满足以下全部条件时，D3 支持跨领域通用性：

1. `Learned-True-Set` 和 `Physics+Residual` 稳定且明显优于 `MLP-only`，说明解析力核能够替代不稳定或数据需求较大的神经模拟器；
2. TaskKernel-GNN 在 ID、Size-OOD、Topology-OOD 和 SE(2)-OOD 中保持有限，并在 ID 和至少三个 OOD 维度上同时优于 MLP-only 与 Learned-Single；
3. 三个正确消息核获得最大的合计校准贡献，并选中正确的二阶更新；
4. 如果 TaskKernel-GNN 落后于 Physics+Residual，但二者稳定性和 OOD 趋势一致，只有当干扰核/MLP 贡献非零且跨种子可复现时，才把差距归因于候选核干扰。

完成冻结方法测试后，才将 D3 与 D2-D5a 共同用于优化无关消息核抑制方法。

## 8. 预检后冻结的数值设置

预检将生成器设置固定为：`G=0.002`、弹簧刚度中心 `0.8`、成对阻尼中心 `0.18`、引力软化长度 `epsilon=0.10`、半隐式 Euler 时间步 `dt=0.01`。正式训练轨迹覆盖物理时间 `4.0`，长时 OOD 覆盖 `8.0`。初始最小间距为 `0.34`，质心和总动量均为零。

40 条独立的粒子数、拓扑和长时预检轨迹全部有限，且均高于 `0.075` 的近距离阈值。总线性动量和 E(2) 力等变性达到数值精度，所有轨迹的机械能均下降。详细结果见 `results/Ablation/d3/particles/preflight/REPORT.zh-CN.md`。

## 9. 冻结正式实验状态

三种子受控比较已经完成。TaskKernel-GNN 在 ID 和九个单独报告的 OOD 数据集上均保持有限，并优于 MLP-only 和验证集选择的 Learned-Single。三个正确消息核获得 `0.6600 +/- 0.0039` 的合计权重，高斯干扰核与 MLP 仍保留 `0.3400 +/- 0.0039`。正确的半隐式节点更新获得 `98.98%` 权重。Physics+Residual 在所有划分上仍然更好，因此当前差距被定位在消息候选干扰，而不是节点更新选择。详细指标与图片见 `results/Ablation/d3/particles/formal/REPORT.zh-CN.md`。

