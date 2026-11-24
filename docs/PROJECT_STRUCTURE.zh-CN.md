[English](PROJECT_STRUCTURE.md)

# 项目结构

## 源码

```text
src/taskkernel_gnn/
├── core/
│   ├── aggregation.py     ADD/MEAN 邻域聚合
│   ├── calibration.py     固定 RMS 候选核校准
│   ├── gating.py          退火温度 softmax
│   └── layer.py           消息 → 聚合 → 更新主层
├── kernels/
│   ├── message/           消息生成核，第二层按领域划分
│   │   ├── common.py      差分、图扩散与距离核
│   │   ├── transport.py   波传播与有向输运
│   │   ├── mechanics.py   引力、弹簧、阻尼及粒子辅助算子
│   │   ├── cardiac.py     心脏 FEM 离散适配器
│   │   └── neural.py      边门控与 MLP 消息
│   ├── update/            节点更新核，第二层按领域划分
│   │   ├── common.py      恒等、线性与通用 Euler 更新
│   │   ├── reaction.py    通用局部反应与阻尼
│   │   ├── mechanics.py   二阶力学积分
│   │   ├── wave.py        波动增广状态更新
│   │   ├── cardiac.py     AP、FHN 与心脏衰减
│   │   └── neural.py      GRU 与 MLP 更新
│   ├── base.py            共用张量与参数辅助函数
│   ├── registry.py        供实验组装使用的具名核工厂
│   └── README.zh-CN.md    核库结构与统一接口说明
├── models/
│   ├── taskkernel.py      可堆叠 TaskKernel-GNN
│   └── variants.py        实验变体 Learned-Single、MLP-only、Physics+Residual
└── data/
    ├── io.py              NPZ 与指标表加载
    ├── cardiac.py         四面体心脏网格类型化加载器
    └── particle.py        粒子系统保存与加载接口
```

`src/` 保存可复用核、门控、主层、工厂、模型和数据接口。实验目录包含数据集构造、训练循环和评估逻辑。与源码镜像的中文核函数手册位于 `docs/kernel-manual/`。

## 实验

```text
experiments/
├── paper/                    PDEBench, openCARP, ISO17
├── d1_scalar/                受控机制研究
├── d2_reaction_diffusion/    Ring、Grid、RGG 图消融
├── d3_particle_dynamics/     混合力粒子研究
└── README.md                 可运行入口
```

## 结果与媒体

运行实验时，结果写入本地 `results/`。仓库首页的三个图例存放在 `visualizations/examples/`。
