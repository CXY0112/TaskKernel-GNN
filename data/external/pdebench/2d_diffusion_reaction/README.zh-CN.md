# PDEBench 二维反应扩散数据

此目录保存 External E1 使用的 PDEBench 官方 `2D_diff-react_NA_NA.h5`。原始 HDF5 不纳入版本控制；`manifest.json` 记录来源、文件校验值和实际结构，`splits.json` 冻结实验索引。

## 固定划分

- `train_pool`：700 条，仅供训练候选使用。
- `fewshot_4/8/16/32/64/128`：从同一个随机顺序截取，因此严格嵌套。
- `validation`：100 条，只用于模型选择和早停。
- `test`：200 条，只用于最终报告，禁止用于调参。
- 随机种子：`20260917`。

索引均指向按字典序排列的 HDF5 顶层样本键。可使用 `taskkernel_gnn.data.PDEBenchDiffusionReaction` 惰性读取，`spatial_stride=1` 对应原生分辨率，`spatial_stride=2` 对应 `64×64` 训练视图。

## 复现准备

```powershell
python tools/prepare_pdebench_diffusion_reaction.py data/external/pdebench/2d_diffusion_reaction/2D_diff-react_NA_NA.h5
```

该命令会先检查官方字节数和 MD5，再检查 1,000 个样本的数据结构并重建两个 JSON 清单。它只准备数据，不训练模型。
