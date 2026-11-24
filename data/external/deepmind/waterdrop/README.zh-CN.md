# DeepMind WaterDrop 紧凑数据

该目录保存 External E3 使用的官方 WaterDrop 数据。为检验 TaskKernel-GNN 的少数据学习能力，本地训练文件只保留官方训练 TFRecord 中前 128 条完整轨迹；验证集和测试集保留官方全部 30 条轨迹。

## 数据文件

- `train_128.tfrecord`：128 条训练轨迹。
- `valid.tfrecord`：官方完整验证集，30 条轨迹。
- `test.tfrecord`：官方完整测试集，30 条轨迹。
- `metadata.json`：官方归一化统计、边界、维度和连接半径。
- `splits.json`：训练轨迹的冻结少样本索引。
- `manifest.json`：来源、文件大小、MD5 和 TFRecord 边界验证结果。

少样本集合使用固定随机种子 `20260917`，并按 `4/8/16/32/64/128` 严格嵌套。所有模型必须使用相同索引；验证集和测试集不参与训练子集选择。
