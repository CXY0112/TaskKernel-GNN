"""Publish E2 preparation summary only after real-data preflight passes."""
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/SOTA_Baseline/external/opencarp/preparation"


def main():
    receipt = json.loads((OUTPUT / "PROTOCOL_READINESS.json").read_text())
    expected = {"train": 16, "validation": 16, "id_test": 30, "mesh_ood": 30,
                "stimulus_ood": 16, "reference": 1}
    assert receipt["ready_for_formal_training"] and not receipt["missing"]
    assert receipt["generated_counts"] == expected
    assert receipt["optimizer_updates"] == 0
    names = {"taskkernel": "TaskKernel-GNN", "gno": "GNO", "meshgraphnet": "MeshGraphNet", "mp_pde": "MP-PDE"}
    files = receipt["files"]
    sizes = {split: sum(row["bytes"] for row in files if f"/generated/{split}/" in row["path"]) / 1048576
             for split in expected}
    descriptions = {
        "train": ("主训练", ".5", "150"), "validation": ("验证", ".5", "150"),
        "id_test": ("ID 测试", ".5", "150"), "mesh_ood": ("Mesh-OOD", ".2", "70"),
        "stimulus_ood": ("Stimulus-OOD", ".5", "150"), "reference": ("细网格参考", ".1", "50")}
    lines = ["# E2 实验准备完成报告", "", "**当前已达到正式训练条件；没有启动正式训练，优化器更新次数为 0。**", "",
             "任务为独立 openCARP 求解器生成的三维心肌长方体电传播，反应模型为 ten Tusscher–Panfilov 心外膜模型。正式训练需等待用户下一步指令。", "",
             "## 数据", "", "| 集合 | 条数 | 分辨率 mm | 时长 ms | NPZ 大小 MiB |", "|---|---:|---:|---:|---:|"]
    for split, count in expected.items():
        label, resolution, duration = descriptions[split]
        lines.append(f"| {label} | {count} | {resolution} | {duration} | {sizes[split]:.2f} |")
    lines += ["", f"共 **109 条**，压缩轨迹文件合计 **{sum(sizes.values()):.2f} MiB**。主实验用 16 条训练轨迹，原 128 条候选索引池不覆盖、不要求全部生成。ID 和 Mesh-OOD 的 30 对物理参数一致，细网格真值由独立重新求解获得。",
              "", "训练、验证、ID 与 Stimulus-OOD 文件集合互不重叠。刺激 OOD 测更强、更长的脉冲，刺激位置仍为角落；不将它称为刺激位置 OOD。", "",
              "## 模型与公共输入", "", "各模型均使用过去 8 帧电位、同样的传导率和刺激条件。自由推进不读未来真值。固定电位归一化、mm/ms 单位转换和各向异性 FEM 适配器已接入。", "",
              "| 模型 | 参数量 | 真实粗/细网格检查 | 10 步损失前向+反向秒数 |", "|---|---:|---|---:|"]
    for key, row in receipt["models"].items():
        assert all(check["forward_backward_short_rollout_finite"] for check in row["checks"].values())
        lines.append(f"| {names[key]} | {row['parameters']:,} | 通过 | {row['training_window_forward_backward_seconds']:.2f} |")
    lines += ["", "上表耗时为本机 CPU、2 线程、一个 10 步训练窗口的预检实测，包含反向传播、不含优化器更新；不是正式训练完成时间，也不是严格独占资源的推理基准。本机没有可用 CUDA。", "",
              "三个基线是项目已有架构的三维历史条件适配，不能写成作者官方实现的完整原样复现。TaskKernel-GNN 使用解析扩散先验和自适应内部子步，细网格计算开销必须另外报告。", "",
              "## 冻结配置", "", "种子 **0、42、37**；ADD；温度 **0.1**；熵惩罚 **0**；MLP 强度 **0.1**；对称性关闭；隐藏宽度 64。解析消息核只用训练集 RMS 校准，MLP 保持原强度；节点整体校准倍率保持 1，以保留各节点核共同的聚合传输项。", "",
              "TaskKernel-GNN 消息候选为各向异性 FEM 扩散、高斯调制 FEM 扩散、条件连续系数 MLP；节点候选为线性反应、多项式反应、历史条件 MLP。后两者近似未知离子动力学，不声称与 ten Tusscher–Panfilov 的全部内部状态严格等价。", "",
              "学习率 0.001，最多 20,000 次更新，10 步训练窗口；验证选优与早停规则在新版 YAML 中。粗网格训练时间窗口预检约数秒，单种子满预算会达到小时级；先训练 TaskKernel-GNN 后按实测收敛和耗时再安排基线。", "",
              "## 已通过的检查", "", "- 官方四个输入文件 SHA256 一致，环境与独立求解器可用。", "- 109 条文件的形状、空间范围、网格分辨率、时间轴、单位、生成参数、有限性与 SHA256 已核验。", "- 划分隔离、30 对网格 OOD 参数匹配、训练集专属校准。", "- 代表性粗/细网格的常量场零扩散、质量守恒、扩散能量耗散检查。", "- 四个模型在训练、验证、ID、Mesh-OOD、Stimulus-OOD 代表案例上的前向、梯度和短程自由推进；三个 TaskKernel 种子的接口检查。", "- 每个模型正式 10 步损失的反向传播检查，参数保持不变，未创建训练 checkpoint。", "",
              "这些检查确认数据与训练接口可用，**不构成模型精度或 OOD 优势的实验结论**。", "",
              "## 评价与图像", "", "分别报告 ID、Mesh-OOD、Stimulus-OOD 的 1/10/50 步电位 MSE、RMSE、相对 L2，以及激活时间误差、漏激活和伪激活比例。主 LAT 使用相同输出间隔的电位计算，native LAT 仅作诊断；没有共同激活节点时不将误差填为 0。", "",
              "以下是独立求解器生成的参考场，尚无训练后的模型预测。", "", "![独立 openCARP 三维传播参考](reference_3d.png)", "",
              "0.1 mm 参考包含 442,401 节点，全部节点在约 40.7 ms 内首次激活。它不进入训练或 checkpoint 选择。本阶段检验去极化传播，不报告需要完整复极周期的 APD90。", "",
              "## 文件与下一步", "", "- [完整实验协议](PROTOCOL.zh-CN.md)", "- [就绪记录与文件哈希](PROTOCOL_READINESS.json)", "- 新配置：`experiments/paper/configs/external/opencarp_prepared.yaml`", "- 新训练入口：`experiments/paper/train_opencarp_prepared.py`", "- 评价入口：`experiments/paper/evaluate_opencarp_prepared.py`", "- 图像/左右对照动画入口：`experiments/paper/plot_opencarp_prepared.py`", "",
              "旧单帧 E2 训练入口已禁用；新入口必须显式指定训练执行，并检查数据、配置与实现哈希。**准备工作在此结束，等待用户指令再启动 TaskKernel-GNN 正式训练。**", ""]
    (OUTPUT / "README.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")
    path = OUTPUT / "WSL_INSTALL_STATUS.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state.update(environment_ready=True, ready_for_formal_training=True, reference_files_generated=109,
                 reference_split_counts=expected, optimizer_updates=0, training_started=False,
                 frozen_training_trajectories_generated=16,
                 next_step="Wait for user instruction before formal TaskKernel-GNN training",
                 next_steps=["Wait for user instruction before formal TaskKernel-GNN training"],
                 prepared_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_path = OUTPUT / "READINESS.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    original = json.loads((ROOT / "data/external/opencarp/generated/run_plan.json").read_text())
    audit.update(ready_for_formal_training=True, training_started=False, optimizer_updates=0,
                 primary_training_files_present=16,
                 planned_files_present=sum((ROOT / "data/external/opencarp/generated" / item["output_npz"]).is_file()
                                           for item in original["runs"]),
                 prepared_protocol="PROTOCOL_READINESS.json", prepared_protocol_file_count=109,
                 legacy_scalar_training_entry_disabled=True, remaining_requirements=[])
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ready": True, "files": 109, "MiB": sum(sizes.values()), "optimizer_updates": 0}))

if __name__ == "__main__":
    main()
