# 从零环境端到端验证报告

- 开始: 2026-09-16 20:07:49
- 结束: 2026-09-16 20:12:54
- 模式: **quick**   起始阶段: PRE   训练步数: 100
- Skill 根: `/root/qwen35-ascend-migrator`
- 数据目录: `/root/data`   MindSpeed-MM: `/root/MindSpeed-MM`

## 裁决表

| 阶段 | 名称 | 结果 | 耗时(s) | 日志 |
|---|---|---|---|---|
| PRE | 预检能力矩阵 | **PASS** | 25 | [out/logs/PRE.log](out/logs/PRE.log) |
| P0 | 环境探测 | **PASS** | 35 | [out/logs/P0.log](out/logs/P0.log) |
| P1 | 迁移点识别(AST) | **PASS** | 0 | [out/logs/P1.log](out/logs/P1.log) |
| P2 | 迁移方案与配置生成 | **PASS** | 0 | [out/logs/P2.log](out/logs/P2.log) |
| P3 | 算子与契约验证 | **PASS** | 8 | [out/logs/P3.log](out/logs/P3.log) |
| P4 | 权重与数据准备 | **PASS** | 3 | [out/logs/P4.log](out/logs/P4.log) |
| P5 | 训练执行(100 步) | **PASS** | 220 | [out/logs/P5.log](out/logs/P5.log) |
| P6 | 性能基准(算子微基准 + 50-100 窗口口径) | **PASS** | 9 | [out/logs/P6.log](out/logs/P6.log) |
| P7 | 精度判定(P7) | **PASS** | 4 | [out/logs/P7.log](out/logs/P7.log) |

**通过 9 · 失败 0 · 超时 0 · 复用已有产物 0**

## ★ 降级登记（不许悄悄降级）

账本共 **5** 条：**本轮 1 条** / **陈旧·非本轮 4 条**（详见 `out/degradations.json`）：

- **package_manager** [degraded] ts=2026-09-16T13:18:06 — 无可用包管理器，系统依赖需手工安装 ⚠**陈旧（上一轮遗留，非本轮事实）**
- **required_commands** [degraded] ts=2026-09-16T13:18:06 — 缺少基础命令: grep,sed,awk ⚠**陈旧（上一轮遗留，非本轮事实）**
- **torch_npu_unavailable** [degraded] ts=2026-09-16T13:18:06 — torch_npu 不可用 ⚠**陈旧（上一轮遗留，非本轮事实）**
- **data_comparability** [degraded] ts=2026-09-16T13:18:06 — 数据未达全量同源（llava=False converted=False coco=0/118287） ⚠**陈旧（上一轮遗留，非本轮事实）**
- **cann_prerelease** [degraded] ts=2026-09-16T20:07:55 — CANN 9.1.0-beta.3 为预发布版

> ⚠ **账本含 4 条陈旧条目**（ts 早于本轮开始 2026-09-16 20:07:49）：package_manager(2026-09-16T13:18:06),required_commands(2026-09-16T13:18:06),torch_npu_unavailable(2026-09-16T13:18:06),data_comparability(2026-09-16T13:18:06)
> 它们由**上一次** PRE 阶段写入，本轮续跑（`--from`）不会重写账本 →
> **复核时不得当作当期降级事实**（坑 118：陈旧账本比空账本更有害）。
> 如需当期口径，请跑 `--from PRE` 重新预检。

## P0 环境档案

```json
{
 "schema": "migrator_env.v1",
 "probed_at": "2026-09-16T20:08:14.223219+00:00",
 "hardware": {
  "npu_smi_raw_ok": true,
  "total_count": "1",
  "npu_id": "5",
  "chip_count": "2",
  "chip_hw_name": "Ascend910",
  "board_id": "0xb1",
  "chip_model": "9382",
  "davinci_nodes": [
   "/dev/davinci10",
   "/dev/davinci11"
  ],
  "driver_version": "25.5.1",
  "torch_npu_device_count": "2",
  "torch_npu_device_name": "Ascend910_9382",
  "hbm_source": "npu_smi_table",
  "hbm_mb_per_chip": 65536
 },
 "software": {
  "python": "3.11.6",
  "pip": "23.3.1",
  "cann_dirs": [
   "/usr/local/Ascend/ascend-toolkit/latest",
   "/usr/local/Ascend/cann-9.1.0-beta.3"
  ],
  "cann_version": "9.1.0",
  "torch": "2.7.1+cpu",
  "torch_npu": "2.7.1.post10",
  "transformers": "5.2.0",
  "triton": "3.2.0",
  "numpy": "1.26.4",
  "yaml": "6.0.3",
  "triton_ascend_backend": "ok",
  "triton_backend_err": "",
  "triton_npu_kernel": "ok",
  "triton_npu_kernel_err": "",
  "mindspeed_mm_dir": "/root/MindSpeed-MM",
  "mindspeed_mm_tag": "v26.1.0",
  "mindspeed": "unknown"
 },
 "prereq": {
  "python_dev_header": "/usr/include/python3.11/Python.h",
  "python3_dev_ok": true,
  "build_essential_ok": true,
  "apt_ok": false
 },
 "resources": {
  "disk_free_gb_root": 266.6,
  "disk_free_gb_data": 266.6,
  "net_modelscope": true,
  "net_obs": true,
  "net_gitcode": true
 },
 "path": "full_npu",
 "degraded": false,
 "degrade_reasons": [],
 "capabilities": {
```

## 失败/超时阶段的关键日志（最后 40 行）

## 数据可比性声明（诚实红线）

> ⚠️ **本次为 quick 模式（`--skip-coco`）：COCO 图片未全量下载，数据可比性降级。**
> 结论只能表述为「流程连通性已验证」，**不可**宣称精度已对齐官方。
> 需改用 `--full` 重跑 P4 及之后阶段。
