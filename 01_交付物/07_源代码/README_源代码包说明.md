# migrated_qwen35_src —— 迁移后源代码包

> C4-AI 2026 昇腾赛道复赛 · Qwen3.5-0.8B 昇腾迁移（1× 昇腾 910C 双 die）
> **迁移方式：配置驱动 + 环境适配（未修改 MindSpeed-MM 框架源码）**——详见 [MIGRATE_NOTES.md](../MIGRATE_NOTES.md)

## 目录结构

```
migrated_qwen35_src/
├── 01_training_config/          训练配置（迁移核心产物）
│   ├── qwen3_5_0_8B_optimized.yaml   ★ 最终验收配置（官方几何 + 8 项优化）
│   └── qwen3_5_0_8B_mock_config.yaml 流程连通性验证配置（mock 数据）
├── 02_launch_scripts/           运行脚本
│   ├── train_100steps.sh            ★ 最终验收训练（含全部必需环境变量）
│   └── selfcheck_env.py             环境自检（0-GPU 可跑）
├── 03_data_pipeline/
│   └── prepare_data.sh              数据下载 + 官方转换脚本（含字节校验锚点）
├── 04_weight_convert/
│   └── convert_hf_to_dcp.sh         HF 权重 → DCP 格式转换
├── 05_judge/                    精度与配置判定链（0-GPU 可复核）
│   ├── scripts/                     配置指纹 / 实测指纹 / 五态裁决 / 哈希链账本 / 数据身份
│   └── configs/baselines/           官方基线声明指纹（含出处）
└── 06_original_poc/             初赛阶段：AST 迁移点识别 POC（迁移范围的程序化界定）
```

## 快速使用

```bash
python3 02_launch_scripts/selfcheck_env.py                      # 环境自检
bash 03_data_pipeline/prepare_data.sh /root/data                # 数据准备
bash 04_weight_convert/convert_hf_to_dcp.sh                     # 权重转换
bash 02_launch_scripts/train_100steps.sh                        # 训练 100 步
python3 05_judge/scripts/fingerprint_cfg.py --config-from-log <log> --baseline officialB
```

## 关键结果（可复核）

| 指标 | 结果 |
|---|---|
| 第 1 步 loss | **1.924621** = 官方 1.924621（偏差 0.000000%） |
| 100 步逐点超 2% 步数 | loss **0/100**、grad_norm **0/100** |
| 50–100 步性能 | 中位 **419.0 ms**（官方 431.3 ms）；19.09 samples/s |
| 单步加速 | 21,393 ms → 419.0 ms（**51.1×**） |
| 配置指纹偏离 | **0 维**（7+2 维全部一致），级别 `pointwise_feasible` |

## 依赖版本

CANN 9.1.0 · torch 2.7.1 · torch_npu 2.7.1.post10 · triton-ascend **3.2.2** ·
transformers 5.2.0 · MindSpeed-MM v26.1.0 · MindSpeed 0.12.1 · Python 3.10
（系统需 `python3-dev build-essential`）
