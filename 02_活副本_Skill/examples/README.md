# examples/ —— 真实运行示例产物

本目录是一台 **1× 昇腾 910C 双 die**（Ascend910_9382，CANN 9.1.0）上运行本 Skill 的**真实产物**，
供评委开箱查看"产物形态 + 真实数据"，也可用于离线验证判定链（P7 可 0-GPU 复跑）。

## 内容

| 文件 | 说明 |
|---|---|
| `train/train.log` | P5 训练日志（100 步，从程序启动开始；交付物 #5 数据源） |
| `train/train_config.yaml` | P2 生成的最终训练配置（官方几何 dp2/mbs4/gas1） |
| `train/official_baseline.log` | 官方 7/24 基线日志（精度对比基准） |
| `bench/round_1.json` | P6 性能数据（50-100 步：中位 419.0ms / 19.09 samples/s） |
| `judge/fp.json` | P7 配置指纹（7+2 维，mismatch=[] 零偏离） |
| `judge/obs.json` | P7 实测指纹（step1 loss 1.924621、triton 生效、0 回退） |
| `judge/verdict.json` | P7 裁决（level=pointwise_feasible，五态=NEEDS_EVIDENCE） |
| `judge/registry.json` | 哈希链账本（5 条，REGISTRY_VERIFY_OK） |

## 离线复跑判定链（0-GPU）

```bash
python3 scripts/70_judge.py \
    --log examples/train/train.log \
    --baseline officialB \
    --baseline-log examples/train/official_baseline.log \
    --out out/judge_example
```

预期输出：

```
逐点 loss : Mean 0.0546% | Max 0.1592% | 超2%步数 0/100
逐点 gn   : Mean 0.1182% | Max 0.8643% | 超2%步数 0/100
step1 loss: 我方 1.924621 vs 官方 1.924621
窗口50-100: loss 偏差 0.0131% | gn 偏差 0.0366%
verdict   : NEEDS_EVIDENCE   level: pointwise_feasible
```

> **注**：`verdict_id` 由输入文件字节哈希决定，在不同路径下重跑会得到不同的 id（纯函数设计，
> 保证同证据必得同 id），但 `level` 与 `verdict` 结论一致。
> 未闭合证据门为 `data_identity`（官方数据 sha256 未公开），故按诚实红线不宣称绿档。
