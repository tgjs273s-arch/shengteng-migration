# qwen35-ascend-migrator · 使用说明

> 昇腾 NPU 自动迁移·调优·判定 Skill。完整能力与设计见 [SKILL.md](SKILL.md)。
> **本 Skill 已在真实环境验证**：1× 昇腾 910C 双 die（Ascend910_9382，CANN 9.1.0）
> 完成 Qwen3.5-0.8B 迁移，100 步训练达成精度逐位对齐（step1 loss 偏差 0.000000%）
> 与性能 51× 优化（50–100 步中位 419ms，优于官方基线 431.3ms）。

---

## 快速上手（8 阶段流水线）

```bash
# ---------- P0 环境探测（决定整条路径 + 配置档）----------
python3 scripts/00_probe_env.py
#   产出 out/probe/env.json；打印执行路径（full_npu / single_die / cpu_only ...）与推荐配置档

# ---------- P1 迁移点识别（AST 静态分析）----------
python3 scripts/10_analyze_points.py refs/modeling_qwen3_5__transformers_v5.2.0.py
#   产出 out/analyze/migrate_points.json + migrate_points_report.md

# ---------- P2 迁移方案与配置生成（按环境自适应）----------
python3 scripts/20_plan_migration.py --env out/probe/env.json \
    --data-json /root/data/output_llava_coco_data.json --data-dir /root/data/coco
#   产出 out/plan/train_config.yaml（可直接用于 P5）+ out/plan/migrate_plan.md

# ---------- P3 算子与契约验证 ----------
python3 scripts/30_verify_ops.py
#   产出 out/verify/ops_matrix.json（无卡时仅证 shape 契约，如实标注 contract-only）

# ---------- P4 权重与数据准备（含字节校验锚点）----------
python3 scripts/40_prepare_assets.py --data-dir /root/data
#   产出 out/assets/assets.json；校验 llava json = 228,941,895 B、转换产物 157,712 样本

# ---------- P5 训练执行（环境自适应 + 故障诊断）----------
python3 scripts/50_train.py --config out/plan/train_config.yaml --log out/train/train.log
#   产出完整训练日志（从启动开始）；失败时自动诊断并给出处置

# ---------- P6 性能基准（官方口径 = 50-100 步）----------
python3 scripts/60_bench.py --log out/train/train.log --gbs 8 --round 1 --tag baseline
#   产出 out/bench/round_1.json（耗时均值/中位 + samples/s）

# ---------- P7 判定（配置指纹 + 实测指纹 + 五态裁决 + 账本）----------
python3 scripts/70_judge.py --log out/train/train.log --baseline officialB \
    --baseline-log /path/to/official_baseline.log \
    --registry evidence/registry.json --tag run_001
#   产出 out/judge/{fp,obs,verdict,judge_summary}.json + loss_compare.csv
```

**自检（0-GPU 可跑，评委入口）**：
```bash
python3 selfcheck.py        # L1 结构 + L2 判定链 + L3 环境一致性 → SELFCHECK_OK
```

---

## 环境要求

| 项 | 要求 | 说明 |
|---|---|---|
| 硬件 | 昇腾 NPU（910B4 / 910C 单 die / 910C 双 die） | **无 NPU 也可运行**（走 CPU 轨，见下） |
| CANN | 9.1.0（其他版本按版本矩阵适配） | 需 `source set_env.sh` |
| torch / torch_npu | 2.7.1 / 2.7.1.post10 | 与 CANN 配对 |
| triton-ascend | **3.2.2**（910C + CANN9.1 + torch2.7.1 + py3.10） | 3.2.0/3.2.1 会回退 CPU |
| transformers | 5.2.0 | MindSpeed-MM v26.1.0 配套 |
| MindSpeed-MM | v26.1.0 | 训练框架 |
| 系统包 | `python3-dev build-essential` | **必需**：triton 驱动需现场编译 C++ 桥接 |

版本全量清单：`config/versions.lock`（42 项）。

---

## 冗余与降级（跨环境可用）

本 Skill 的每个单点失败都有替代路径，**绝不因环境不支持而中断流水线**：

| 维度 | 降级行为 | 标注 |
|---|---|---|
| 无 NPU | P1/P2 照常；P3 仅证 shape 契约；P5/P6 跳过；**P7 判定链完整可跑** | `path=cpu_only`, `degraded=true` |
| 仅 1 个 die | 训练可跑，但无法复刻官方 dp2 几何 | `path=single_die`，仅窗口口径可比 |
| triton 不可用 | 算子后端降级链 triton → ascendc → eager（三者数值等价已验证 Δ<0.2%） | `path=*_triton_off` |
| 32GB 显存档 | 自动开 recompute/chunk_loss/activation_offload，mbs 上限 2 | `profile=910b4_low_mem` |
| 数据不全 | 降级为子集/mock | `data_comparability` 字段标注等级 |
| 验收口径不明 | 同时输出逐点（4 指标）与窗口（50-100）两套指标 | 双轨齐备 |

**运行时间不敏感**：允许 P0 多轮探测与版本试装、P1-P7 分步执行与断点续跑，优先保证完备性。

---

## 配置与数据

| 路径 | 内容 |
|---|---|
| `config/env_matrix.yaml` | ★ 环境矩阵：执行路径 / 芯片档位 / 版本矩阵 / 资产锚点 / 必需环境变量 / 可比性等级 |
| `config/versions.lock` | 验收环境版本锁（42 项，含数据/权重/配置/env） |
| `config/baselines/official{A,B}.yaml` | 官方基线声明指纹（含出处行号） |
| `config/templates/qwen3_5_0_8B_base.yaml` | 训练配置模板（P2 按环境改写生成） |
| `refs/modeling_qwen3_5__transformers_v5.2.0.py` | P1 AST 分析的目标源码 |
| `docs/PITFALLS_坑表.md` | ★ 实测踩坑与修复（含 910C triton 3.2.2 修复链） |
| `docs/SK04_判定链说明.md` | 判定链设计（配置指纹 / 可达性谓词 / 五态裁决 / 哈希链） |

---

## 产物与交付物映射

| 产物 | 对应复赛交付物 |
|---|---|
| `out/train/train.log` | #5 原始日志（从程序启动开始的完整日志） |
| `out/bench/round_<n>.json` | #4 性能测试报告 |
| `out/judge/*.json` + `loss_compare.csv` | #3 精度分析报告 |
| `out/analyze/migrate_points_report.md` | #1 创意书技术路线 / #2 README 代码逻辑 |
| `out/plan/migrate_plan.md` + `train_config.yaml` | #7 迁移后源码的生成依据 |

---

## 常见问题

**Q: triton 报 "roll back to CPU"？**
A: 三步排查——① 版本配对（910C+CANN9.1+torch2.7.1+py3.10 → 3.2.2）；② `apt install python3-dev build-essential`；
③ 重跑 `00_probe_env.py` 确认 backend/kernel 状态。详见坑表 19/22。

**Q: 训练报 `No module named 'megatron'`？**
A: 缺 `NON_MEGATRON=true`（`50_train.py` 已自动注入；手工运行需补）。

**Q: 慢步（周期性 3.7s）？**
A: 开启 `fsdp_plan.pregather=true`；注意 prefetch 加深（2/2）反而变慢，保持 1/1。详见坑表 23 与 `docs/` 性能文档。

**Q: 性能数字怎么取？**
A: 官方口径 = **50–100 步**统计；`60_bench.py` 同时给出均值/中位/剔除慢步值与 samples/s（GBS×1000/STEP_TIME）。

**Q: 为什么判定是 NEEDS_EVIDENCE 而不是绿档？**
A: 配置零偏离 + 逐点可行已达成；未闭合的证据门是**官方数据 sha256 不可得**（`data_identity`），
按诚实红线不得宣称绿档。闭合路径：官方提供哈希，或书面认可"构造性同源"。
