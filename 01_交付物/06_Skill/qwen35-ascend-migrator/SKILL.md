---
name: "qwen35-ascend-migrator"
description: "将 HF 风格多模态大模型自动迁移到昇腾 NPU（MindSpeed-MM）：AST 迁移点识别、迁移方案生成、算子验证、资产准备、训练执行、性能基准、精度判定，全程证据落盘。具备跨环境冗余降级能力（有无 NPU / 不同芯片 / 单双卡 / 不同 CANN-torch 版本 / triton 可用性）。Invoke when asked to migrate or adapt a model to Ascend NPU, run/verify training under MindSpeed-MM, or produce migration evidence."
---

# qwen35-ascend-migrator —— 昇腾自动迁移·调优·判定 Skill

> **定位**：从"输入模型源码 + 目标环境"到"昇腾上跑通 + 性能达标 + 精度对齐 + 证据落盘"的**全流程自动化流水线**。
> **设计原则**（本项目核心要求）：
> 1. **完备优先于速度**——允许阶段耗时长（版本试装、多轮探测），但每一步必须有明确判据与产物；
> 2. **综合能力**——识别、迁移、验证、训练、优化、判定、出证据，端到端闭环；
> 3. **冗余降级**——任何单点失败都有替代路径（见 §3 冗余矩阵），**绝不因环境不支持而中断**；
> 4. **诚实红线**——降级必标注原因；无证据不宣称；不伪造任何数字。

---

## 1. 能力概览

| 能力 | 说明 | 对应阶段 |
|---|---|---|
| 环境探测与自适配 | 探测 NPU/CANN/torch/triton 可用性 → 决定执行路径与配置档 | P0 |
| 迁移点识别 | AST 静态解析建模源码，程序化输出迁移点矩阵（带行号/符号） | P1 |
| 迁移方案生成 | 迁移点 → MindSpeed-MM 落点映射 + 训练配置生成 | P2 |
| 算子/契约验证 | 算子 forward/shape 矩阵 + 数值一致性核对 | P3 |
| 资产准备 | 权重（hf/dcp）、数据集下载与格式转换，含字节级校验 | P4 |
| 训练执行 | 环境自适应的训练启动（几何/开关/后端自动选择） | P5 |
| 性能基准 | 官方口径（50-100 步均值/中位数 + samples/s）+ 逐轮优化登记 | P6 |
| 精度判定 | 配置指纹 + 实测指纹 + 五态裁决 + 哈希链账本（0-GPU 可复核） | P7 |
| 证据产出 | 全量日志、逐轮 JSON、判定 verdict、可直接喂交付物的报告数据 | 全程 |

---

## 2. 流水线（9 阶段 = P0–P7 + P6b，上一步产物是下一步输入）

```
P-1 preflight ─► P0 probe ─► P1 analyze ─► P2 plan ─► P3 verify ─► P4 assets ─► P5 train ─► P6 bench ─► P6b 可报告性 ─► P7 judge
    │              │            │             │            │             │            │            │             │            │
 能力矩阵        环境档案     迁移点矩阵   迁移方案+配置  算子矩阵     资产清单      训练日志     性能数据     可报告性判定    判定账本
capability.json  env.json   points.json   plan.md+cfg  ops.json    assets.json  train.log   bench.json  reportability.json verdict.json
```

> **P6b 是硬闸门**：任何"看起来能变快"的改动，都必须先在**同批次**量出噪声底噪，
> 才允许把它的数字写成结论（`62_reportability.py` v2：**95% 区间下界 ≥ 阈值**才算达标）。
> 详见 [优化候选评估固定流程](docs/优化候选评估固定流程_20260922.md)。

**P-1 · preflight —— 开工前能力矩阵（★ 鲁棒性第一道闸门）**

```bash
python3 scripts/05_preflight.py --json [--want quick|full]
```
**产出**：`out/preflight/capability.json` + 降级账本 `out/degradations.json`
**三态判定**：`OK` / `DEGRADED`（走降级路径，已登记）/ `BLOCKED`（依赖它的阶段将跳过，**不伪造结果**）/ `SKIP`
**关键**：对每项不可用能力**显式写出"因此不得宣称什么"**，例如数据非全量同源时输出
`pointwise_claim_allowed = BLOCKED → 不得宣称逐点可比/精度已对齐`。
→ 把"跑到一半才发现环境不支持"变成"**开工前就知道会怎样**"。

### P0 · probe —— 环境探测（决定整条路径）

```bash
python3 scripts/00_probe_env.py            # 输出 out/probe/env.json + 人读摘要
```
**探测项**：NPU 是否可见（`npu-smi` / `torch.npu.device_count()`）、die 数、芯片型号、
CANN 版本、torch/torch_npu/triton-ascend/transformers 版本、`python3-dev` 可用性、
磁盘空间、网络可达性（modelscope/obs/gitcode）。

**判据与产物**：
- 输出 `out/probe/env.json`，含 `path` 字段：`full_npu` / `single_die` / `cpu_only`
- **不判停**：即使 NPU 不可用也继续（走 CPU 轨），仅在后续阶段标注降级

### P1 · analyze —— 迁移点识别

```bash
python3 scripts/10_analyze_points.py <modeling_source.py>
```
**产出**：`out/analyze/migrate_points.json` + `migrate_points_report.md`
**内置 4 类规则**：`linear_attn_GatedDeltaNet`（非标件）/ `full_attn` / `vision_3d_conv_patch` / `custom_act_or_norm`
**红线**：源文件为占位/无效（4 类全 0 命中且内容可疑）→ 诚实回退内嵌自检，**不伪造命中**

### P2 · plan —— 迁移方案与配置生成

```bash
python3 scripts/20_plan_migration.py --points out/analyze/migrate_points.json \
    --env out/probe/env.json --model qwen3_5 --out out/plan/
```
**产出**：
- `out/plan/migrate_plan.md`：每个迁移点 → 目标文件 / 适配方式 / 验收标准
- `out/plan/train_config.yaml`：**按环境档案自动选择配置档**（几何、显存开关、后端、并行度）
**配置档选择逻辑**（冗余矩阵核心，见 §3）：按 `env.json` 的 die 数、HBM 容量、triton 可用性自动生成

### P3 · verify —— 算子与契约验证

```bash
python3 scripts/30_verify_ops.py --env out/probe/env.json
```
**产出**：`out/verify/ops_matrix.json`（每类算子 forward/shape 状态）+ 数值一致性（有卡时）
**诚实规则**：无本地 kernel 的算子标 `contract-only`（只声明 shape 契约）；单算子失败不中断其余

### P4 · assets —— 权重与数据准备

```bash
python3 scripts/40_prepare_assets.py --env out/probe/env.json --data-dir /root/data
```
**产出**：`out/assets/assets.json`（每个资产的路径 + 字节数/sha256 + 校验结果）
**内置校验锚点**：`llava_instruct_150k.json` = 228,941,895 B（官方一致）；
转换产物 `output_llava_coco_data.json` 应 157,712 样本
**冗余**：modelscope 失败 → 备用源；全量数据不可得 → 子集/mock 并标注可比性等级

### P5 · train —— 训练执行

```bash
python3 scripts/50_train.py --config out/plan/train_config.yaml --steps 100 \
    --log out/train/train.log
```
**特性**：
- 自动注入必备环境变量（`NON_MEGATRON`/`TASK_QUEUE_ENABLE`/`set_env.sh` 等，缺失即失败项）
- 后台 + 轮询模式（长任务不阻塞；SSH 场景必须）
- 失败自动诊断：常见错误（缺 megatron / libhccl 找不到 / triton 回退 / OOM）→ 打印对应处置
**产出**：训练日志（**从程序启动开始的完整日志**，直接作为交付物 #5 数据源）

### P6 · bench —— 性能基准

```bash
python3 scripts/60_bench.py --round 1 --tag baseline --out out/bench --note "关联 log 编号"
```
**产出**：`out/bench/round_<n>.json`（输入日志默认取自 `out/train/train.log`）
**口径**（官方确认）：**50–100 步**统计；同时输出均值/中位数/剔除慢步值 + samples/s（GBS×1000/STEP_TIME）
**判停**：无基线轮 → 拒绝出对比数据

#### P6b · 优化候选评估固定流程（**先量噪声**，硬闸门）

拿到任何"看起来能变快"的改动，按**固定顺序**评估 —— 完整说明见
[优化候选评估固定流程](docs/优化候选评估固定流程_20260922.md)：

```
① 探测 → ② 候选试跑 → ③ 先量噪声(闸门) → ④ 正确性/资源检查 → ⑤ 选档 → ⑥ 输出证据与边界
                            ↑ 62_reportability.py 在这里拦人
```

```bash
# ③ 先在**同一批次**里量噪声（不改任何键）
python3 scripts/59_config_ab.py --null-test 3 --steps 100 --out out/ab_null
# ③ 再让闸门判"这个效应配不配被报告"
python3 scripts/62_reportability.py --noise out/ab_null/null_summary.json \
    --summary out/ab/summary.json --denominator baseline \
    --baseline-variant B --candidate-variant A --accuracy-validity ok \
    --out out/reportability.json
```
**★ 硬门（v2）**：`performance_pass = (95% 区间下界 ≥ 阈值)`，阈值 = `max(2 × 实测底噪, 5%)`
（底噪倍数出自 `protocols/prefetch_depth_20260922.json` 的 `rule_added`，**仅作工程筛选**；
5% 下限出自性能纪律）。**缺同批次底噪或缺区间 ⇒ `INVALID_INPUT`，不许出数字**——
**不允许用点估计顶替区间**（v1 的缺口：点估计 10%、区间跨零也曾判"可报告"）。

**三类结果必须拆开**：`run_validity` / `accuracy_validity` / `performance_pass`，
另派生 `adoption_ready`（三者同时 OK 才可采纳）。
**退出码**：`0` 达标且可采纳 / `3` **达标但采纳关未闭合** / `4` 判得动但不达标
/ `5` 判不动 —— "不达标""判不动""达标但不可采纳"是**三种不同的事**，必须分开。

★ **角色与分母**：`variant` 的 A/B 指的是**两个配置**，**不等于**基线/候选
（要评估「配置 A 相对 B 省时」⇒ 候选是 A、基线是 B ⇒ 必须 `--baseline-variant B --candidate-variant A`；
未显式声明时产物标 `roles_assumed=true` 并告警）。
产物统一存 `baseline_run_id` / `candidate_run_id` / `savings_pct` / `ci95_savings_pct` / `denominator`，
其中 `savings_pct = (baseline − candidate) / 分母 × 100`，**正号恒等于「候选更快」**；
并按**逐对比值再取均值**（配对设计）计算。

CPU 配额/线程瓶颈诊断时，阅读 [T1/T4 实测记录](docs/CODEX_T1_T4_20260916.md)，
使用 `96_cpu_quota_probe.py`、`97_cpu_thread_sweep.py` 采集，再用
`98_audit_cpu_sweep.py` 核对原始证据。该矩阵固定 dp2/mbs4/gas1，
不是任意拓扑通用基准；30 步诊断窗口不得替代官方 50–100 步指标。
需要区分节流发生阶段或估计运行间波动时，使用 `99_repeat_phase.py`；
执行和离线审计命令、采样边界及统计假设见上述实测记录的“T3 阶段采样与重复实验”一节。

筛选 AdamW 的 fused/foreach 路径时，先阅读
[优化器筛选边界与运行记录](docs/OPTIMIZER_SCREEN_20260917.md)，再使用
`58_optimizer_screen.py`。这是合成张量探针，不测整模型性能或 kernel 发射数；
不得把其耗时差当作官方几何加速或发射弹性系数。

### P7 · judge —— 精度与配置判定

```bash
python3 scripts/70_judge.py --log out/train/train.log --config out/plan/train_config.yaml \
    --data-json /root/data/output_llava_coco_data.json --baseline officialB \
    --baseline-log examples/train/official_baseline.log \
    --out out/judge --registry sk04_judge/evidence/registry.json --tag <唯一tag>
```
**产出**：`out/judge/{fp,obs,verdict}.json` + 账本追加
**五态裁决**：POINTWISE_OK / WINDOW_OK / NEEDS_EVIDENCE / NOT_COMPARABLE / INVALID_INPUT
**纯函数契约**：`verdict_id = sha256(基线字节 + 指纹字节 [+ 实测字节])`，同证据必得同 id

---

## 3. ★ 冗余矩阵（跨环境适配，本 Skill 的核心竞争力）

任何维度不满足时，**自动降级到可用路径并如实标注**，绝不中断流水线。

### 3.1 执行路径（P0 决定）

| 路径 | 触发条件 | 行为 |
|---|---|---|
| `full_npu` | NPU 可见 + triton 可用 | 全流程真机：训练 + 性能 + 判定 + 数值一致性 |
| `single_die` | NPU 可见但仅 1 die（或官方拓扑需 dp2 而本机 1 die） | 训练可跑但**标注"几何与官方不同，仅窗口口径可比"** |
| `np0_triton_off` | NPU 可见但 triton 不可用 | 降级算子后端：triton → **ascendc** → **eager**（三者数值等价已验证 Δ<0.2%） |
| `cpu_only` | 无 NPU | P1/P2 静态分析与方案生成照常；P3 仅证 shape 契约；P5-P6 跳过并说明；**P7 判定链可完整运行**（0-GPU） |

### 3.2 芯片与显存档位（P2 自动选配置）

| 芯片档 | 显存 | 配置策略 |
|---|---|---|
| 910B4 | 32GB/chip | `recompute=true`、`chunk_loss=true`、`activation_offload=true`（省显存优先）；mbs 上限 2；单 die |
| 910C 单 die | 64GB | 可关显存节省开关；mbs 4–8 |
| **910C 双 die** | 2×64GB | **官方几何**：`world_size=2 / mbs=4 / gas=1 / dp=2`；关闭显存节省开关；`pregather=true` |

### 3.3 软件版本矩阵（P0 探测 → 自动选版本与安装源）

| CANN | torch | Python | triton-ascend | 安装源 |
|---|---|---|---|---|
| 9.1.0 | 2.7.1 | 3.10 | **3.2.2** | huaweicloud ascend 源 + 主源补 `attrs==24.2.0` |
| 9.1.0 | 2.7.1 | 3.11 | 3.2.2 | 同上（需验证） |
| **9.1.0-beta.3** | 2.7.1 | **3.11.6** | **3.2.2** | ✅ **已实测（2026-09-16）**：在全新机器（CANN 9.1.0-beta.3 + Python 3.11.6 + openEuler 24.03 SP3）上**从零跑通官方几何 100 步 + 判定链**，产出第二组隔离证据（`examples/judge_ascend_beta/`，`verdict_id=f2ad023d…`，与 A3 的 `606056b3…` **互不覆盖**） |
| 9.0.x | 2.6.x | 3.10 | 3.2.0 / 3.2.1 | osinfra 源或 pypi |
| 其他 | — | — | 探测后按官方配对表选择 | 失败则降级后端（§3.1） |

**前置依赖**：`apt install python3-dev build-essential`（缺 → triton 驱动编译 `Python.h` 失败 → 回退 CPU）
> ⚠️ **openEuler/其他发行版**：包管理器可能是 `dnf`/`microdnf`/`yum` 而非 `apt`，
> 对应包名为 `python3-devel` / `gcc` / `gcc-c++`。**先探测 `which apt dnf yum microdnf` 再装**，不要硬套 apt 命令。
> ⚠️ **Python 版本是独立维度**：triton-ascend 的 wheel 按 cpython 版本区分（`cp310`/`cp312`…），
> 换 Python 版本等于换配对，**必须重跑 P0 的 triton 最小 kernel 判据**，不可沿用 py3.10 的结论。

### 3.4 数据冗余

| 数据档 | 触发 | 可比性标注 |
|---|---|---|
| 官方全量同源 | 默认目标 | 逐点可比候选（配官方几何） |
| 子集 | 全量不可得/时间不足 | **窗口口径可比**，标注"数据规模不同" |
| Mock 数据 | 仅验证流程连通性 | **不可比**，仅证明"能跑通" |

### 3.5 判定口径冗余

无论环境如何，判定链**同时输出**：
- 逐点（pointwise）指标：逐步相对误差 + 4 指标（Mean/MSE/Max/Min）
- 窗口（window）指标：50–100 步均值/中位数偏差
→ 适配官方可能采用的不同验收口径（官方尚未明确逐点或窗口，故双轨齐备）

### 3.6 连接冗余（远程执行场景）

| 问题 | 处置 |
|---|---|
| 跳板 token 过期（约 5 分钟） | 长任务必须 `setsid nohup` 后台；脚本自等待 + 落盘，不依赖在线会话 |
| 单条 SSH 命令 120s 被掐 | 一律后台执行 + 日志轮询；绝不用前台长命令 |
| SFTP 被跳板阻断 | base64 编码传输：`echo <b64> \| base64 -d > file` |
| 网络间歇失败（github/clone） | 重试 + 镜像兜底（gitcode / modelscope / huaweicloud 源） |

---

## 4. 诚实红线（不可违反）

1. **降级必标注**：任何走 CPU 轨/降级后端/子集数据的结果，必须在产物中标注 `degraded: true` + 原因；
2. **数字必溯源**：报告只引用日志编号与 `verdict_id`，禁止手抄数字；无证据的不确定性标注"待验证"；
3. **不伪造命中/贯通**：AST 识别不到真实迁移点就不报数字；算子无实现就标 `contract-only`；
4. **判定不越权**：可达级别由机器裁决（`pointwise_feasible`/`window_feasible`/`none`），
   未闭合的证据门（如官方数据哈希不可得）必须显式列为 open gate，不得宣称绿档。

---

## 5. 产物 → 复赛交付物映射

| Skill 产物 | 交付物 |
|---|---|
| `out/train/train.log`（全量、从启动开始） | #5 原始日志 |
| `out/bench/round_<n>.json` | #4 性能测试报告 |
| `out/judge/*.json` + `loss_compare.csv` | #3 精度分析报告 |
| `out/analyze/migrate_points_report.md` | #1 创意书技术路线 / #2 README 代码逻辑 |
| `out/plan/migrate_plan.md` + `train_config.yaml` | #7 迁移后源码的生成依据 |
| 本 Skill 全量（含 `selfcheck.py`） | #6 Skill 文件夹 |

---

## 6. 使用示例（Agent 调用）

> **用户**：把 `modeling_qwen3_5.py` 迁到昇腾 NPU 上，MindSpeed-MM 环境，跑 100 步对齐官方基线。

**Skill 执行**：
1. **P0** 探测 → 发现 1×910C 双 die、CANN 9.1.0、triton 3.2.2 可用 → `path=full_npu`
2. **P1** 解析源码 → "识别 4 类 N 处迁移点（含行号）"
3. **P2** 生成方案 + 配置 → 自动选官方几何档（dp2/mbs4/gas1）+ 关闭显存节省开关 + pregather
4. **P3** 算子矩阵 → GDN 走 NPU Triton（rollback=0 实证）
5. **P4** 资产 → 权重 hf+dcp、数据全量同源（字节校验通过）
6. **P5** 训练 100 步 → 完整日志落盘
7. **P6** 性能 → 50–100 步中位 419ms / 19.09 samples/s（对照官方 431.3ms）
8. **P7** 判定 → `mismatch=[]`、`pointwise_feasible`、`verdict_id=...`、账本追加
9. 输出交付物映射表，指出每份报告可直接引用的证据编号

---

## 7. 自检（0-GPU 可跑，评委复现入口）

```bash
python3 selfcheck.py                    # 四层：结构 / 判定链 / 环境 / 判据可证伪
python3 selfcheck.py --no-gates         # 跳过 L4（慢）
python3 scripts/_envcompat.py --selftest  # 兼容层自身
# 期望：SELFCHECK_OK
```

**四层含义**：
| 层 | 内容 |
|---|---|
| L1 | 结构完整性 + `.py` 语法 + **`.sh` 语法（`bash -n`）** + 配置可解析 |
| L2 | 判定链可用性（SK04，含**包内示例重放** → 无 NPU、无仓库也能自证红线 `verdict_id`） |
| L3 | 环境一致性：**acceptance**（须是 lock 描述的那台验收环境）与 **portability**（只要求栈内部自洽）**分别判定** |
| L4 | **判据负向对照**：每个关键判据都要证明"喂坏输入时会失败"，否则标 `GATE_VACUOUS` |

> **L4 是本 Skill 鲁棒性的核心。** 干净环境验证曾暴露 4 个假阳性判据与 1 个假阴性判据，
> 它们的共性是"**判据从没被验证过能不能失败**"。一个不会失败的判据不是判据，是装饰。
> 详见 `docs/ROBUSTNESS.md`（7 条可检验不变量）与 `docs/PITFALLS_坑表.md` 坑 30-55。

```bash
python3 scripts/90_selfcheck_gates.py   # 单独跑判据负向对照 + 静态回归守卫
# 期望：GATES_OK（SKIP 会显式列出，它既不算通过也不算失败）
```

---

## 7b. ★ 从零环境端到端验证（新机器一键跑通）

> 场景：换了一台**全新的、什么都没有**的机器 → 验证本 Skill 能否独立完成全部任务。

```bash
bash scripts/run_from_zero.sh              # quick 模式（跳过 COCO 19GB，先验流程连通）
bash scripts/run_from_zero.sh --full       # 全量模式（下载全量 COCO，产出可比数据）
bash scripts/run_from_zero.sh --from P4    # 中断后从某阶段续跑
```

**核心特性**：
- **失败不中断**——逐个阶段独立裁决 PASS/FAIL/TIMEOUT/**RESUMED**/**SKIP**，**一次性暴露全部缺口**，不在第一个错误处停下
- **判据必须覆盖收尾**——产物存在不等于训练成功；P5 必须同时核对退出码、完整步数与错误标记，缺退出证据不得算成功
- **红线自检内建**——核算 `GBS = mbs × gas × world = 8`，且声明 dp 必须等于实际 world；P5 后打印 step1 loss（官方锚点 1.924621）
- **降级账本一致性**——阶段日志出现 `DEGRADED` 而 `out/degradations.json` 为空 → 计为失败（**不许悄悄降级**）
- **续跑是产物校验型的**——`--resume` 仅在"产物比它的每个输入都新"时跳过，并记为 `RESUMED` 而非 PASS（防坑 35 陈旧产物）
- **诚实降级标注**——quick 模式会在报告里显式声明"数据可比性降级，不可宣称精度已对齐"

**产物**：`out/logs/<stage>.log`、`out/logs/_verdicts.tsv`（裁决表）、`out/from_zero_report.md`（含缺口清单 + ★ 降级登记章节）
**退出码** = 失败阶段数（0 = 全通过）

---

## 8. 泛化路线

| 版本 | 范围 |
|---|---|
| **v1（本次提交）** | Qwen3.5 专用规则内置；已有双 die 真机迁移证据。精度裁决为 NEEDS_EVIDENCE（黄），data_identity/sample_order 未闭合；尚未证实性能达标，不宣称逐位对齐或通用加速倍数 |
| v2 | 迁移点规则外置 `profiles/<model>.json`，AST 扫描器通用化 → 支持任意 HF 多模态模型 |
| v3 | 上游 PR 合入后，P2 直接消费 MindSpeed-MM 官方适配层，本 Skill 收敛为"探测+验证+基准+判定"角色 |
