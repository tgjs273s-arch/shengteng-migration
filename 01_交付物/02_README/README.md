# Qwen3.5-0.8B 昇腾训练迁移 · README

> C4-AI 2026 昇腾赛道复赛 · 交付物说明
> 模型：Qwen3.5-0.8B（多模态，GatedDeltaNet + Vision）
> 框架：MindSpeed-MM v26.1.0（FSDP2 路径）
> 硬件：1× 昇腾 910C 双 die（Ascend910_9382，2×64GB HBM）

---

## 0. 成果概览（先看这里）

| 维度 | 结果 | 核验方式 |
|---|---|---|
| **迁移** | Qwen3.5-0.8B 在 1× 昇腾 910C 双 die 上完整跑通 100 步训练（官方几何 world2/mbs4/gas1/dp2） | 训练日志（从程序启动开始） |
| **精度** | 第 1 步 loss 与官方基线**逐位一致**（1.924621 = 1.924621，偏差 **0.000000%**）；100 步逐点 **0/100** 超 2% | `70_judge.py` 判定链（0-GPU 可复跑） |
| **性能** | 单步 **21,393 → 419.0 ms（51.1×）**；50–100 步中位 **419.0 ms**，优于官方基线 431.3 ms（**数值绑定 2026-09-10 的宿主节点，见下**） | `60_bench.py` + 原始日志 |
| **性能的节点依赖性** | 同一 Skill/几何/窗口在不同"容器运行时段"上可差 **2.27×**（最慢 946.4 ms ↔ 最快 417.1 ms；9/16 批次 925.4 ms）；判据与适用条件见性能测试报告文末一节 | 跨时段重复实验（`identity.boot` 自证）+ `VERIFY_T1_OK` |
| **配置一致性** | 7+2 维配置指纹 **mismatch=[]（零偏离）**，可达级别 `pointwise_feasible` | `fingerprint_cfg.py` |
| **可复核性** | `verdict_id` 固化 + append-only 哈希链账本（5 条，`REGISTRY_VERIFY_OK`） | `registry_append.py --verify` |
| **Skill** | 8 阶段自动迁移流水线（6 维冗余降级：跨芯片 / 单双卡 / 有无 NPU 均可运行） | `selfcheck.py` → `SELFCHECK_OK` |

**技术路线一句话**：复刻官方并行几何 → 同源数据管线 → NPU Triton 算子实现 →
以 profiling 定位瓶颈（launch-bound）→ 八级单变量优化 → 机器判定链固化结论。

> **关于性能数字的适用范围（必读）**：419.0 ms 来自 2026-09-10 的宿主时段。
> 2026-09-16/17 的重复实验显示，**同一份代码、同一官方几何（GBS=8）、同一窗口（50–100）
> 在不同"容器运行时段（epoch）"上的窗口中位相差 2.27×**
> （最慢 946.4 ms / 2026-09-16 批次 925.4 ms ↔ 最快时段 417.1–451.7 ms，6 组）。
> 因此：① 引用性能数字必须同时说明所在时段/节点身份（`identity.boot` / `hostname`）；
> ② 上限结论以"在同级时段上达到 417–452 ms ≈ A3 的 419 ms、优于官方 431.3 ms"表述；
> ③ 该时段依赖性本身是**需要披露的鲁棒性问题**，不因某台机器上跑得快而消失。
> 判据、九组线程矩阵与机制边界见 `04_性能测试报告/性能优化报告.md` 文末「附：性能的节点依赖性」。

---

## 1. 环境版本

### 1.1 镜像环境

| 项 | 值 |
|---|---|
| 基础镜像 | `quay.io/ascend/cann:<devel tag>`（Ubuntu 22.04，Python 3.10.12，devel 版） |
| 容器内路径 | 工作目录 `/root/MindSpeed-MM` |
| 昇腾设备 | `/dev/davinci6`、`/dev/davinci7`（同一物理卡的两个 die） |
| 镜像补充安装 | `apt install python3-dev build-essential`（**必需**：triton-ascend 驱动需现场编译 C++ 桥接，缺 `Python.h` 会导致 triton 回退 CPU） |

### 1.2 CANN 与框架版本

| 组件 | 版本 | 说明 |
|---|---|---|
| **CANN** | **9.1.0** | `/usr/local/Ascend/cann-9.1.0`，驱动 25.5.0 |
| **torch_npu** | **2.7.1.post10** | 与 CANN 9.1.0 官方配对 |
| torch | 2.7.1 | |
| **triton-ascend** | **3.2.2** | CANN 9.1.0 + torch 2.7.1 + py3.10 的**官方配对版本**；导入名为 `triton`。注意 3.2.0/3.2.1 在 910C 上不可用 |
| Python | 3.10.12 | 系统解释器 `/usr/bin/python3`（无 venv） |
| transformers | 5.2.0 | MindSpeed-MM v26.1.0 配套版本 |
| MindSpeed-MM | v26.1.0（tag） | 训练框架本体 |
| MindSpeed | 0.12.1 | MSMM 核心依赖（editable 安装，`--no-deps`） |
| 其他依赖 | pyyaml / numpy / pybind11 / protobuf / scipy / sentencepiece / torchdata / torchvision / ftfy / diffusers / qwen_vl_utils / einops / av / pandas / attrs==24.2.0 | 按 pyproject 全量补齐 |

完整版本锁文件：`config/versions.lock`（含数据/权重/配置/环境变量全部锁定项）。

### 1.3 关键环境变量（缺失将导致启动失败）

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh          # 缺则 libhccl.so 找不到
export NON_MEGATRON=true                                    # 缺则走 Megatron 路径报 No module named 'megatron'
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/root/triton_cache
```

---

## 2. 运行启动脚本

### 2.1 一键环境自检（0-GPU 可跑，评委复现入场券）

```bash
python3 selfcheck_env.py            # 对照 versions.lock 逐项校验本机环境
# 输出 SELFCHECK_OK / SELFCHECK_FAIL
```

### 2.2 正式训练（最终验收配置，100 步）

```bash
cd /root/MindSpeed-MM

# 见 §1.3 环境变量（务必先 export）
torchrun --nproc_per_node 2 --nnodes 1 --node_rank 0 \
  --master_addr localhost --master_port 6111 \
  mindspeed_mm/fsdp/train/trainer.py \
  examples/qwen3_5/e4b_final100.yaml \
  > /root/a3_e4b_100.log 2>&1
```

配置要点（`e4b_final100.yaml`）：

| 项 | 值 | 说明 |
|---|---|---|
| 并行 | `world_size=2`（2 die）、`mbs=4`、`gas=1`、`dp=2`、GBS=8 | 与官方 7/24 基线几何完全一致 |
| 算子实现 | `gdn_implementation: triton`、`causal_conv1d_implementation: triton` | NPU Triton 实现 |
| 显存节省开关 | `recompute=false`、`enable_chunk_loss=false`、`enable_activation_offload=false` | 64GB/die 无需省显存，关闭以消除开销（优化点 7） |
| FSDP | `pregather=true`、`num_to_forward/backward_prefetch=1` | 预收集参数消除加载阻塞慢步（优化点 8） |
| 数据 | `dataset: /root/data/output_llava_coco_data.json`，`shuffle=false`，`seed=42` | 官方同源管线数据 |
| 权重 | `load: /root/Qwen3.5-0.8B-dcp`（DCP 格式）、`model_name_or_path: /root/Qwen3.5-0.8B-hf` | |
| 步数 | `train_iters=100`，`lr=1e-5` cosine（warmup 10%），`weight_decay=0` | |

### 2.3 数据准备脚本

```bash
pip install modelscope==1.38.1

# COCO2017 图片（约 18GB，118,287 张）
mkdir -p data/coco && modelscope download --dataset PAI/COCO2017 train2017.zip --local_dir ./data/coco
cd data/coco && unzip train2017.zip && cd ../..

# LLaVA-Instruct-150K 提示词（228,941,895 字节）
mkdir -p data/llava && modelscope download --dataset AI-ModelScope/LLaVA-Instruct-150K \
  llava_instruct_150k.json --local_dir ./data/llava

# 转换为训练格式（157,712 样本，与官方基线同源同脚本）
python mindspeed_mm/fsdp/tools/data_tool/llava_instruct_2_mllm_demo_format.py \
  --llava_json_path ./data/llava/llava_instruct_150k.json \
  --coco_path ./data/coco \
  --output_json_path ./data/output_llava_coco_data.json
```

### 2.4 权重转换（hf → DCP）

```bash
python3 -m checkpoint.convert_cli GenericDCPConverter hf_to_dcp \
  --load_path /root/Qwen3.5-0.8B-hf \
  --save_path /root/Qwen3.5-0.8B-dcp
```

### 2.5 判定链（精度/配置机器判定，0-GPU 可复现）

```bash
cd /root/skills_R2/skill_R2-SK04

# ① 配置指纹门（7+2 维偏离比对）
python3 scripts/fingerprint_cfg.py --config-from-log /root/a3_e4b_100.log \
  --baseline officialB --data-json /root/data/output_llava_coco_data.json \
  --out verdict/fp.json

# ② 运行实测指纹（GBS/kernel 生效性/step1 loss 带判定）
python3 scripts/fingerprint_observed.py --log /root/a3_e4b_100.log \
  --declared verdict/fp.json --out verdict/obs.json

# ③ 可比性裁决（五态判定 + verdict_id）
python3 scripts/judge_comparable.py --run verdict/fp.json --observed verdict/obs.json \
  --baseline officialB --out verdict/verdict.json

# ④ 追加到哈希链账本 + 校验
python3 scripts/registry_append.py --tag <tag> --run-dir verdict \
  --manifest verdict/verdict.json --fingerprint verdict/fp.json --observed verdict/obs.json
python3 scripts/registry_append.py --verify
```

预期输出：
```
FINGERPRINT_OK level=pointwise_feasible mismatch=[] n_eff=8/8
OBSERVED_OK    step1_loss=1.924621 kernel=npu_triton_active rollback=0
JUDGE_OK       verdict=NEEDS_EVIDENCE deviations=0 verdict_id=606056b3cb04ecbb
REGISTRY_VERIFY_OK entries=5
```

---

## 3. 代码结构说明

### 3.1 交付目录结构

```
.
├── README.md                          # 本文件
├── config/
│   ├── versions.lock                  # 验收环境版本锁（42 项：硬件/CANN/框架/数据/配置/env）
│   └── e4b_final100.yaml              # 最终验收训练配置
├── selfcheck_env.py                   # 环境一致性自检（0-GPU，对照 versions.lock）
├── run_bringup.sh                     # 新卡一键迁移入口（phase0-4：探测→环境→框架→资产→验证）
├── skill_R2-SK04/                     # 判定链（配置指纹 / 实测指纹 / 裁决 / 账本）
│   ├── scripts/fingerprint_cfg.py     #   7+2 维配置指纹门 + 可达性谓词
│   ├── scripts/fingerprint_observed.py#   运行实测指纹（GBS/kernel/step1 band）
│   ├── scripts/judge_comparable.py    #   五态裁决机（纯函数，输出 verdict_id）
│   ├── scripts/registry_append.py     #   append-only 哈希链账本
│   ├── scripts/data_id.py             #   数据集字节/顺序身份
│   ├── scripts/selfcheck_sk04.py      #   判定链自检（24/24）
│   └── configs/baselines/officialB.yaml # 官方基线声明指纹（含出处）
├── deliverables/
│   ├── 精度对齐报告.md                 # 官方模板：标题 + 对比图 + 4 误差指标
│   ├── 性能优化报告.md                 # 官方模板：汇总表 + 8 个优化点小节
│   ├── loss_compare_100steps.csv      # 100 步逐点对比原始数据
│   └── fig1~fig4.svg                  # 精度对比图（loss/误差/grad_norm）
└── docs/
    ├── official/triton精度日志.txt     # 官方 7/24 基线日志（对比基准）
    └── *.md                           # 实验记录与判定证据
```

### 3.2 训练栈代码结构（MindSpeed-MM v26.1.0）

```
MindSpeed-MM/
├── mindspeed_mm/fsdp/
│   ├── train/trainer.py               # 训练入口（torchrun 调用）
│   ├── distributed/
│   │   ├── fully_shard_parallel.py    # FSDP2 包装逻辑（含 pregather_fsdp_params、DDP 分支）
│   │   └── parallel_state.py          # 并行状态（dp/tp/fsdp 尺寸解析）
│   ├── ops/
│   │   ├── fully_shard/fully_shard.py # FSDP2 定制 patch（layer-wise hook + 多流通信）
│   │   └── gdn/                       # GatedDeltaNet 算子
│   │       ├── triton/                #   NPU Triton 实现（本次采用）
│   │       └── ascendc/               #   AscendC 实现（备选）
│   ├── models/qwen3_5/                # Qwen3.5 模型适配
│   ├── data/                          # 数据集与预处理
│   └── tools/data_tool/
│       └── llava_instruct_2_mllm_demo_format.py  # 官方数据转换脚本
└── examples/qwen3_5/
    ├── qwen3_5_0_8B_final_config.yaml # 0.8B 基础配置（v26.1.0 tag 未自带，本次补齐）
    └── e4b_final100.yaml              # 最终验收配置（本文档使用）
```

### 3.3 代码逻辑说明

**（1）训练主流程**
`torchrun --nproc_per_node 2` 启动两个进程（各绑定一个 die）→ `trainer.py` 读取 YAML →
`parallel_state.py` 按 `world_size=2` 解析出 `dp=2 / fsdp=2 / tp=1` →
`fully_shard_parallel_modules()` 对模型逐层施加 FSDP2 分片（参数 bf16、梯度 reduce fp32）→
数据侧按 `shuffle=false + seed=42` 顺序取 8 样本/步（每进程 4 样本）→
前反向计算（GDN 走 NPU Triton kernel）→ 梯度 reduce-scatter 同步 → AdamW 更新。

**（2）本次未修改框架源码**，全部通过官方配置项达成（含 3 个显存开关与 `pregather`）：
| 配置项 | 作用 | 收益 |
|---|---|---|
| `gdn_implementation: triton` | GDN 算子走 NPU Triton | 3.44× |
| `micro_batch_size / gradient_accumulation_steps` | 控制每步 micro-batch 个数（launch-bound 主因） | 2.04× |
| `features.recompute=false` | 关闭激活重算（64GB 显存无需省） | 1.22× |
| `features.enable_chunk_loss=false` | 关闭分块 loss | 同上 |
| `features.enable_activation_offload=false` | 关闭激活 CPU 卸载 | 同上 |
| `fsdp_plan.pregather=true` | 前向前预收集 FSDP 参数，消除加载阻塞慢步 | 1.08× + 消长尾 |
| `num_to_forward/backward_prefetch=1` | 保持默认（实验证明加深反而慢 40ms） | — |

**（3）数值路径**：权重 bf16、梯度 reduce fp32、GDN 全程 Triton 实现；
第 1 步 loss（lr=0 纯前向）= **1.924621**，与官方基线逐位一致，构成数值等价直接证据。

**（4）判定链逻辑**（`skill_R2-SK04`，纯 CPU、可 0-GPU 复核）
- `fingerprint_cfg`：从训练日志内嵌的 Configuration Details 提取 7+2 维配置指纹，
  与官方基线声明逐维比对，输出偏离表与可达性谓词（pointwise/window/none）；
- `fingerprint_observed`：从日志提取实测指纹（GBS、kernel 生效性、step1 loss 带、LR 序列哈希）；
- `judge_comparable`：纯函数裁决机，输出五态之一（POINTWISE_OK / WINDOW_OK / NEEDS_EVIDENCE /
  NOT_COMPARABLE / INVALID_INPUT）与 `verdict_id`（哈希链固化，防人工粘贴数字）；
- `registry_append`：append-only 哈希链账本，任何篡改/重排在 `--verify` 时被检出。

---

## 4. PR 链接

> ⚠️ 待确认/补充

**当前状态**：本方案的性能与精度成果**未修改 MindSpeed-MM 框架源码**，全部通过官方配置项实现，
因此无可提交至上游的代码 PR。

**可提交的实质贡献**（如需 PR，建议以下三项，均为复赛过程中实证发现的框架缺口）：

| # | 贡献内容 | 依据 |
|---|---|---|
| 1 | **Qwen3.5-0.8B 示例配置**（`examples/qwen3_5/qwen3_5_0_8B_final_config.yaml`） | MindSpeed-MM v26.1.0 tag **未自带** 0.8B 示例配置（仅 4B~397B）；缺此配置无法复现复赛基线 |
| 2 | **triton-ascend 版本配对说明**（文档补充） | CANN 9.1.0 + torch 2.7.1 + py3.10 必须用 **3.2.2**（3.2.0/3.2.1 在 910C 上回退 CPU）；且需 `python3-dev + build-essential`，否则驱动编译 `Python.h` 失败 |
| 3 | **判定链工具**（`skill_R2-SK04`） | 精度验收的机器化判定与哈希链账本，可 0-GPU 复核 |

**PR 链接**：⬜ 待提交后补充（请确认目标仓库与提交范围）

---

## 5. 复现性说明

- 环境：`python3 selfcheck_env.py` → `SELFCHECK_OK`
- 数据：llava 文件字节数 228,941,895（与官方一致）；转换产物 157,712 样本
- 训练：`torchrun --nproc_per_node 2 ... e4b_final100.yaml`（§2.2）
- 预期结果：100 步完成；50–100 步均值 430.1 ms / 中位 419.0 ms；
  第 1 步 loss 1.924621；grad_norm 50–100 均值 9.860
- 判定：`verdict_id=606056b3cb04ecbb`，
  registry `entries=5`、`REGISTRY_VERIFY_OK`