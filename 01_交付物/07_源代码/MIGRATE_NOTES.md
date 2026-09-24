# MIGRATE_NOTES —— 迁移说明（每处改动的依据）

> C4-AI 2026 昇腾赛道复赛 · Qwen3.5-0.8B 昇腾迁移
> 目标环境：1× 昇腾 910C 双 die（Ascend910_9382，2×64GB）· CANN 9.1.0 · MindSpeed-MM v26.1.0
> 迁移方式：**配置驱动 + 环境适配**（未修改 MindSpeed-MM 框架源码）

---

## 一、迁移方式说明（重要）

本项目**没有修改 MindSpeed-MM 框架源码**。原因如下：

1. MindSpeed-MM v26.1.0 已内置 Qwen3.5 模型适配与 GatedDeltaNet（GDN）的 **NPU Triton 实现**，
   上游能力足以支撑迁移；官方亦确认"GDN 用 triton 实现即可，精度对齐 triton 版本"
   （2026-09-07 官方 QQ 群答复）。
2. 迁移的实质工作是**环境适配**（CANN/torch/triton 版本配对、系统依赖）+ **训练配置对齐**
   （并行几何、显存策略、算子后端、数据管线），这些均在官方配置项范围内完成。
3. 选择"不改源码"是**工程上的有意决策**：上游源码改动会增加评委复现成本与版本漂移风险；
   通过配置达成可复现性更强，且便于将来把成果回馈上游（见 §6 PR 计划）。

因此本包的"源代码"= **训练配置 + 运行/准备脚本 + 验证工具（判定链）**，
另附初赛阶段的 AST 迁移点分析 POC（说明迁移范围的程序化界定方法）。

---

## 二、环境适配改动（每项含依据）

### 2.1 系统依赖

| 项 | 改动 | 依据 |
|---|---|---|
| `python3-dev` + `build-essential` | **必须安装** | triton-ascend 驱动首次初始化需现场编译 C++ 桥接；缺 `Python.h` 会导致 backend 初始化中止 → 日志出现 `roll back to CPU`，算子静默回退 CPU 实现（性能骤降但结果仍正确，极难察觉）。实测：安装前 triton 不可用，安装后最小 kernel 通过 |
| `pip` 引导 | 镜像无 pip/ensurepip → `curl get-pip.py` | CANN devel 镜像特性（实测 `No module named ensurepip`） |

### 2.2 版本配对（关键）

| 组件 | 采用版本 | 依据 |
|---|---|---|
| CANN | 9.1.0 | 官方镜像自带 |
| torch / torch_npu | 2.7.1 / 2.7.1.post10 | 与 CANN 9.1.0 官方配对 |
| **triton-ascend** | **3.2.2** | **实测关键项**：910C（9382）+ CANN 9.1.0 + torch 2.7.1 + Python 3.10 的官方配对版本为 3.2.2。3.2.0/3.2.1 在本环境会回退 CPU（经实测验证）；且 3.2.2 需从 huaweicloud ascend 源安装并补 `attrs==24.2.0` |
| transformers | 5.2.0 | MindSpeed-MM v26.1.0 配套版本 |
| MindSpeed-MM | v26.1.0（tag） | 勿用默认分支（其 pyproject 要求 torch 2.10，与本栈冲突） |
| MindSpeed | 0.12.1 | MSMM 核心依赖；FSDP 训练路径**不需要 Megatron-LM** |

### 2.3 必需环境变量

| 变量 | 值 | 缺失后果 |
|---|---|---|
| `source set_env.sh` | CANN toolkit 环境 | `libhccl.so: cannot open shared object file` |
| `NON_MEGATRON` | `true` | 框架误走 Megatron 路径 → `ModuleNotFoundError: No module named 'megatron'` |
| `TASK_QUEUE_ENABLE` | `2` | **实测 =1 会导致周期性慢步回归**（性能劣化） |
| `ASCEND_LAUNCH_BLOCKING` | `0` | — |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | 显存碎片整理 |
| `TRITON_CACHE_DIR` | `/root/triton_cache` | 无缓存则每步重复编译 kernel（首步骤慢）；注：官方脚本每次清空，故性能报告需双口径披露 |

---

## 三、训练配置改动（每项含实验依据与收益）

配置文件：`01_training_config/qwen3_5_0_8B_optimized.yaml`（最终验收配置）

| # | 配置项 | 基线值 | 迁移后值 | 依据与收益 |
|---|---|---|---|---|
| 1 | `gdn_implementation`<br>`causal_conv1d_implementation` | `eager` | **`triton`** | GDN 走 NPU Triton 实现。日志实证 `use NPU triton ops` 36 行、`roll back to CPU` **0** 行。<br>**收益：单步 21,393 → 6,213 ms（3.44×）** |
| 2 | `data_parallel_size` / 进程数 | 1 | **2（双 die）** | 实测本机为 1 张 910C 双 die 卡，与官方基线拓扑一致。复刻官方几何后每步 batch 组成与官方物理相同 → **逐点可比**。<br>**收益：752 → 549.8 ms（1.37×）+ 精度可比性** |
| 3 | `micro_batch_size` / `gradient_accumulation_steps` | 2 / 4 | **4 / 1** | GBS=8 为官方红线（不可动）；profiling 证明瓶颈是 **kernel 发射次数**，而 kernel 数 ∝ micro-batch 个数 → 减少 micro 数即减少发射。<br>**收益：6,213 → 3,051 ms（2.04×）** |
| 4 | `features.recompute` | `true` | **`false`** | 该项为 32GB 显存环境设计；128GB 卡上实际激活占用仅约 230MB，重算属纯开销。**step1 loss 逐位不变**（证明不影响数值路径）。<br>**收益（含 #5/#6）：549.8 → 451.0 ms（1.22×）** |
| 5 | `features.enable_chunk_loss` | `true` | **`false`** | 同上（分块计算 loss 的显存优化在 128GB 下无收益） |
| 6 | `features.enable_activation_offload` | `true` | **`false`** | 同上（激活 CPU 往返搬运的开销在显存充足时纯属浪费） |
| 7 | `fsdp_plan.pregather` | `false` | **`true`** | 前向前预收集 FSDP 参数，消除**周期性 3.7s 加载阻塞慢步**（100 步中由 8 个降至 1 个）。<br>**收益：451.0 → 419.0 ms（1.08%）+ 消除长尾** |
| 8 | `fsdp_plan.num_to_forward/backward_prefetch` | 1 / 1 | **1 / 1（保持）** | **实测加深到 2/2 反而慢约 40 ms**（单变量对照实验，5 变体 × 20 步），故保持默认 |
| 9 | `init_model_with_meta_device` | `true` | **`true`（保持）** | 保持 FSDP 路径；实测改为 `false`（DDP 分支）会触发框架未完成的 DDP 路径报错，故不采用 |

> 上述 1-9 项的合并效果：**21,393 ms → 419.0 ms（51.1×）**，且最终 step1 loss 与官方基线逐位一致
> （1.924621 = 1.924621）。

---

## 四、数据与权重管线（依据）

| 环节 | 做法 | 校验依据 |
|---|---|---|
| 权重 | ModelScope `Qwen/Qwen3.5-0.8B`（官方确认与基线同款）→ DCP 转换 | 官方答复："两个是一样的，这个差距是正常范围内" |
| 提示词数据 | `AI-ModelScope/LLaVA-Instruct-150K` 的 `llava_instruct_150k.json` | **字节数 228,941,895 = 官方教程一致** |
| 图片数据 | `PAI/COCO2017` 的 `train2017.zip`（118,287 张全量） | 图片数校验 |
| 格式转换 | **官方脚本** `llava_instruct_2_mllm_demo_format.py` | 产物 **157,712 样本**，文件名 `output_llava_coco_data.json` 与官方基线声明一致 |
| 数据序 | `shuffle=false`、`seed=42` | 官方同值（精度对比前提） |

**数据同源性的最强证据**：第 1 步（纯前向，lr=0）loss 与官方逐位相同 →
说明权重、前 8 个样本的内容与顺序、算子数值实现三者同时与官方一致。

> 诚实说明：官方数据文件的 sha256 未公开，故判定链的 `data_identity` 证据门保持 open
> （判定结果为 `NEEDS_EVIDENCE`，非"绿档"）。已通过官方渠道问询哈希。

---

## 五、复现步骤

```bash
# 1) 环境自检（0-GPU 可跑）
python3 02_launch_scripts/selfcheck_env.py          # 期望 SELFCHECK_OK

# 2) 数据准备（约 18GB 下载 + 转换）
bash 03_data_pipeline/prepare_data.sh /root/data

# 3) 权重转换（hf → DCP）
bash 04_weight_convert/convert_hf_to_dcp.sh /root/Qwen3.5-0.8B-hf /root/Qwen3.5-0.8B-dcp

# 4) 训练 100 步（最终验收配置）
bash 02_launch_scripts/train_100steps.sh

# 5) 精度/配置判定（0-GPU，可复核全部结论）
python3 05_judge/scripts/fingerprint_cfg.py      --config-from-log <train.log> --baseline officialB
python3 05_judge/scripts/fingerprint_observed.py --log <train.log> --declared <fp.json>
python3 05_judge/scripts/judge_comparable.py     --run <fp.json> --observed <obs.json> --baseline officialB
```

---

## 六、上游贡献计划（开源 PR）

迁移过程中发现的可回馈上游的实质内容：

| # | 内容 | 依据 |
|---|---|---|
| 1 | **Qwen3.5-0.8B 示例配置** | MindSpeed-MM v26.1.0 tag **未自带** 0.8B 示例（仅有 4B~397B），缺失该配置无法复现复赛基线 |
| 2 | **triton-ascend 版本配对与前置依赖说明** | CANN 9.1.0 + torch 2.7.1 + py3.10 需 triton-ascend **3.2.2**（3.2.0/3.2.1 回退 CPU），且需 `python3-dev + build-essential`；该组合在公开资料中无记载，为实测得出 |
| 3 | 判定链工具（配置指纹 + 五态裁决 + 哈希链账本） | 可 0-GPU 复现的精度验收工具，供社区复用 |

---

## 七、本包与交付报告的对应关系

| 本包内容 | 对应报告/结论 |
|---|---|
| `01_training_config/*.yaml` | 性能测试报告 §优化点 1-8；精度分析报告 §对齐条件 |
| `05_judge/` | 精度分析报告 §判定链（verdict_id、registry 哈希链） |
| `06_original_poc/` | 创意书 §技术路线（AST 迁移点识别） |
| `02_launch_scripts/`, `03_data_pipeline/` | README §运行启动脚本 |

---

*本说明中的所有数字均可在交付物 #5 原始日志中复核；判定结论可通过 `05_judge/` 离线复跑验证。*
