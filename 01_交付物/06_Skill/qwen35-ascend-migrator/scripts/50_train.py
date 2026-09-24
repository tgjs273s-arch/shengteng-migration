#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
50_train.py — P5 训练执行（环境自适应 + 故障诊断 + 完整日志落盘）

职责：
  1. 读 P0 环境档案 + P2 生成配置 → 组装 torchrun 命令（进程数 = world_size）
  2. 注入全部必需环境变量（缺失即失败项，见 config/env_matrix.yaml → required_env）
  3. 执行训练（默认后台 + 轮询，适配 SSH 120s 掐断；本地可 --foreground）
  4. 失败自动诊断：常见错误 → 打印对应处置建议
  5. 日志从"程序启动开始"完整落盘（交付物 #5 数据源）

用法：
  python3 scripts/50_train.py --config out/plan/train_config.yaml \
      --env out/probe/env.json --log out/train/train.log [--steps 100]

退出码：0 训练完成；1 训练失败（附诊断）；2 参数/IO 错误；3 环境不满足（无 NPU 等）
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 必需环境变量（依据 config/env_matrix.yaml → required_env；此处内置以保证自包含）
REQUIRED_ENV = [
    ("NON_MEGATRON", "true",
     "缺 → 框架误走 Megatron 路径报 ModuleNotFoundError: No module named 'megatron'"),
    ("TASK_QUEUE_ENABLE", "2",
     "实测 =1 会导致慢步回归"),
    ("ASCEND_LAUNCH_BLOCKING", "0", ""),
    ("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True", "显存碎片整理"),
    ("TRITON_CACHE_DIR", "/root/triton_cache",
     "官方脚本每次清空；我方持久化以规避重编译（报告需双口径披露）"),
]

# 常见故障 → 诊断与处置（来自 docs/PITFALLS_坑表.md 的实测结论）
def _python_hint():
    """★ 坑 37：诊断建议也要跨发行版 —— 不能在 dnf 机器上教用户跑 apt。"""
    try:
        import os as _os
        import sys as _sys
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from _envcompat import install_hint
        return install_hint("pyd")
    except Exception:
        return "安装 Python 开发头文件（dnf: python3-devel / apt: python3-dev）"


DIAGNOSTICS = [
    (r"No module named 'megatron'",
     "框架走了 Megatron 路径",
     "缺少环境变量 NON_MEGATRON=true（本脚本应已注入；若手工运行请补上）"),
    (r"libhccl\.so|Failed to load the backend extension: torch_npu",
     "CANN 环境未加载",
     "先 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`；或补 LD_LIBRARY_PATH 含 CANN lib64"),
    (r"roll back to CPU|not supported on current platform",
     "triton 回退 CPU（算子后端降级）",
     "① **先实测** triton kernel 是否可用（唯一权威判据 scripts/00b_triton_min_kernel.py）——"
     "注意 triton 与 CANN/torch/py 的配对**取决于 CANN 版本**，不要只按版本号盲判"
     "② 补 Python 头文件：%s（缺 Python.h 会导致驱动编译失败 → 静默回退 CPU）"
     "③ 重新运行 P0 探测确认 backend 状态" % _python_hint()),
    (r"out of memory|OOM|NPU out of memory",
     "显存不足",
     "降低 micro_batch_size（保持 GBS=8：mbs 减半则 gas 加倍），或开启 recompute/activation_offload"),
    (r"PackageNotFoundError|No package metadata was found for ([\w\-]+)",
     "某依赖**元数据**缺失（包可能能 import，但 importlib.metadata 查不到版本）",
     "★ 只看 import 发现不了这类问题。修法（以 protobuf 为例）：\n"
     "      python3 -m pip install --force-reinstall --no-deps protobuf\n"
     "      验证: python3 -c \"import importlib.metadata as m;print(m.version('protobuf'))\""),
    (r"Processor was not found|TypeError: argument of type 'NoneType' is not iterable",
     "★ 极具误导性的错误：报的是「模型文件」，真因通常是**缺 torchvision**",
     "AutoProcessor 在 torchvision 缺失时会**回退并抛出 NoneType TypeError**，"
     "MSMM 再包装成 'Processor was not found, please check and update your model file'\n"
     "      先别去查模型文件！按顺序验证：\n"
     "        python3 -c \"import torchvision; print(torchvision.__version__)\"\n"
     "      缺则装（必须与 torch 配对，且不要让它改动 torch）：\n"
     "        python3 -m pip install torchvision==0.22.1 --no-deps\n"
     "      另可自测：python3 -c \"from transformers import AutoProcessor;"
     "AutoProcessor.from_pretrained('/root/Qwen3.5-0.8B-hf')\""),
    (r"ModuleNotFoundError: No module named '(?!megatron)([\w\.]+)'",
     "缺少 Python 依赖（★ 坑 81：MSMM 自带 requirements.txt 未被安装）",
     "装 MSMM 自带依赖清单，而不是逐个猜包：\n"
     "      python3 -m pip install -r <MSMM>/requirements.txt\n"
     "      常见缺项：einops / pydantic（DCP 转换需要 pydantic）\n"
     "      （bringup.sh phase2 已内建该步骤；若仍缺请贴出上面第一个缺失模块名）"),
    (r"ValidationError|Configuration Details.*(?:error|Error|invalid|missing)",
     "配置校验失败",
     "配置可能被错误编辑（注意：训练日志中的 Configuration Details 是 repr，None 是字符串，"
     "不能直接当 yaml 用；请用 P2 生成或人工核对 yaml）"),
    # ★ 坑 107：**诊断规则本身产生了假阳性** —— 原规则写成 `Configuration Details|ValidationError`，
    #   而 `Configuration Details` 出现在**每一份正常训练日志**里（它就是配置 dump）。
    #   于是任何失败都会被诊断成"配置校验失败"，把排查方向带偏。
    #   **诊断规则的匹配串必须是"只在该故障下才出现"的**，不能是正常日志里也有的字样。
    # ★ 坑 113：**收尾保存崩溃**与"训练失败"是两件完全不同的事，必须给出专属诊断。
    #   症状同为 ACL 507018，但发生时机在**所有迭代都打完之后**；若按下方泛化的 ACL 规则
    #   去排查（清残留进程/查设备），方向完全错。
    #   ★ 注意匹配串的选择：**不能**用 `HCCL doesn't support gather` 作为判据 ——
    #   那是一句 UserWarning，在"走普通 DCP 载入路径"的健康运行里同样会出现
    #   （那正是坑 107 的形态：把正常日志里也有的字样当故障特征）。
    #   真正专属的是 **save 计划**（`central_plan: SavePlan`），load 侧对应的是 LoadPlan。
    (r"central_plan: SavePlan|SavePlan = distW\.reduce_scatter",
     "收尾保存撞上 HCCL gather 缺陷（★ 坑 113：训练可能已跑完，失败在其之后）",
     "**第一步先数迭代行**：若 `iteration N/N` 已到末步 → 训练本身成功，崩溃只在收尾保存。\n"
     "      根因：`dcp.save` 的 SavePlan 规划走 `distW.reduce_scatter('plan')` → `scatter_object_list`，\n"
     "      而本 CANN 上 `HCCL doesn't support gather at the moment` → aicpu 失败（ACL 507018）。\n"
     "      处置（**配置级，不需改训练代码**）：\n"
     "        ① 置 `training.save_format: hf` 绕开 DCP plan 广播；\n"
     "           注意 `trainer.py:439-448` 规定 `save_format != dcp` 时需同时\n"
     "           `no_save_optim: true` 与 `no_save_rng: true`，否则会被强制回退 dcp；\n"
     "        ② 或不需要 checkpoint 时置 `training.save: null`\n"
     "           （`train_engine.py:315` 的 `if args.training.save:` 守卫为假 → 完全不保存）。\n"
     "      ⚠ `load_rank0_and_broadcast`（坑 108）**只修 load，不影响 save** ——\n"
     "        它在 load 阶段生效会让人误以为「这个 CANN 缺陷已绕过」，直到收尾保存才炸。"),
    # ★ 坑 140：坑 108 的**真身** —— 与坑 113（收尾保存）同族、症状几乎一样，
    #   但发生在**训练开始之前**（`TrainEngine.__init__ → load()`）。
    #   ⚠ 二者只差一个词：load 侧 `central_plan: LoadPlan` / save 侧 `central_plan: SavePlan`。
    #   专属且"只在该故障下出现"的特征是那个 AI CPU kernel 名：
    #   `libscatter_aicpu_kernel.so` / `HcclLaunchAicpuKernel`（健康日志里不存在）。
    #   ★ 若按下方泛化的 `ACL 507018` 规则排查（清残留进程/查设备），方向完全错 ——
    #     实测同一台机器上 2 rank `all_reduce` 完全正常：**collective 本身没坏，
    #     坏的只是 scatter/gather 这条 AICPU 路径**。
    (r"libscatter_aicpu_kernel\.so|HcclLaunchAicpuKernel|central_plan: LoadPlan",
     "载入阶段撞上 HCCL scatter 缺陷（★ 坑 140 = 坑 108 的真身，训练一步都没跑）",
     "**第一步先数迭代行**：一个 `iteration` 行都没有 → 故障在 `TrainEngine.__init__ → load()`，\n"
     "      训练**一步都没跑到**，因此排查方向不是「训练代码/设备坏了」。\n"
     "      根因：`dcp.load` → `distW.reduce_scatter('plan')` → `scatter_object_list` →\n"
     "        `HcclLaunchAicpuKernel`（libscatter_aicpu_kernel.so）AICPU 异常，runtime 507018。\n"
     "      ⚠ 不要因为同机 `all_reduce` 正常就否定此判断：collective 本身是好的。\n"
     "      处置（**配置级，不改训练代码**）：置 `training.load_rank0_and_broadcast: true`\n"
     "        （`dcp_checkpointer.py:251` 支持）→ 改走「rank0 读取 + broadcast」，\n"
     "        绕开 `scatter_object_list`。\n"
     "      ★ 这个开关是 **P0/P2 按实测 CANN 版本自动写入的**（CANN 含 beta/RC/dev → true）。\n"
     "        若你**手工造了一份配置**（例如拿 `config/templates/` 去改），就会丢掉它 ——\n"
     "        这是本故障最常见的成因：**不是机器坏了，是配置绕过了自动档位**。\n"
     "      ★ **save 侧是另一条独立路径**（坑 113 的 `save_format: hf`）：只设 load 侧会\n"
     "        「载入正常、训练跑完 100 步才在收尾保存处 SIGABRT」。两条都要设。"),
    # ★ 坑 144：**官方数据侧参数 ≠ 通用参数**。官方 `cutoff_len: 1024` 是给 COCO **单图**
    #   样本调的；喂**多图** mock 数据时序列被截断，`<|image_pad|>` 占位符被截掉一部分，
    #   而视觉侧仍按**全部图**产出特征 → 占位符数 ≠ 特征数（真机实测 tokens 2992 / features 262144）。
    #   专属特征：`Image features and image tokens do not match`（健康日志里不存在）。
    (r"Image features and image tokens do not match",
     "图像占位符数与视觉特征数不匹配（★ 坑 144：多半是 cutoff_len 截断了多图样本）",
     "**先问「这份数据是几图样本、cutoff_len 够不够装下」**：\n"
     "      官方 `cutoff_len: 1024` 是为 COCO 单图样本调的。若数据是**多图**（或图更大），\n"
     "      序列被截断到 cutoff_len 后，`<|image_pad|>` 占位符被截掉一部分，\n"
     "      而视觉侧仍按**全部图**产出特征 → 两边计数对不上。\n"
     "      处置：把 `cutoff_len` 提到能装下**整个**样本（已验证的 mock 模板用 **2048**）。\n"
     "      ★ 这是「参数是为数据调的」的典型：**换数据就要重调数据侧参数**，不能照抄。\n"
     "      ★ 同族：坑 145（数据侧 `num_workers` 照抄官方值 → 写满 /dev/shm）。"),
    # ★ 坑 145：worker 数 × rank 数把 `/dev/shm` 写满。
    #   ⚠ 这一条里**报错文案是对的**，而坑 143 里同族文案（insufficient shm）是**错的** ——
    #   两者必须分成独立规则，否则会互相带偏。
    (r"unable to write to file.*No space left on device|out of shared memory|"
     r"killed by signal: Bus error",
     "DataLoader worker 把 /dev/shm 写满（★ 坑 145；与坑 143 同族文案、结论相反）",
     "**这一条报错文案是对的**（与坑 143 相反）：\n"
     "      官方 `num_workers: 8` × 2 个 rank = **16 个 worker** 同时把批次张量写进 `/dev/shm`，\n"
     "      把 tmpfs 写满 → `unable to write to file </torch_..._0>: No space left on device (28)`\n"
     "      → worker 以 `Bus error` 死掉。\n"
     "      处置（按顺序）：\n"
     "        ① 数据侧 `num_workers` 取**已验证模板**的值（本项目 mock 模板是 **2**）；\n"
     "        ② 确需更多 worker 时先确认容量：`df -h /dev/shm`\n"
     "           （容器内可 `mount -o remount,size=32G /dev/shm`，但需权限）。\n"
     "      ⚠ **与坑 143 的判别**：坑 143 是 `Killed` + 「insufficient shm」的**猜测**，\n"
     "        真因是 **cgroup OOM**；本条是 `Bus error` + `/torch_*` 的 **ENOSPC**，真因确实是 shm。\n"
     "        **不要按文案下结论** —— 去读该资源的记账：\n"
     "        `memory/oom_control` 的 `oom_kill` 计数 vs `df -h /dev/shm`。"),
    # ★ 坑 143：`Killed`（SIGKILL）时 torch 那句 "insufficient shared memory (shm)" 只是**猜测**。
    #   真机实测：`/dev/shm` 16 GB、用量 0%；真因是**容器 cgroup 内存撞上限**（`oom_kill` +2）。
    #   专属特征只用 `is killed by signal: Killed`（**不能**用 "insufficient shared memory" ——
    #   那句也出现在坑 145 的 Bus error 文案里，拿它当特征会让两条规则互相污染，即坑 107 复发）。
    (r"is killed by signal: Killed",
     "DataLoader worker 被 SIGKILL（★ 坑 143：torch 猜「shm 不足」，真凶通常是容器 cgroup OOM）",
     "**先读「该资源的记账文件」，再读报错文案**：\n"
     "      torch 那句 `insufficient shared memory (shm)` 是它在 worker 意外死亡时的**通用猜测**，\n"
     "      不是诊断。真机实测该提示出现时 `/dev/shm` 是 **16 GB / 用量 0%**。\n"
     "      处置（按顺序）：\n"
     "        ① 读 cgroup 记账：`cat /sys/fs/cgroup/memory/memory.oom_control`（看 `oom_kill` 增量）、\n"
     "           `memory.max_usage_in_bytes` vs `memory.limit_in_bytes`；\n"
     "        ② `oom_kill` 有增量 → **容器内存撞上限**（不是 shm）：\n"
     "           降 worker 数 / 降 mbs，或申请更大内存配额；\n"
     "        ③ 若 `oom_kill` 无增量，再去看 `/dev/shm`（那时才是坑 145）。\n"
     "      ★ 通用规矩：**报错文案点名了哪个资源，不等于那个资源有问题** ——\n"
     "        资源归属只能由**该资源自己的记账**判定。"),
    (r"ACL stream synchronize failed|error code:507018",
     "ACL 流同步失败（error 507018）—— NPU 侧执行异常",
     "常见原因与处置（按顺序排查）：\n"
     "      ① **设备残留进程占着显存/流**：查 `npu-smi info` 的进程表，清理上一次崩溃留下的进程\n"
     "         （清理时**不要用 `pkill -f` 直接匹配名字**——那条命令行自身会命中，见坑 72/105；\n"
     "          可用分片 pattern：`PAT=$(printf 'trai%s' 'ner.py'); pkill -f \"$PAT\"`）\n"
     "      ② 上一次运行以非正常方式终止，NPU 处于脏状态 → 重试一次通常即可恢复\n"
     "      ③ 确认 `nproc_per_node` 与可见 die 数一致（本机 2）\n"
     "      ④ 看 ACL 报错**之前的第一条真实异常**（本提示只描述 ACL 层，根因往往在其上方）"),
    # ★ 坑 134：**HCCL 自己的 socket 端口**与 torchrun 的 master_port 是**两条路径**。
    #   连续起两次训练时，即使 torchrun 已退出，HCCL 的 NPU socket 端口（默认 16666）
    #   仍可能被占用 → `hcclCommInitRootInfoConfig` 失败（error code 7）/ EI0020 Bind_IP_Port。
    #   这与坑 105（master_port 冲突）**同族但不同层** —— 而本脚本的自动端口规避只管 master_port。
    #   **修一条路径时要问对称的那条。**
    (r"Bind_IP_Port|HCCL_NPU_SOCKET_PORT_RANGE|hcclCommInitRootInfoConfig",
     "HCCL socket 端口被占用（★ 坑 134：与坑 105 同族，但在 HCCL 层）",
     "根因：上一次运行的 **HCCL NPU socket 端口**（默认 16666）尚未释放 →\n"
     "      `hcclCommInitRootInfoConfig(...)` 失败（error code is 7）/ EI0020 Bind_IP_Port。\n"
     "      ⚠ 本脚本的自动端口规避**只管 torchrun 的 master_port，不管 HCCL 这个端口**。\n"
     "      处置（按顺序）：\n"
     "        ① **等上一次运行完全退出后重试**（该端口会自行释放）；\n"
     "        ② 或显式换端口段：`export HCCL_NPU_SOCKET_PORT_RANGE=60500-60600`；\n"
     "        ③ **同一实验禁止连起两次**（本项目已多次自伤：坑 72/105/121 同族）。"),
    (r"Address already in use",
     "端口占用（torchrun master_port）",
     "更换 --master_port（★ 注意：本脚本只管这一条路径；HCCL 的 socket 端口是另一条，见上）"),
    (r"Address family not supported|Connection refused.*分布式|init_process_group",
     "分布式初始化失败",
     "确认 torchrun --nproc_per_node 与设备数一致；单卡环境请设 world_size=1"),
]


# ---------------------------------------------------------------- 诊断规则双向自检
# ★ INV-2：每个判据都必须被证明"喂坏输入时会失败"。**诊断规则同样是判据** ——
#   坑 107 就是"规则匹配串在正常日志里也出现"导致的假阳性，而它当时**没有任何测试**。
#   这里对每条规则喂一个**必须命中**的坏样本，并额外喂一个**健康日志样本**要求 0 命中
#   （健康样本里刻意保留了 `HCCL doesn't support gather` 警告与配置 dump —— 它们不得触发任何规则）。
DIAG_SAMPLES = [
    # (样本名, 期望命中的标题关键字（空串 = 要求 **0 命中**）, 正文)
    ("healthy 健康日志（配置 dump + 迭代行 + 良性警告）", "",
     "============ Configuration Details ============\n"
     "training:\n  micro_batch_size: 4\n  gradient_accumulation_steps: 1\n"
     "================================================\n"
     "[Rank 0 | Local Rank 0] 2026-09-16 19:13:31 INFO =>  iteration      100/     100 | "
     "consumed samples: 800 | elapsed time per iteration (ms): 934.4 | "
     "learning rate: 3.045865E-09 | global batch size:     8 | loss: 1.447652E+00 | grad norm: 10.075 |\n"
     "UserWarning: HCCL doesn't support gather at the moment. Implemented with allgather instead.\n"
     "W0916 Warning: Failed to generate log message. (function operator())\n"),
    ("megatron 未装", "Megatron", "ModuleNotFoundError: No module named 'megatron'\n"),
    ("torch_npu backend 未加载", "CANN 环境未加载",
     "Failed to load the backend extension: torch_npu\n"),
    ("算子回退 CPU", "CPU", "warning: roll back to CPU for op x\n"),
    ("OOM", "显存不足", "RuntimeError: NPU out of memory. Tried to allocate 2.00 GiB\n"),
    ("包元数据缺失", "元数据",
     "PackageNotFoundError: No package metadata was found for mindspeed-mm\n"),
    ("Processor 未找到", "极具误导性",
     "ValueError: Processor was not found for model /root/Qwen3.5-0.8B-hf\n"),
    ("缺第三方依赖", "依赖", "ModuleNotFoundError: No module named 'jsonargparse'\n"),
    ("配置校验失败", "配置校验", "jsonargparse._util.error_utils.ParserError: ValidationError\n"),
    ("★收尾保存撞 HCCL gather（坑 113）", "收尾保存",
     "UserWarning: HCCL doesn't support gather at the moment. Implemented with allgather instead.\n"
     "[rank0]:     central_plan: SavePlan = distW.reduce_scatter(\"plan\", local_step, global_step)\n"
     "[rank0]:     result = self.scatter_object(all_results)\n"
     "[rank0]: RuntimeError: ACL stream synchronize failed, error code:507018\n"),
    ("★载入撞 HCCL scatter（坑 140/108，训练一步未跑）", "载入阶段",
     "E39999[PID: 30791] 2026-09-21-11:04:02 (E39999):  The error from device(chipId:2, dieId:0), "
     "serial number is 460, an exception occurred during AICPU execution, stream_id:42, task_id:38, "
     "errcode:11006, msg:inner error.\n"
     "       AI CPU kernel execute failed, device_id=0, stream_id=42, task_id=38, "
     "soName=libscatter_aicpu_kernel.so, funcName=HcclLaunchAicpuKernel, "
     "kernelName=HcclLaunchAicpuKernel, errorCode=0x2a.\n"
     "[rank1]:   File \"/usr/local/lib/python3.11/site-packages/torch/distributed/checkpoint/utils.py\", "
     "line 217, in reduce_scatter\n"
     "[rank1]:     central_plan: LoadPlan = distW.reduce_scatter(\"plan\", local_step, global_step)\n"
     "[rank1]:   File \"/usr/local/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py\", "
     "line 3627, in scatter_object_list\n"
     "[rank1]: RuntimeError: ACL stream synchronize failed, error code:507018\n"),
    ("★多图样本被 cutoff_len 截断（坑 144）", "占位符",
     "[rank1]:     raise ValueError(\n"
     "[rank1]: ValueError: Image features and image tokens do not match, "
     "tokens: 2992, features: 262144\n"),
    ("★worker 写满 /dev/shm（坑 145，文案正确）", "/dev/shm",
     "RuntimeError: unable to write to file </torch_63792_2473652340_0>: "
     "No space left on device (28)\n"
     "RuntimeError: DataLoader worker (pid 63782) is killed by signal: Bus error. It is possible "
     "that dataloader's workers are out of shared memory. Please try to raise your shared memory "
     "limit.\n"),
    ("★worker 被 SIGKILL（坑 143，文案错误：真凶是 cgroup OOM）", "SIGKILL",
     "[rank1]: RuntimeError: DataLoader worker (pid 50440) is killed by signal: Killed. \n"
     "ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient "
     "shared memory (shm).\n"),
    ("纯 ACL 错误（无 save 计划）", "ACL",
     "[rank1]: RuntimeError: ACL stream synchronize failed, error code:507018\n"),
    ("端口占用", "端口", "OSError: [Errno 98] Address already in use\n"),
    ("★HCCL socket 端口占用（坑 134）", "HCCL socket 端口",
     "[rank0]: RuntimeError: create_config:../torch_npu/csrc/distributed/HCCLUtils.cpp:140 "
     "HCCL function error: hcclCommInitRootInfoConfig(numRanks, &rootInfo, rank, config, "
     "&(comm->hcclComm_)), error code is 7 \n"
     "[PID: 48276] Communication_Error_Bind_IP_Port(EI0020): Failed to enable listening for "
     "the NPU network adapter socket. Reason: The IP address 192.27.2.193 and port 16666 "
     "have already been bound.\n"),
    ("分布式初始化失败", "分布式", "RuntimeError: Connection refused during init_process_group\n"),
]


def diag_selftest():
    """诊断规则双向自检。返回 rc（0=全过）。"""
    print("== 诊断规则双向自检（%d 条规则 / %d 个样本）==" % (len(DIAGNOSTICS), len(DIAG_SAMPLES)))
    ok = True
    for name, want, text in DIAG_SAMPLES:
        pats = [t for p, t, _f in DIAGNOSTICS if re.search(p, text, re.I)]
        if want == "":
            good = (len(pats) == 0)          # ★ 坑 107 守卫：健康日志必须 0 命中
            detail = "0 命中 ✓" if good else ("**误报 %d 条**: %s" % (len(pats), pats))
        else:
            good = any(want.lower() in t.lower() for t in pats)
            detail = ("命中 %s" % [t[:22] for t in pats]) if pats else "**未命中（规则看不见坏样本）**"
        ok = ok and good
        print("  [%s] %-44s %s" % ("PASS" if good else "FAIL", name, detail))
    uncovered = [t for p, t, _f in DIAGNOSTICS
                 if not any(re.search(p, s[2], re.I) for s in DIAG_SAMPLES)]
    if uncovered:
        print("  ~ SKIP 无正向样本的规则 %d 条（**不算通过**）: %s" % (len(uncovered), uncovered))
    print("DIAG_SELFTEST_%s samples=%d rules=%d uncovered=%d"
          % ("OK" if ok else "FAIL", len(DIAG_SAMPLES), len(DIAGNOSTICS), len(uncovered)))
    return 0 if ok else 1


def sh(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return str(e)


def build_env_string(env):
    """拼接环境变量前缀（含 set_env.sh 与必需项）。"""
    parts = [
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null",
        "export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:"
        "/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:"
        "/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH",
    ]
    for name, val, _ in REQUIRED_ENV:
        parts.append("export %s=%s" % (name, val))
    return "; ".join(parts)


def diagnose(log_path):
    """扫描日志中的已知故障模式，返回 [(标题, 原因, 处置)]。"""
    try:
        text = open(log_path, encoding="utf-8", errors="replace").read()
    except Exception:
        return []
    hits = []
    for pat, title, fix in DIAGNOSTICS:
        if re.search(pat, text, re.I):
            hits.append((title, pat, fix))
    return hits


def count_iters(log_path):
    try:
        text = open(log_path, encoding="utf-8", errors="replace").read()
        return len(re.findall(r"iteration\s+\d+\s*/\s*\d+", text))
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser(description="P5 训练执行（环境自适应）")
    ap.add_argument("--config", default="out/plan/train_config.yaml")
    ap.add_argument("--env", default="out/probe/env.json")
    ap.add_argument("--log", default="out/train/train.log")
    ap.add_argument("--workdir", default="/root/MindSpeed-MM", help="MindSpeed-MM 目录")
    ap.add_argument("--port", type=int, default=6111)
    ap.add_argument("--steps", type=int, default=None, help="覆盖配置中的 train_iters")
    ap.add_argument("--timeout", type=int, default=5400, help="训练超时（秒）")
    ap.add_argument("--foreground", action="store_true", help="前台执行（本地/短任务）")
    ap.add_argument("--dry-run", action="store_true", help="只打印将执行的命令")
    # ★ 坑 104：GBS=8 是硬红线，本脚本启动前会拦截 GBS≠8；此开关仅用于"流程连通性验证"
    ap.add_argument("--allow-gbs-mismatch", action="store_true",
                    help="显式允许 GBS≠8（结果不可用于官方精度对标）")
    # ★ 坑 103：显式指定并行进程数（覆盖 P0 档位推断），便于在探测失误时人工纠正
    ap.add_argument("--world-size", type=int, default=None,
                    help="覆盖 P0 推断的 world_size（进程数）")
    ap.add_argument("--diag-selftest", action="store_true",
                    help="只跑诊断规则双向自检并退出（好样本必中 / 健康日志必须 0 命中）")
    args = ap.parse_args()

    # ★ INV-2：诊断规则也是判据，必须可证伪（含坑 107 的"健康日志 0 命中"守卫）
    if args.diag_selftest:
        return diag_selftest()

    if not os.path.isfile(args.config):
        print("FATAL 配置不存在：%s（先跑 scripts/20_plan_migration.py）" % args.config, file=sys.stderr)
        return 2

    env = {}
    if os.path.isfile(args.env):
        try:
            env = json.loads(open(args.env, encoding="utf-8").read())
        except Exception:
            env = {}
    profile = env.get("recommended_profile", {})
    world = int(profile.get("world_size") or 1)
    # ★ 坑 103：允许显式覆盖（探测失误时的人工纠正通道）
    if getattr(args, "world_size", None):
        print("!! 人工覆盖 world_size: %s → %s（P0 推断值可能不可靠）"
              % (world, args.world_size))
        world = int(args.world_size)
    path = env.get("path", "unknown")

    print("== P5 训练执行 ==")
    print("环境路径 : %s | world_size=%s" % (path, world))

    # ★ 坑 104：**GBS=8 是本项目的硬红线，但代码里从来没人检查它。**
    #   实测 P5 用 world_size=1 跑起来，日志显示 `global batch size: 4` —— **违反了红线却无人报警**。
    #   （根因是坑 103：HBM 探测失败 → 档位掉到 32GB 档 → world_size=1。）
    #   现把红线**变成可执行的断言**：从配置读 mbs/gas，算 GBS = mbs × gas × world，
    #   不等于 8 即**大声拒绝执行**（除非显式 --allow-gbs-mismatch）。
    gbs_info = {"mbs": None, "gas": None, "gbs": None, "source": None}
    # ★ 坑 111：旧实现用**正则扫文本**取 mbs/gas —— 配置若是非法 YAML（坑 109 的
    #   缩进/块映射损坏），正则照样能匹配到数字 → 红线"检查通过"，然后训练器在
    #   几分钟后抛 `ConfigValidationError`。**运行条件读原始文本、验收条件读解析结果**，
    #   两者不同源。现在：先解析 YAML，解析不了就在烧 GPU 之前拒启动。
    try:
        import yaml as _yaml
    except ImportError:
        _yaml = None
    _cfg_txt = open(args.config, encoding="utf-8", errors="replace").read()
    _cfg_doc, _parse_err = None, None
    if _yaml is not None:
        try:
            _cfg_doc = _yaml.safe_load(_cfg_txt)
            if not isinstance(_cfg_doc, dict):
                _parse_err = "顶层不是映射（got %s）" % type(_cfg_doc).__name__
        except Exception as _e:
            _parse_err = str(_e)
    if _parse_err:
        print("")
        print("!! 配置 YAML 无法解析 —— **在启动训练前拒绝执行**")
        print("   配置: %s" % os.path.abspath(args.config))
        print("   原因: %s" % _parse_err)
        print("   这属于坑 109 类故障（逐行改 yaml 导致缩进/块映射损坏）。")
        print("   处置: 用 P2 重新生成：`python3 scripts/20_plan_migration.py ...`")
        print("         （P2 现已**先校验后落盘**，产不出坏配置）")
        return 3

    def _as_int(v):
        if v is None or isinstance(v, bool):
            return None
        try:
            return int(v)
        except Exception:
            return None

    if _cfg_doc is not None:
        _tr = _cfg_doc.get("training") or {}
        _pl = _cfg_doc.get("parallel") or {}
        gbs_info["mbs"] = _as_int(_tr.get("micro_batch_size"))
        gbs_info["gas"] = _as_int(_tr.get("gradient_accumulation_steps"))
        gbs_info["source"] = "yaml"
        _dp = _as_int(_pl.get("data_parallel_size"))
        # ★ 同源原则：配置声明的 dp 必须与本次实际 world 一致，否则"GBS=8"用哪个数
        #   算都不诚实（这正是坑 110 让判定链必然 NOT_COMPARABLE 的同源漏洞）。
        if _dp is not None and _dp != max(1, world):
            print("")
            print("!! 配置声明的 data_parallel_size=%s 与本次运行 world=%s **不一致** —— 拒绝启动"
                  % (_dp, max(1, world)))
            print("   同源原则：跳过/验收/运行/闭环条件必须来自同一事实。")
            print("   处置: 用 P2 重新生成配置（按 P0 探测到的档位写入 dp），或修正 --world-size。")
            return 3
    else:
        # 降级路径：无 pyyaml 时只能行扫描，**明确标注为降级**，不假装与解析等价
        import re as _re

        def _g(k):
            m = _re.search(r"(?m)^\s*%s\s*:\s*(\S+)" % k, _cfg_txt)
            if not m:
                return None
            try:
                return int(str(m.group(1)).strip("\"'"))
            except Exception:
                return None

        gbs_info["mbs"] = _g("micro_batch_size")
        gbs_info["gas"] = _g("gradient_accumulation_steps")
        gbs_info["source"] = "regex_fallback(no_pyyaml)"
        print("⚠ 未安装 pyyaml → GBS 自检降级为正则行扫描（不等价于解析校验）")

    if gbs_info["mbs"] and gbs_info["gas"]:
        gbs_info["gbs"] = gbs_info["mbs"] * gbs_info["gas"] * max(1, world)
    print("几何自检 : mbs=%s × gas=%s × world=%s = GBS %s（官方红线 = 8；取值来源 %s）"
          % (gbs_info["mbs"], gbs_info["gas"], world, gbs_info["gbs"], gbs_info["source"]))
    if gbs_info["gbs"] is not None and gbs_info["gbs"] != 8:
        if getattr(args, "allow_gbs_mismatch", False):
            print("⚠ 已显式允许 GBS≠8（--allow-gbs-mismatch）：结果**不可用于官方精度对标**")
        else:
            print("")
            print("!! GBS ≠ 8 —— 违反项目硬红线，**拒绝执行**（本脚本在启动训练前主动拦截）")
            print("   官方验收要求 GBS = mbs × gas × dp = 8；当前为 %s。" % gbs_info["gbs"])
            print("   常见原因：")
            print("     · P0 探测的档位选择错误（如 HBM 探测失败 → 掉到 32GB 档 → world_size=1）")
            print("       → 检查 out/probe/env.json 的 recommended_profile（world_size/mbs/gas）")
            print("     · 或配置文件里的 mbs/gas 与选定的并行度不匹配")
            print("   绕过（仅用于流程连通性验证）: 加 --allow-gbs-mismatch")
            return 3
    if not env.get("capabilities", {}).get("can_train", False):
        print("⚠ 当前环境不具备训练能力（capabilities.can_train=false，通常是 %s）" % path)
        print("  本阶段按设计**降级跳过**：不执行训练，不产出性能数字（诚实红线）")
        print("  可执行阶段：P1 静态分析 / P2 方案生成 / P3 shape 契约 / P7 判定链（0-GPU）")
        return 3

    # 组装命令
    log_path = os.path.abspath(args.log)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cfg_arg = os.path.abspath(args.config)

    # ★ 坑 105：**端口冲突导致训练启动失败**（`Address already in use`）。
    #   根因有二：① 上一次运行的残留进程仍占用端口；② 我想用 `pkill -f` 清理，
    #   而那条 SSH 命令行本身就含该模式 → **pkill 匹配并杀掉了自己的 shell**（坑 72 复发，破坏性版本）。
    #   机制化解法：**自动挑选一个空闲端口**，不再依赖"记得先清理"。
    port = args.port

    def _port_free(p):
        """★ 坑 106：端口可用性探测**不能只用 bind**，且**绝不能设 SO_REUSEADDR** ——
        在 Windows 上 SO_REUSEADDR 允许重新绑定一个已被占用的端口，于是探测恒返回"空闲"
        （本地自测就因此失败，误判率 100%）。改为两段判断：
          ① **先尝试连接**：能连上 = 有服务在监听 = 已占用（跨平台最可靠）
          ② 再尝试 bind（**不设 SO_REUSEADDR**）
        """
        import socket as _s
        c = _s.socket(_s.AF_INET, _s.SOCK_STREAM)
        c.settimeout(0.4)
        try:
            if c.connect_ex(("127.0.0.1", p)) == 0:
                return False           # 有人在监听
        except Exception:
            pass
        finally:
            try:
                c.close()
            except Exception:
                pass
        s = _s.socket(_s.AF_INET, _s.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", p))   # 刻意不设 SO_REUSEADDR
            return True
        except OSError:
            return False
        finally:
            try:
                s.close()
            except Exception:
                pass

    if not _port_free(port):
        print("!! 端口 %d 被占用 → 自动寻找空闲端口（避免 'Address already in use'）" % port)
        for cand in range(port + 1, port + 200):
            if _port_free(cand):
                port = cand
                break
        print("   改用端口: %d" % port)
    steps_note = ("（train_iters 已由配置决定，--steps=%s 仅作提示）" % args.steps) if args.steps else ""
    # ★ 坑 71：注入 PYTHONPATH 含 MSMM 目录 —— 即使 mindspeed_mm 未 `pip install -e`，
    #   在 workdir 下也能导入（兜底）。`mindspeed` 核心包仍必须真的安装（见 bringup.sh phase2）。
    train_cmd = ("cd %s && PYTHONPATH=%s:${PYTHONPATH:-} timeout %d torchrun "
                 "--nproc_per_node %d --nnodes 1 --node_rank 0 "
                 "--master_addr localhost --master_port %d "
                 "mindspeed_mm/fsdp/train/trainer.py %s > %s 2>&1"
                 % (args.workdir, args.workdir, args.timeout, world, port,
                    cfg_arg, log_path))
    # ★ 坑 167（真机实测 2026-09-21）：原写法末尾是 `; echo train_rc=$?` ——
    #   **最后一条命令是 echo，它永远成功** ⇒ `subprocess.call` 返回的 `rc` **恒为 0**，
    #   于是下面 `if rc == 0 and n > 0` 把一次以 `ChildFailedError` 收尾的运行判成 `TRAIN_OK`
    #   （现场：`TRAIN_OK iters=96` 而日志结尾是 `trainer.py FAILED`）。
    #   而 `59`/`61` 是从 driver.log 里抓 `train_rc=(\d+)`（那个值是真 1）⇒ 两者**结论相反**。
    #   修法：让 shell **用训练的真实返回码退出**，使"打印的值"与"进程返回码"同源。
    full = "bash -lc '%s; %s; rc=$?; echo train_rc=$rc; exit $rc'" % (
        build_env_string(env).replace("'", "'\\''"), train_cmd.replace("'", "'\\''"))

    print("训练命令 : %s" % train_cmd)
    print("日志     : %s %s" % (log_path, steps_note))
    if args.dry_run:
        print("\n[dry-run] 将执行：\n%s" % full)
        return 0

    t0 = time.time()
    if args.foreground:
        print("前台执行中（超时 %ds）..." % args.timeout)
        rc = subprocess.call(full, shell=True)
    else:
        # 后台 + 轮询（适配 SSH 120s 掐断：即便会话断开，nohup 进程继续）
        runner = "/tmp/_p5_train_runner.sh"
        with open(runner, "w") as f:
            f.write("#!/bin/bash\n%s\n" % full.replace("bash -lc '", "").rstrip("'"))
        os.system("chmod +x %s" % runner)
        os.system("setsid nohup bash %s > /tmp/_p5_runner.log 2>&1 < /dev/null &" % runner)
        print("已后台启动（setsid nohup）。轮询进度中...")
        rc = None
        last = 0
        while True:
            time.sleep(20)
            n = count_iters(log_path)
            el = int(time.time() - t0)
            if n != last:
                print("  [%4ds] iteration 行数=%d" % (el, n))
                last = n
            wrap = ""
            try:
                wrap = open("/tmp/_p5_runner.log", encoding="utf-8", errors="replace").read()
            except Exception:
                pass
            m = re.search(r"train_rc=(\d+)", wrap)
            if m:
                rc = int(m.group(1))
                break
            if el > args.timeout + 300:
                print("轮询超时（%ds），训练进程可能仍在运行；请查看日志" % el)
                rc = None
                break

    n = count_iters(log_path)
    print("\n=== 训练结果 ===")
    print("iteration 行数 : %d" % n)
    print("耗时           : %ds" % int(time.time() - t0))
    print("日志           : %s" % log_path)

    # ★ 坑 167 的第二道护栏：`TRAIN_OK` 必须**同时**满足"返回码为 0 + 有 iteration 行 +
    #   日志里没有任何失败标志"。只靠返回码是不够的（返回码本身刚刚错过一次），
    #   而只靠"有没有 iteration 行"更不够（跑到一半崩掉的日志同样有 iteration 行）。
    _ltext = ""
    try:
        _ltext = open(log_path, encoding="utf-8", errors="replace").read()
    except Exception:
        pass
    failure_markers = [m for m in ("ChildFailedError", "trainer.py FAILED",
                                   "Traceback (most recent call last)", "ERR00006",
                                   "ERR99999", "OutOfMemoryError")
                       if m in _ltext]
    if failure_markers:
        print("日志含失败标志：%s" % failure_markers)

    if rc == 0 and n > 0 and not failure_markers:
        # 输出末尾几行供快速确认
        try:
            tail = [l for l in open(log_path, encoding="utf-8", errors="replace") if "iteration" in l][-2:]
            for l in tail:
                print("  " + l.strip()[:200])
        except Exception:
            pass
        print("TRAIN_OK iters=%d log=%s" % (n, log_path))
        print("下一步: P6 性能 → python3 scripts/60_bench.py --log %s" % log_path)
        return 0

    print("TRAIN_FAIL rc=%s iters=%d" % (rc, n))
    hits = diagnose(log_path)
    if hits:
        print("\n=== 自动诊断 ===")
        for title, pat, fix in hits:
            print("● %s" % title)
            print("    匹配: %s" % pat)
            print("    处置: %s" % fix)
    else:
        print("未匹配已知故障模式，请查看日志尾部：")
        try:
            for l in open(log_path, encoding="utf-8", errors="replace").read().splitlines()[-15:]:
                print("  " + l[:200])
        except Exception:
            pass
    return 1


if __name__ == "__main__":
    sys.exit(main())
