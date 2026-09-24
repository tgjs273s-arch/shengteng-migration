# -*- coding: utf-8 -*-
r"""_patch_replay_results.py —— 把一次回放的**实测结果**并入协议（判据一字不动）"""
import io
import json
import os
import shutil
import time

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "qwen35-ascend-migrator_整合版", "protocols", "one_step_replay_20260922.json")
KEY = "results_2026_09_22"
d = json.load(io.open(P, encoding="utf-8"))
if KEY in d:
    print("PATCH_SKIP 已并入")
    raise SystemExit(0)
d[KEY] = {
    "status": "已实测（两轮同配置、固定完整状态；两 rank 结论一致）",
    "runs": {
        "root": "/root/ops/replay_20260922_111356",
        "a": "a/（rc=0）", "b": "b/（rc=0）",
        "machine": "199.98.58.213（跳板 113.47.8.48:2234）",
        "config": "/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml（GBS=8）",
        "command_note": "启动命令照 50_train.py 的 --dry-run 原文，仅换入口为 52_replay.py；"
                        "只跑一步（第一次 optimizer.step 后抛出 _StopReplay 跳出），不进入 100 步循环",
        "artifacts": {
            "a_dump_rank0": "1194571 B sha256=51c3767e8ccf",
            "b_dump_rank0": "1193485 B sha256=6a9a44afcea5",
            "diff": "diff.txt 4034 B sha256=e12bf0d6e783（本地已存 远端证据_20260922/replay/）"
        }
    },
    "preconditions_identical": {
        "input_sha_all": "7a99c6b2031e4049（两轮相同）",
        "params_before_sha_all": "rank0 945de97b1d4be973 / rank1 3f10e13c5b44c2e0（两轮相同）",
        "rng_cpu_sha": "36ef520e19dc7020（两轮相同）",
        "lr": "0.0（两轮相同，见下 caveat）",
        "iteration": "0（两轮相同）"
    },
    "verdict": {
        "C1_forward": "★ **逐字节相同**（loss sha 4627dcf31a3c、logits sha b09acedeea5b88f1；两轮两 rank 都相同）",
        "C2_pre_comm_grad": "UNVERIFIED（FSDP2 在 backward 内部 reduce-scatter，用户侧不可观测；未用 C3 顶替）",
        "C3_post_comm_grad": "★★ **不同**（sha_all rank0 72448bb123d14d47 ≠ 223de54e52741f6d；"
                             "逐张量不同 200/200；抽样元素最大相对差 rank0 106.75%、rank1 379.65%）",
        "C4_after_one_update": "逐字节相同，但**无区分力**（见 caveat）",
        "first_diff": "**C3**（两 rank 一致）⇒ 按判定树：**分岔起源于反向传播段**"
                      "（在 backward 之内；因 C2 不可观测，无法进一步区分"
                      "\"反向计算本身\"与\"反向中的集合通信\"）"
    },
    "caveats": [
        "★ **C4 在本配置下没有区分力**：`lr = 0.0`（warmup 起点）⇒ 第一次 `optimizer.step()` 是**空更新**，"
        "`params_after_sha_all == params_before_sha_all`（两轮皆然）。"
        "⇒ C4 的『相同』**不能**作为『优化器路径可复现』的证据；若将来需要走 C4 分支，"
        "必须改用**步 0 学习率非零**的配置（并在协议里声明该改动），否则该判据形同虚设。",
        "逐元素口径为**固定抽样**（每张量 head64 + 固定种子 rand64），不是全量；"
        "`sha_all` 才是全量字节判等。抽样里 `embed_tokens.weight` 的梯度显示 0 处不同，"
        "说明差异**不是均匀分布**在所有权重上——不能把 200/200 读成\"每个张量都在抽样点上不同\"。",
        "只覆盖**一步**；不推出 100 步后权重是否逐位相同（误差会累积）。",
        "本运行是诊断运行（最外层包了一层只读观测），`plan_consistent=false`，不得用于性能结论。"
    ],
    "why_it_matters": [
        "在此之前只有两条弱结论：『100 步会分叉』与『某些开关能压住』。"
        "现在有了**段落级定位**：同输入、同初始权重、同 RNG、同学习率下，"
        "**前向逐字节相同、通信后梯度不同** ⇒ 分歧产生于反向段。",
        "这与机制证据相互印证：`use_deter_comp=true` 会设 `HCCL_DETERMINISTIC=True` 与 "
        "`CLOSE_MATMUL_K_SHIFT=1`（关掉 matmul K 轴 shift）——两者都作用在计算/通信的数值路径上，"
        "而本实验把『前向没问题』排除了，把嫌疑收缩到**反向的 matmul/集合通信**这一侧。"
    ],
    "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")
}
bak = P + ".bak_results_%s" % time.strftime("%Y%m%d_%H%M%S")
# ★ 备份写到 Skill **树外**（坑 201：树内备份会被打进交付包）
out_bak_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_backup", "skill_baks")
os.makedirs(out_bak_dir, exist_ok=True)
shutil.copy2(P, os.path.join(out_bak_dir, os.path.basename(bak)))
json.dump(d, io.open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
back = json.load(io.open(P, encoding="utf-8"))
assert KEY in back and "criteria" in back and "fixed_state" in back, "回读校验失败"
print("PATCH_OK 已并入 %s；status=%s；判据未动" % (KEY, back["status"]))
