# -*- coding: utf-8 -*-
r"""_patch_batch4_results.py —— 把四臂批次结果与"噪声底噪"结论并入协议（判据不动）"""
import io
import json
import os
import shutil
import time

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "qwen35-ascend-migrator_整合版", "protocols", "prefetch_depth_20260922.json")
d = json.load(io.open(P, encoding="utf-8"))
if "batch4_2026_09_22" in d:
    print("PATCH_SKIP 已并入")
    raise SystemExit(0)
d["batch4_2026_09_22"] = {
    "why": "上一轮（prefetch_depth 的 3 档 × 2 轮）提示『预取 4 省 3.91%』。为了确认它、并检验"
           "『每步张量格式化同步』这条 with-stack 线索，同一批次内跑 4 臂 × 2 轮。",
    "arms": {"a": "基线（预取 1，log_interval 1）", "b": "预取 4（log1）",
             "c": "log_interval=10（预取 1）", "d": "预取 4 + log10"},
    "run": {"base": "/root/ops/batch4_20260922_122428", "all_rc": "8/8 rc=0",
            "logs_local": "远端证据_20260922/batch4/（16 文件，双侧 SHA256 校验）",
            "note": "log_interval 原不在 yaml 中，但它是框架真实字段（training_args.py:299），"
                    "派生脚本已显式添加并打印说明"},
    "per_run_median_ms_11_100": {"a_r1": 707.7, "a_r2": 725.4, "b_r1": 728.8, "b_r2": 715.0,
                                 "c_r1": 724.3, "c_r2": 742.1, "d_r1": 740.1, "d_r2": 730.4},
    "per_arm_median_ms": {"a": 716.5, "b": 721.9, "c": 733.2, "d": 735.2},
    "saving_pct_vs_a": {
        "b_prefetch4": {"median": -0.76, "per_round": [-1.72, 0.21], "same_direction": False},
        "c_log10": {"median": -2.33, "per_round": [-1.09, -3.57], "same_direction": True},
        "d_prefetch4_log10": {"median": -2.62, "per_round": [-3.29, -1.94], "same_direction": True}
    },
    "verdict": "**没有任何一臂优于基线**：预取 4 本轮为 -0.76%（方向不一致 ⇒ UNCERTAIN，"
               "**上一轮的 3.91% 未复现**）；`log_interval=10` 稳定慢 -2.33%；组合慢 -2.62%。"
               "⇒ 按冻结判据：**本轮无候选收益**，这两类开关都不写入推荐。",
    "key_measurement_finding": {
        "noise_floor": "★ 同机同会话、跨批次的两轮基线中位数：批次3 = 731.6 ms、批次4 = 716.5 ms ⇒ "
                       "**批次间漂移约 2%**。",
        "implication": "上一轮报的『预取 4 省 3.91%』正好落在 2% 漂移的 2 倍以内，且本轮未复现 "
                       "⇒ 它只能算**噪声量级**，不构成收益。**这解释了为什么 ≥5% 的门槛是必要的**："
                       "本设置的噪声底噪决定了任何 <5% 的『改善』都不可报告。",
        "rule_added": "今后任何小于**噪声底噪 2 倍**的效应，一律先按 UNCERTAIN 处理；"
                      "要报告小效应必须增加重复数并给出区间（本轮 n=2 只够做方向判断）。"
    },
    "other_findings": {
        "log_interval_semantics": "`log_interval=10` 时 iter_ms 的**可观测点从 90 个降到 9 个**（窗口 11..100），"
                                  "且首点 loss 是第 10 步的值（0.0357）而非第 1 步（0.0969）⇒ "
                                  "**它是性能对照臂，不能当数值协议的等价产物**（协议 criteria 已要求声明此代价）。",
        "log10_slower": "把同步去掉反而慢 2.33%（两轮同向）——说明『每步 loss 格式化』并不是那条 46.6% "
                        "`wait_event` 的主要来源（该线索**未被本轮支持**，不得再作为优化方向）。"
    },
    "next_with_remaining_budget": [
        "① 开关类候选（预取深度、log_interval）**均已排除**（无 ≥5% 且低于噪声底噪）；",
        "② 若要继续，只剩**代码级**候选：`clip_grad=0.0` 时仍每步调用 `clip_grad_norm`（梯度范数归约）、"
        "`average_losses_across_data_parallel_group` 每步 all_reduce、以及小算子/拷贝/cast 簇的融合 —— "
        "这些都要改代码、先立协议、且预期增益大概率也低于 5%（须先算清噪声底噪与重复数）；",
        "③ 现实建议：把**已有的 A vs B 组合证据（19.6%，同批次同机）**作为性能主结论，"
        "把新开关探索记为『已排查、均低于噪声底噪』（这本身是有价值的信息）；",
        "④ 若要让性能结论达到验收口径，关键仍是**官方数据**（`official_comparable`），而不是再薅 1–2% 的开关。"
    ]
}
out_bak_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_backup", "skill_baks")
os.makedirs(out_bak_dir, exist_ok=True)
shutil.copy2(P, os.path.join(out_bak_dir, "prefetch_depth_20260922.json.bak_b4_%s"
                             % time.strftime("%Y%m%d_%H%M%S")))
json.dump(d, io.open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
back = json.load(io.open(P, encoding="utf-8"))
assert "criteria" in back and "batch4_2026_09_22" in back
print("PATCH_OK 已并入 batch4_2026_09_22（判据未动）")
