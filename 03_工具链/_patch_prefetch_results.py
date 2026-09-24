# -*- coding: utf-8 -*-
r"""_patch_prefetch_results.py —— 把预取实验的结果并入已冻结的协议（判据不动）"""
import io
import json
import os
import shutil
import time

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "qwen35-ascend-migrator_整合版", "protocols", "prefetch_depth_20260922.json")
d = json.load(io.open(P, encoding="utf-8"))
if "results_2026_09_22" in d:
    print("PATCH_SKIP 已并入")
    raise SystemExit(0)
d["results_2026_09_22"] = {
    "runs": {"base": "/root/ops/pref_20260922_120450", "all_rc": "6/6 rc=0",
             "logs_local": "远端证据_20260922/prefetch/（12 文件，全部双侧 SHA256 校验）"},
    "per_run_median_ms_11_100": {
        "v1_r1": 734.0, "v1_r2": 729.2,
        "v2_r1": 704.3, "v2_r2": 723.8,
        "v4_r1": 698.2, "v4_r2": 707.8
    },
    "per_level_median_ms": {"1": 731.6, "2": 714.0, "4": 703.0},
    "saving_pct_vs_baseline": {
        "2": {"median": 2.40, "per_round": [3.73, 1.07], "same_direction": True},
        "4": {"median": 3.91, "per_round": [4.56, 3.26], "same_direction": True}
    },
    "verdict": "**低于获胜线**：最佳档（预取 4）稳定省 **3.91%** < 5% ⇒ 按 criteria.win_rule 记为"
               "『同向但无可报告收益』。**不写入 Skill 推荐**（除非后续与新批次独立确认后仍达标）。",
    "numerical_neutrality": {
        "result": "**本检查无区分力**：连基线自己的两轮（v1_r1 vs v1_r2）都自**第 3 步**起 loss 不同 ⇒ "
                  "各档与基线的差异属于**已知的同臂非确定性**（与 P0 结论一致：超阈 90–93/100、首次超阈步 3–4）。",
        "implication": "所以本轮**既不能**据此说『预取改了数学』，**也不能**据此说『预取数值中性』——"
                       "要判定中性必须换尺子（`use_deter_comp=true` 的确定性路径，或逐层梯度比对 / 一步回放的 C1 比较）。",
        "note": "这是本项目第四次遇到『判据缺乏区分力』（前三次：loss 打印精度、mock 每步同批、lr=0 空更新）。"
    },
    "measurement_integrity_caveat": {
        "warning": "★ **今天的绝对步时长不可与昨天的 419.03 / 521.23 / 432.7 ms 对比**："
                   "本轮运行在**换了 IP 的机器**上（`199.98.58.213`，host `4ac9ca3712ba`，npu-smi 26.1.1），"
                   "而昨天那批在 `199.103.55.150`（CANN 9.1.0-beta.3）上。",
        "consequence": "只有**同一会话、同一台机器**内的比较有效（本例：3 档 × 2 轮同一批完成）；"
                       "跨机器的绝对时间、以及『比昨天快多少』这类说法**一律作废**。",
        "recorded_evidence": "今天的基线同配置中位步时长 ≈731.6 ms（预取 1）；昨天同配置 A 臂为 419.03 ms —— "
                             "两者相差近 1.75×，只能解释为**机器/驱动差异**，不是配置差异。"
    },
    "next_with_remaining_budget": [
        "① 独立确认：再跑 2 轮预取 4 与 2 轮基线（新批次），若仍 ≥5% 才可考虑写入推荐；否则维持『无可报告收益』；",
        "② 换更大的杠杆：`training.log_interval` 1→10（去掉每步张量格式化的设备→主机同步；"
        "这一步正是 with-stack 抓到的 `_tensor.py: __format__ → training_log`）。"
        "**必须声明代价**：per-step loss 序列变稀 ⇒ 数值协议的可观测点减少；",
        "③ 另一条（需改代码，先立协议）：`clip_grad=0.0` 时仍每步调用 `clip_grad_norm`（含梯度范数归约）；"
        "以及 `average_losses_across_data_parallel_group` 每步 all_reduce —— 都是『等待/同步』的候选来源；",
        "④ 全部只做 profile 支持的项；任何组合获胜后按 Codex 第六步做 AB/BA 独立确认。"
    ],
    "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")
}
out_bak_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_backup", "skill_baks")
os.makedirs(out_bak_dir, exist_ok=True)
shutil.copy2(P, os.path.join(out_bak_dir, "prefetch_depth_20260922.json.bak_res_%s"
                             % time.strftime("%Y%m%d_%H%M%S")))
json.dump(d, io.open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
back = json.load(io.open(P, encoding="utf-8"))
assert "criteria" in back and back["criteria"]["win_rule"]
print("PATCH_OK 已并入 results_2026_09_22（判据未动）")
