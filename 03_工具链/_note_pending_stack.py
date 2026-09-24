# -*- coding: utf-8 -*-
r"""_note_pending_stack.py —— 把"已挂到后台、结果待读"的 with-stack 运行登记进协议

为什么必须写下来：我刚刚**漏传 `--out`**，这一轮用的是工具默认目录 `/root/ops/p2prof`。
"某个后台运行的结果在哪个目录"如果只存在于对话里，下一轮就会像工具链那样丢掉（坑 196 的教训）。
"""
import io
import json
import os
import shutil
import time

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "qwen35-ascend-migrator_整合版", "protocols", "profile_short_20260922.json")
KEY = "pending_with_stack_2026_09_22"
d = json.load(io.open(P, encoding="utf-8"))
if KEY in d:
    print("NOTE_SKIP 已登记")
    raise SystemExit(0)
d[KEY] = {
    "status": "已挂后台运行，**结果待读**（本轮对话结束时仍在跑）",
    "purpose": "回答 profile 协议里最后一项 still_not_decided：Computing（70.9–78.0%）里**重计算占比**是多少。"
               "开 `--with-stack` 后可以把 kernel 归因到 Python 调用点，从而区分『真实计算』与『重计算』。",
    "command": "python3 scripts/56_profile_run.py --skill /root/qwen35-ascend-migrator "
               "--config /root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml --steps 100 "
               "--prof-start 50 --prof-end 51 --with-stack",
    "out_dir": "/root/ops/p2prof（★ 注意：本轮**漏传 --out**，用的是工具默认目录）",
    "launch_log": "/root/ops/profstack_20260922_113307/launch.log",
    "how_to_read": [
        "① `cat /root/ops/p2prof/result.json` 看桶比例（与 50–51 窗口对照）；",
        "② 在 `/root/ops/p2prof/prof/*/ASCEND_PROFILER_OUTPUT/` 下找带调用栈的产物"
        "（kernel_details 的调用栈列 / `trace_view.json`），把 Computing 按调用点归因；",
        "③ 判定：若重计算相关调用点占 Computing 的显著比例 ⇒ 支持『按模块重计算 / chunk loss』方向；"
        "若不显著 ⇒ 收回该方向，改看小算子簇（16.3%、7150 发射/步）。",
        "④ 纪律：仍然**只用比例**；被 profiling 拖慢的绝对时间不对标稳态。"
    ],
    "boundaries": ["窗口只有 2 个步（50–51），样本小；", "with_stack 会显著增大产物与运行时间；",
                   "数据仍为 mock（每步同一批）⇒ 数据等待成分被低估。"],
    "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")
}
out_bak_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_backup", "skill_baks")
os.makedirs(out_bak_dir, exist_ok=True)
shutil.copy2(P, os.path.join(out_bak_dir, "profile_short_20260922.json.bak_pending_%s"
                             % time.strftime("%Y%m%d_%H%M%S")))
json.dump(d, io.open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
back = json.load(io.open(P, encoding="utf-8"))
assert KEY in back and "criteria" in back
print("NOTE_OK 已登记 %s（判据未动）" % KEY)
