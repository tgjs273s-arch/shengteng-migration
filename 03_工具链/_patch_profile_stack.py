# -*- coding: utf-8 -*-
r"""_patch_profile_stack.py —— 把 with-stack 那一轮（栈维度归因）并入协议（判据不动）"""
import io
import json
import os
import shutil
import time

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "qwen35-ascend-migrator_整合版", "protocols", "profile_short_20260922.json")
KEY = "stack_round_2026_09_22"
d = json.load(io.open(P, encoding="utf-8"))
if KEY in d:
    print("PATCH_SKIP 已并入")
    raise SystemExit(0)
d[KEY] = {
    "note": "第三轮：`--with-stack`（窗口 50–51，1 个采样步），目的是回答『Computing 里有多少是重计算』"
            "以及把耗时归到调用点。★ **本轮的桶比例不可与前两轮对比**：with_stack 把 Free 抬到 65.1%"
            "（`Computing 33.6 / Free 65.1 / Communication 2.5 / NotOverlapped 1.3`）——"
            "追踪开销本身进了 Free 桶。**只用同轮之内的相对比例**。",
    "run": {"out_dir": "/root/ops/p2prof（漏传 --out，工具默认目录）",
            "prof_dir": "/root/ops/p2prof/prof/4ac9ca3712ba_29018_..._ascend_pt/",
            "rc": "P2PROF_OK；train 100/100",
            "artifacts": {"operator_details.csv": "48.7 MB（含 Call Stack 列）",
                          "trace_view.json": "149.8 MB", "kernel_details.csv": "2.3 MB",
                          "task_time.csv": "1.2 MB"}},
    "stack_attribution_device_self_us": {
        "total": 1198465,
        "by_marker_pct": {"backward": 55.74, "其它（含前向）": 37.07, "communication": 5.19,
                          "optimizer": 1.22, "recompute": 0.78},
        "top_ops": {
            "wait_event（backward 栈）": {"pct": 33.47},
            "wait_event（其它栈）": {"pct": 11.46},
            "aclnnMatmul": {"pct": 11.60},
            "prepare_wy_repr_bwd_kernel": {"pct": 3.27},
            "causal_conv1d_bwd_kernel": {"pct": 2.91},
            "aclnnMul": {"pct": 2.36},
            "HcclAllGather": {"pct": 1.86},
            "wait_event（communication 栈）": {"pct": 1.66},
            "aclnnApplyAdamWV2": {"pct": 0.88},
            "recompute_w_u_fwd_kernel": {"pct": 0.67}
        }
    },
    "findings": [
        "★★ **`wait_event` 类目合计 ≈46.6%**（33.47 + 11.46 + 1.66）——设备侧最大的单一支出不是算术，"
        "而是**等待/同步**。这与另外两个独立量一致：无插桩那一轮的 `Free 19.3%`（主机/流间隙）与 "
        "`7150 次 kernel 发射/步`。⇒ 瓶颈方向是**重叠/同步与发射开销**，而不是通信（5.19%）或算子本身。",
        "★★ **重计算仅 0.78%**（`recompute_w_u_fwd_kernel` 0.67%）⇒ **『按模块重计算』在时间维度上没有收益空间**；"
        "若要动 loss/logits 那条链（峰值显存 5.86 GB），必须把动机如实写成**显存**（chunk loss），"
        "**不得**宣称它省时间。",
        "通信 5.19%（`HcclAllGather` 1.86% 为主）与桶层结论（总通信 5.2–5.6%、未重叠 2.5–2.7%）**互相印证**"
        "⇒ 通信方向继续排除。",
        "前向主导算子仍是大 `MatMul`（11.60%），GDN 反向核合计约 6.2%，与第二轮 kernel 归因方向一致。"
    ],
    "caveats": [
        "**wait_event 占比可能被插桩本身放大**：追踪让主机变慢 ⇒ 流等待变多。"
        "但『等待可观』这一点有独立旁证（无插桩轮次的 Free 19.3% + 7150 发射/步），故方向成立、**数值不可当精确值**。",
        "窗口仅 1–2 个步；数据仍为 mock（每步同一批）⇒ 数据加载/等待成分被低估。",
        "调用栈的**叶子帧解析失败**（我的 `;` + ` in ` 切分与实际格式不符）：实测格式是 "
        "`<file>(<line>): <func>;\\n<file>(<line>): <func>`（分隔符是**字面的反斜杠 n**，不是换行）"
        "⇒ 叶子帧聚合输出全为 `?`，**该维度本轮无结论**（只有按标记的归类有效）。"
    ],
    "revised_decision": {
        "supported_now": [
            "① **等待/同步/发射侧**（wait_event ≈46.6% + Free 19.3% + 7150 发射/步 + 小算子簇 16.3%）："
            "下一步应先查**必要的同步点**（每步回主机的取值/日志、梯度裁剪的通信、以及 cast/copy/transpose 小算子簇），"
            "这是唯一有**三个独立量**指向的方向。",
            "② **显存侧（动机=显存，不是时间）**：logits/loss 物化占峰值 ≈5.86 GB / 8.5 GB ⇒ chunk loss 类改动"
            "可作为**显存**手段立项；且必须先立协议（它改变 loss 计算与显存布局）。"
        ],
        "excluded": ["通信（5.19%，三轮一致）", "重计算（0.78%，时间维度无收益）"],
        "still_not_decided": ["步内峰值显存（需进程内 allocator 统计）；",
                              "叶子帧归因（解析格式已记录，重跑一次即可得）；",
                              "官方 COCO 数据下的数据等待成分（mock 低估）。"]
    },
    "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")
}
out_bak_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_backup", "skill_baks")
os.makedirs(out_bak_dir, exist_ok=True)
shutil.copy2(P, os.path.join(out_bak_dir, "profile_short_20260922.json.bak_stack_%s"
                             % time.strftime("%Y%m%d_%H%M%S")))
json.dump(d, io.open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
back = json.load(io.open(P, encoding="utf-8"))
assert KEY in back and "criteria" in back
print("PATCH_OK 已并入 %s（判据未动）" % KEY)
