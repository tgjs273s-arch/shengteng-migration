# -*- coding: utf-8 -*-
r"""_patch_profile_memory.py —— 把"带显存采集"的第二轮 profile 并入协议（判据不动）"""
import io
import json
import os
import shutil
import time

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "qwen35-ascend-migrator_整合版", "protocols", "profile_short_20260922.json")
KEY = "memory_round_2026_09_22"
d = json.load(io.open(P, encoding="utf-8"))
if KEY in d:
    print("PATCH_SKIP 已并入")
    raise SystemExit(0)
d[KEY] = {
    "note": "第二轮 profile：开启显存采集（为此给 56_profile_run.py 新增 `--with-memory` —— 原实现把 "
            "`with_memory` **硬编码为 False**，导致『激活/logits 占用是否突出』这类问题在本项目里一直无法回答）。"
            "窗口缩到 50–52（显存记录很重）。同时挂了一个独立的 HBM 采样循环（`npu-smi info`）。",
    "run": {
        "base": "/root/ops/profmem3_20260922_112940",
        "prof_out": "/root/ops/profmem3_20260922_112940/run",
        "config": "/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml（GBS=8）",
        "profiling_window": [50, 52],
        "with_memory": True, "with_stack": False,
        "rc": "P2PROF_OK；train 100/100"
    },
    "buckets_with_memory": {
        "shares_pct": {"Computing": 70.9, "Free": 26.6, "Communication": 5.2,
                       "Communication(Not Overlapped)": 2.5},
        "mean_us": {"Computing": 577850, "Free": 217047, "Communication": 42107,
                    "Communication(Not Overlapped)": 20311},
        "note": "与第一轮（Computing 78.0 / Free 19.3）方向一致；显存采集本身把 Free 抬高（开销进了 Free 桶）"
                "⇒ 两轮的 Free 不可直接比较，比例只用于定性。"
    },
    "hbm_curve": {
        "source": "npu-smi info 采样（每 5 秒尝试一次，但**单次调用本身要十几秒** ⇒ 实际约 7 点/124 秒，粗粒度）",
        "chip0": {"first_mb": 3124, "last_mb": 8458, "min_mb": 3124, "max_mb": 8531, "delta_mb": 5334},
        "chip1": {"first_mb": 2896, "last_mb": 8220, "min_mb": 2896, "max_mb": 8299, "delta_mb": 5324},
        "reading": "两块 die 从**空闲基线 ~3.0 GB** 升到 **~8.5 GB 峰值**（连续 100 步内）；"
                   "末端略降（收尾阶段）⇒ 本配置下**未见 64 GB 量级风险**，但**跨步增长曲线是粗粒度**的，"
                   "不能据此宣布『显存稳定』。",
        "boundary": "npu-smi 的采样间隔 >> 步时长（~0.43–0.7 s/步）⇒ **看不到步内峰值**；"
                    "要拿步内峰值必须用进程内 allocator 统计（本轮未做）。"
    },
    "operator_memory": {
        "file": "…/ASCEND_PROFILER_OUTPUT/operator_memory.csv（17458 行）",
        "columns": ["Name", "Size(KB)", "Allocation Time(us)", "Release Time(us)",
                    "Active Release Time(us)", "Duration(us)", "Active Duration(us)",
                    "Allocation Total Allocated(MB)"],
        "top_peak_size_kb": {
            "aten::empty_strided": 6145920.5,
            "aten::_log_softmax": 6145920.5,
            "aten::nll_loss_backward": 6145920.5,
            "aten::_log_softmax_backward_data": 6145920.5,
            "aten::_foreach_copy_": 3119104.0,
            "aten::_convolution": 3119104.0,
            "aten::mul": 3119104.0,
            "aten::copy_": 3119104.0,
            "aten::cat": 3119104.0,
            "aten::stack": 3119104.0
        },
        "reading": "★ 峰值最高的四项**是同一峰值**（6145920.5 KB ≈ **5.86 GB**）：`empty_strided` 是通用分配入口，"
                   "而 `_log_softmax` / `nll_loss_backward` / `_log_softmax_backward_data` 是**同一个峰值时刻的持有者**"
                   "⇒ **logits/loss 物化是最大显存持有者**（8.5 GB 峰值里约 5.9 GB 落在这一条链上）。",
        "aggregation_boundary": "本表的『Top』是按**算子名取最大 Size**（`max` 聚合），因此"
                                "① `empty_strided` 这类**通用分配入口**会与语义算子并列，不能当成『某个算子特别费显存』；"
                                "② 同名多次分配只保留最大值 ⇒ **不能**用本表求『总占用』。要更细的归因需按 `Active Duration`×Size 或时间轴重建。"
    },
    "revised_decision": {
        "now_supported": [
            "**显存方向（证据最具体）**：logits/loss 物化占峰值约 5.9 GB / 8.5 GB ⇒ 按 Codex 的判定表对应"
            "『激活或 logits 占用突出 ⇒ 单独测试 **chunk loss**、按模块重计算』——**这是目前唯一同时有"
            "显存证据与时间证据支持的方向**（时间侧：`_log_softmax` 2.08% + `LogSoftmaxGrad` 2.53%）。",
            "**算子族方向**：GDN/linear-attn 族占时 28.2%（第一轮 kernel 归因）⇒ 仍待回答『是否落在关键路径』。",
            "**小算子/主机方向**：小算子簇 16.3%、7150 发射/步、Free 19–27% ⇒ 三条独立线索互相印证。",
            "**通信方向：排除**（未重叠 2.5–2.7%，两轮一致）。"
        ],
        "next_before_any_change": [
            "① chunk loss 类改动必须先立协议（它改变 loss 计算与显存布局，属于『改计算』而非『改开关』）；",
            "② 若要做 GDN 方向，必须先采**进程内步内峰值显存**（决定『恢复更快路径』的代价），"
            "本轮 npu-smi 的粗粒度曲线**不足以**支撑该决定。"
        ],
        "still_not_decided": [
            "重计算（recompute）占比仍未量化：`with_stack` 尚未开启（决定『Computing 里有多少是重算』）；",
            "步内显存峰值未知（采样粒度限制）；",
            "mock 数据每步同一批 ⇒ 数据加载成分被低估（沿用第一轮边界）。"
        ]
    },
    "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")
}
out_bak_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_backup", "skill_baks")
os.makedirs(out_bak_dir, exist_ok=True)
shutil.copy2(P, os.path.join(out_bak_dir, "profile_short_20260922.json.bak_mem_%s"
                             % time.strftime("%Y%m%d_%H%M%S")))
json.dump(d, io.open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
back = json.load(io.open(P, encoding="utf-8"))
assert KEY in back and "criteria" in back
print("PATCH_OK 已并入 %s（判据未动）" % KEY)
