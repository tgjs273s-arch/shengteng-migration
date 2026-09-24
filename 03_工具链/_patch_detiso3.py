# -*- coding: utf-8 -*-
r"""_patch_detiso3.py —— 把 v3/v4 结果并入确定性分离协议（判据不动）"""
import io
import json
import os
import shutil
import time

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "qwen35-ascend-migrator_整合版", "protocols", "determinism_isolation_20260922.json")
d = json.load(io.open(P, encoding="utf-8"))
if "results_detiso3_2026_09_22" in d:
    print("PATCH_SKIP 已并入")
    raise SystemExit(0)
d["results_detiso3_2026_09_22"] = {
    "why": "上一轮（v1/v2 + 对照）证明单独 export 两个 NPU 环境变量**都不足以**恢复重复性；"
           "剩两个可能：第三个机制（torch 分派层开关），或三者组合。本轮用 `sitecustomize.py` "
           "在解释器启动时注入 torch 层开关，并加『三者全设』正对照验证注入手段本身有效。",
    "base": "/root/ops/detiso3_20260922_125603",
    "injection_verification": {
        "marker": "sitecustomize 在每个进程启动时打印 `DETSC active torch.use_deterministic_algorithms(True)`",
        "counts": {"v3_r1": 3, "v3_r2": 3, "v4_r1": 3, "v4_r2": 3, "DETSC FAILED": 0},
        "meaning": "3 = torchrun 主进程 + 2 个 rank worker 各一行 ⇒ 注入**确实生效**（不是静默失败）"
    },
    "results": {
        "v3_torch_flag_only": {"over": 0, "compared": 200, "verdict": "满足"},
        "v4_all_three_positive_control": {"over": 0, "compared": 200, "verdict": "满足"}
    },
    "conclusion": "★★ **关键机制是 `torch.use_deterministic_algorithms(True)`**：只设它（不设任何 NPU 环境变量）"
                  "就能把同臂超阈从 186–187/200 降到 **0/200**；而单独 export `HCCL_DETERMINISTIC=True` 或 "
                  "`CLOSE_MATMUL_K_SHIFT=1` 都**与对照无差别**。⇒ 此前『改善来自 HCCL 确定性 / matmul K-shift』"
                  "的推论**被证伪**（至少对『启动前 export』这一种设置方式而言）。",
    "positive_control_validity": "v4（三者全设）同样 0/200 ⇒ 说明注入手段有效，因此 v3 的『有效』**不是**注入失败造成的假象"
                                 "（没有这条正对照就不能下结论 —— 这是上一轮坑的教训）。",
    "mechanistic_implication": [
        "PyTorch 的确定性模式会改变**算子分派**（选择确定性实现）⇒ 这从机制上解释了为什么 "
        "`use_deter_comp=true` 时**第 1 步 loss 就不同**（0.1154926 → 0.1169518）：前向本身被换成了另一组实现。",
        "与一步回放的结论相容：不加该开关时 **C1（前向）逐字节相同、C3（通信后梯度）不同** ⇒ "
        "非确定性出在**反向**；而打开该开关后前向也换了实现（所以 loss 变化）但仍然可重复。",
        "⇒ 对交付的实际含义不变：该开关**只能作诊断**（它改变数值），但它现在是**唯一被隔离验证过的**『能让同臂重复』的机制。"
    ],
    "boundary": [
        "本结论只覆盖『启动前 export 无效 / 解释器启动时注入 torch 开关有效』这两种设置方式；"
        "框架内部（`set_deterministic_algorithms()` 里）三者是同一次调用里设置的，无法据此推断"
        "『NPU 环境变量在框架内部也无用』。",
        "仍未定位到**具体算子/通信**（C2 通信前局部梯度在 FSDP2 下不可观测）；本轮只回答『哪个开关是机制』。",
        "同机 mock 数据、100 步窗口；不涉及长程与官方数据。"
    ],
    "status": "criteria_frozen_and_executed（机制已隔离：torch 层开关为关键；两个 env 单独无效）"
}
out_bak_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_backup", "skill_baks")
os.makedirs(out_bak_dir, exist_ok=True)
shutil.copy2(P, os.path.join(out_bak_dir, "determinism_isolation.json.bak_%s"
                             % time.strftime("%Y%m%d_%H%M%S")))
json.dump(d, io.open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
back = json.load(io.open(P, encoding="utf-8"))
assert "criteria" in back and "results_detiso3_2026_09_22" in back
print("PATCH_OK 已并入 results_detiso3_2026_09_22（判据未动）")
