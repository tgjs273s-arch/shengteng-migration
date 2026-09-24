# -*- coding: utf-8 -*-
r"""_record_reanalysis.py —— 把 2026-09-22 第二轮重析**留档**到 v2 协议（只增不改判据）

纪律：v2 协议的**判据**已冻结，本次**不改判据**，只追加一节结果（与既有 `results_2026_09_22` 同性质）。
写盘前备份、写盘后回读校验；已有同名键就报错退出（不覆盖）。
"""
import io
import json
import os
import shutil
import sys
import time

P = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "protocols", "numeric_diag_phase1b_v2.json")
KEY = "reanalysis_2026_09_22_r2"

d = json.load(io.open(P, encoding="utf-8"))
if KEY in d:
    print("RECORD_FAIL 已存在 %s（不覆盖）" % KEY)
    sys.exit(1)

d[KEY] = {
    "note": "★ 本轮**只修工具、不改判据**：判据仍以 inherited_criteria 为准（一字未动）。"
            "本节记录'修好分析器后，用归档日志重跑历史产物'的结果，用于证明**历史裁决没有被修复翻转**。",
    "why": [
        "外部复核实测指出：第一版 v2 分析器会把坏例**打印**成 INVALID，却**汇总**成『满足』"
        "（NaN / 自报 8 步但序列仅 2 步 / 零分母 三种情形都中招）。",
        "⇒ 先修工具，再用**归档产物**重跑，验证 P0（未满足）与 P2（满足）两条历史结论**不变**。",
    ],
    "commands": [
        "python logs_to_results.py --out _reanalysis/results_p0.json --data-identity "
        "mock_1img_cutoff1024_dp2_nonofficial 00_A#A=…/_00_A__train.log 03_A#A=…/_03_A__train.log "
        "04_A#A=…/_04_A__train.log",
        "python logs_to_results.py --out _reanalysis/results_p2.json --data-identity "
        "mock_1img_cutoff1024_dp2_nonofficial P2r1#P2=…/run1__train.log P2r2#P2=…/run2__train.log",
        "python numeric_diag_v2.py --protocol protocols/numeric_diag_phase1b_v2.json "
        "--results _reanalysis/results_p0.json",
        "python numeric_diag_v2.py --protocol protocols/numeric_diag_phase1b_v2.json "
        "--results _reanalysis/results_p2.json",
    ],
    "source": "远端证据_20260922/logs/（归档 train.log，本轮**未连远端**）",
    "p0_verdict": {
        "verdict": "未满足", "rc": 0, "window": "1..100", "thr_pct": 2.0,
        "pairs": {
            "00_A vs 03_A": {"loss_over": "90/100", "grad_norm_over": "88/100", "first_over_step": 4},
            "00_A vs 04_A": {"loss_over": "93/100", "grad_norm_over": "89/100", "first_over_step": 3},
            "03_A vs 04_A": {"loss_over": "86/100", "grad_norm_over": "92/100", "first_over_step": 4,
                             "note": "★ 本对是**本轮新增**（旧记录只登记了两对）；同臂 538/600 比较点超阈"},
        },
        "matches_archived_record": "是（90/100 与 93/100、首次超阈步 4 与 3 逐项一致）",
    },
    "p2_verdict": {
        "verdict": "满足", "rc": 0, "pairs": {"P2r1 vs P2r2": {"loss_over": "0/100",
                                                             "grad_norm_over": "0/100", "max_pct": 0.0}},
        "matches_archived_record": "是（0/100、0/100、max 0.00%）",
    },
    "identity_now_derived": {
        "运行状态": "status=completed(100/100)（由日志末步与声明总步数导出，另扫致命标记）",
        "配置身份": "config_sha256=dump_<16>（对日志的**有效配置 dump** 归一化后取摘要）",
        "数据身份": "data_identity=**必须显式声明**（由 --data-identity 传入，不由日志推断）",
        "边界": "配置身份=『该日志打印出来的有效配置』的摘要，**不是**『实际生效配置』的证明；"
                "dump 区段按首个 iteration 行截取（启发式），已记录行数（230）供核对。",
    },
    "new_capability_single_variable_evidence": {
        "what": "配置身份现在能**直接从产物**给出『单变量』证据：",
        "A_arm_three_runs": "dump_abe2f33e5286dd21（00_A / 03_A / 04_A **同一身份**）",
        "P2_two_runs": "dump_364d2c6bfb7349ea（P2r1 / P2r2 同一身份）",
        "A_vs_P2_diff_keys": ["use_deter_comp（A=False，P2=True）"],
        "key_point": "差异**只剩被声明的那个键** ⇒ 与协议声明的 delta 一致（工具给出的、不是申明的）。",
        "false_distinction_fixed": "★ 归一化前三个 A 臂给出**三个不同摘要**（差异键只有 `save` 输出路径）"
                                  "——若不修，'同配置三次运行'会被判成三个不同配置（假区分 = 另一种错判）。"
                                  "现按白名单显式排除纯输出路径键（`save`），并在产物里登记 config_excluded_keys。",
    },
    "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "recorded_by": "_record_reanalysis.py（写盘前备份、写盘后回读校验）",
}

bak = P + ".bak_%s" % time.strftime("%Y%m%d_%H%M%S")
shutil.copy2(P, bak)
json.dump(d, io.open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

back = json.load(io.open(P, encoding="utf-8"))
assert KEY in back and back["inherited_criteria"] == d["inherited_criteria"], "回读校验失败"
# ★ 这里的键路径第一版写错了层级（写成 back["p2_verdict"]，实际在 back[KEY]["p2_verdict"]）——
#   于是**断言自己崩了**，而文件其实已正确写入。教训：回读校验的路径也要跟写入结构对齐（坑同族：
#   "验证的不是刚写的那个对象"）。现在按真实层级断言。
assert back[KEY]["p2_verdict"]["verdict"] == "满足", "回读校验失败：P2 判定"
assert back[KEY]["p0_verdict"]["verdict"] == "未满足", "回读校验失败：P0 判定"
print("RECORD_OK 已追加 %s（判据 inherited_criteria 未改动）" % KEY)
print("  备份: %s" % bak)
print("  文件大小: %d B" % os.path.getsize(P))
