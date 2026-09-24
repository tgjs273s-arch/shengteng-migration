#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""numeric_diag_v2.py —— 逐步数值分析器 v2（按外部复核裁决重写；v1 原样保留）

v1 → v2 改了什么（逐条对应复核裁决，不是我的发挥）
------------------------------------------------
1. **窗口不变**：不做"截到 k−1"这种"按观测选范围"。v2 永远按**协议声明的完整窗口**裁决；
   `k`（首次超阈位置）只作为**诊断量**输出，且**必须注明是哪一种 k**（打印值不同 / 相对差超阈 /
   原始张量比特不同 —— 本工具只能给前两种，第三种需张量级取证，故显式声明它没做）。
2. **不再输出"判据不可用"**：同臂大量超阈的**正确含义**是"**未满足完整窗口的逐点一致性约束**；
   仅凭现有 A/B 轨迹差异**不能归因于配置改动**" —— 不是"判据失效所以可以缩窗"。v1 的措辞已证错。
3. **契约兼容**：同时接受 v1 协议（顶层 `metrics/window/thresholds`）、阶段 1b 协议
   （`run_protocol.window` + `criteria.thresholds`）与**阶段 1b v2 协议**（`inherited_criteria`）。
   v1 之前直接吃 1b 协议会**崩**，这是缺陷。
4. **坏例加固**（复核指出的四条，逐条落成失败判据）：
   · **长度不等**：不再只比公共前缀 ⇒ 该配对记 `INVALID(length_mismatch)`；
   · **非有限值**（NaN/Inf）：记 `INVALID(non_finite)` —— 不得让它产生"零个超阈点"；
   · **零分母**：记 `undecidable` 并**计数上报**（不静默丢弃）；
   · **步号集合**：要求每次运行带完整 `steps`；缺失或非 1..N 连续 ⇒ `steps_unverified`，
     且**拒绝给出"满足"结论**；声明总步数与实际不符同样拒绝。
5. **归因纪律**：v2 只输出三段结论——`重复性要求`、`首次超阈位置`、`原因/配置效应/官方验收`
   （后三者在本工具能给出的证据下分别为"未定位 / 尚不能确定 / 未闭合"）。

★★ 2026-09-22 第二轮加固（**本轮新增；这一节是外部复核确认的真实缺陷，不是我的发挥**）
------------------------------------------------------------------------------
第一版 v2 有一个**方向性错误**：坏例能被**打印**成 `INVALID`，却**仍被汇总成"满足"**。
根因在聚合函数：它只累加 `status == "OK"` 的配对 ⇒ **被判无效的配对贡献 0 个超阈点、0 个比较点**，
于是"没有任何超阈"被读成"满足"。四条已实测确认的缺陷（每条都对应新增的失败判据）：

| # | 缺陷（实测） | 为什么它是错的 | 现在的处置（出处） |
|---|---|---|---|
| D1 | 某指标出现 NaN：前面打印 `INVALID`，结论仍可能说"重复性要求满足" | 无效判据**不能**被当成"没发现问题"；'无法裁决' 与 '满足' 是两种状态 | 任一配对 `INVALID` ⇒ 总体 `无法裁决`，**绝不**输出"满足"（冻结判据 §fail_closed「非有限值⇒INVALID」+ 复核要求"不能汇总成满足"） |
| D2 | `steps` 声称 8 步，但两条 loss 序列只有 2 步，仍可能说"满足" | 旧代码用 `hi = min(len(series), last)` ⇒ **短序列被静默当成合法覆盖**，"窗口内一致性"实际只比了 2 个点 | 新增 `INVALID(coverage_shortfall)`：指标序列长度必须与运行自报步数一致，且必须覆盖协议窗口 `first..last`（出处：冻结判据「窗口=协议声明的完整窗口」+ 复核要求"每指标长度与完整步号集合一致"） |
| D3 | 零分母被计数，但没有阻止最终"满足" | `undecidable` 只打印不参与判定 ⇒ "无法计算的点"被算作"通过的点" | 存在 `undecidable > 0` ⇒ 总体 `不可判定`（`NUMERIC_V2_INCONCLUSIVE`），**不得**判满足（出处：冻结判据「零分母记 undecidable **并上报**」——"上报"落到整体判定上才有意义） |
| D4 | 分析器不能直接读当前协议 `numeric_diag_phase1b_v2.json` 的 `inherited_criteria` 结构 | 冻结在用的协议工具读不了 ⇒ 要么改用旧协议（口径不符）、要么手工转写（引入人工误差） | `load_protocol()` 新增第三分支识别 `inherited_criteria`（window / per_step_rel_pct） |

另有两条**身份认证**（出处：v1 协议 `run_protocol.must_declare` 的三项声明义务 + 复核要求
"运行状态、配置身份、数据身份缺失时，输出未验证"）：

- 每次运行必须声明**运行状态**、**配置身份**、**数据身份**（任一缺失或为空 ⇒ `UNVERIFIED_IDENTITY`，rc=4）；
- 这三项**只做"存在且非空"校验并把原文打印出来**，不做"看起来像不像"的猜测 ——
  ★ 边界：它证明的是"这条结论是在某身份下产生的"，**不是**"身份声明本身为真"（那要靠日志/哈希取证）。

退出码（**三层状态，绝不混用**）：
  0 = 判定完成（结论可能是"满足"或"未满足"，两者都是**有效裁决**）
  2 = 用法/文件错误        3 = 运行认证未通过（步号）     4 = 身份未认证
  5 = 无法裁决（配对被判无效 / 存在不可判定点）

用法：
  python numeric_diag_v2.py --protocol <协议 json> --results <results.json>
  python numeric_diag_v2.py ... --allow-missing-identity   # 仅用于**事后重析**历史产物：
      # 会打印 NUMERIC_V2_POSTHOC_NO_IDENTITY，明确标注"该结论不含身份认证"
"""
import argparse
import json
import itertools
import math
import os
import sys

# 三层身份：只要有一组全缺/全空 ⇒ 该运行未认证（**不做**"像不像"的猜测）
IDENTITY_GROUPS = (
    ("运行状态", ("train_ok", "rc", "status", "completed", "driver_rc")),
    ("配置身份", ("config_sha256", "config_sha", "config_id", "config_path")),
    ("数据身份", ("data_identity", "data_path", "dataset_id", "official_comparable")),
)


def load_protocol(path):
    """兼容三种协议结构，归一化为 dict（含 source/fields/first/last/thr/steps）。"""
    p = json.load(open(path, encoding="utf-8"))
    if "metrics" in p and "window" in p:                       # v1 结构
        w = p["window"]
        return {"fields": [m["field"] for m in p["metrics"]],
                "first": int(w["first_step"]), "last": int(w["last_step"]),
                "thr": float(p["thresholds"]["per_step_rel_pct"]),
                "steps": None, "source": "v1-schema", "schema": p.get("schema", "")}
    if "criteria" in p and "run_protocol" in p:                 # 阶段 1b 结构
        w = p["run_protocol"]["window"]
        return {"fields": ["loss", "grad_norm"],
                "first": int(w["first_step"]), "last": int(w["last_step"]),
                "thr": float(p["criteria"]["thresholds"]["per_step_rel_pct"]),
                "steps": p["run_protocol"].get("steps"),
                "source": "phase1b-schema", "schema": p.get("schema", "")}
    # ★ 新增（缺陷 D4）：阶段 1b **v2** 协议 —— 判据在 `inherited_criteria` 里
    ic = p.get("inherited_criteria")
    if isinstance(ic, dict) and "window" in ic:
        w = ic["window"]
        return {"fields": ["loss", "grad_norm"],
                "first": int(w["first_step"]), "last": int(w["last_step"]),
                "thr": float(ic["per_step_rel_pct"]),
                "steps": p.get("declared_steps") or p.get("run_protocol", {}).get("steps"),
                "source": "phase1b-v2-inherited_criteria", "schema": p.get("schema", ""),
                "fail_closed": list(ic.get("fail_closed") or [])}
    raise SystemExit("NUMERIC_V2_FAIL 无法识别的协议结构（顶层键=%s）" % sorted(p.keys()))


def _identity(r):
    """返回 (ok, 缺失/为空的组名列表, 已声明项的原文字典)。"""
    missing, seen = [], {}
    for label, keys in IDENTITY_GROUPS:
        hit = None
        for k in keys:
            v = r.get(k)
            if v is not None and not (isinstance(v, str) and not v.strip()):
                hit = (k, v)
                break
        if hit is None:
            missing.append(label)
        else:
            seen[label] = "%s=%s" % hit
    return (not missing), missing, seen


def validate_run(r, required_steps=None):
    """返回 (ok, 原因列表, 身份字典)。步号集合缺失即不可认证 —— 复核指出的第 4 条坏例。"""
    bad = []
    steps = r.get("steps")
    if not steps:
        bad.append("steps_unverified:缺完整步号集合")
    else:
        if steps != list(range(1, len(steps) + 1)):
            bad.append("steps_unverified:步号非 1..N 连续（前 8：%s）" % steps[:8])
        if required_steps and len(steps) != int(required_steps):
            bad.append("steps_unverified:步数 %d ≠ 协议声明 %s" % (len(steps), required_steps))
    if r.get("valid") is False:
        bad.append("run_invalid:运行自报无效")
    id_ok, id_missing, id_seen = _identity(r)
    if not id_ok:
        bad.append("identity_unverified:缺 %s" % "、".join(id_missing))
    return (not bad), bad, {"ok": id_ok, "missing": id_missing, "seen": id_seen}


def pair_compare(ra, rb, field, first, last, thr):
    """逐点比较。返回统计 + 明确的状态（不静默丢弃任何点）。"""
    sa, sb = ra.get(field) or [], rb.get(field) or []
    if len(sa) != len(sb):
        return {"status": "INVALID", "why": "length_mismatch", "len_a": len(sa), "len_b": len(sb)}
    # ★ 缺陷 D2：指标序列长度必须与运行自报步数一致，且必须覆盖协议窗口
    for tag, r, n in ((ra.get("tag"), ra, len(sa)), (rb.get("tag"), rb, len(sb))):
        need = len(r.get("steps") or []) or None
        if need and n != need:
            return {"status": "INVALID", "why": "coverage_shortfall",
                    "detail": "%s 的 %s 长度 %d ≠ 自报步数 %d" % (tag, field, n, need)}
        if n < last:
            return {"status": "INVALID", "why": "coverage_shortfall",
                    "detail": "%s 的 %s 只覆盖到第 %d 步 < 协议窗口末 %d" % (tag, field, n, last)}
    nf = [i + 1 for i, (x, y) in enumerate(zip(sa, sb))
          if not (isinstance(x, (int, float)) and isinstance(y, (int, float))
                  and math.isfinite(x) and math.isfinite(y))]
    if nf:
        return {"status": "INVALID", "why": "non_finite", "at_steps": nf[:8], "count": len(nf)}
    lo, hi = max(0, first - 1), min(len(sa), last)
    over, undec, first_over, maxpct = 0, 0, None, 0.0
    for i in range(lo, hi):
        a, b = sa[i], sb[i]
        if a == 0:
            undec += 1                     # 记可见，不静默丢（D3：必须进入总体判定）
            continue
        d = abs(b - a) / abs(a) * 100.0
        if not math.isfinite(d):
            return {"status": "INVALID", "why": "non_finite_diff", "at_step": i + 1}
        if d > maxpct:
            maxpct = d
        if d > thr:
            over += 1
            if first_over is None:
                first_over = i + 1
    return {"status": "OK", "compared": hi - lo, "window_points": last - first + 1,
            "undecidable": undec, "over": over, "first_over_step": first_over,
            "max_pct": round(maxpct, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--allow-missing-identity", action="store_true",
                    help="仅用于事后重析历史产物：允许缺身份，但会显式打印『不含身份认证』并降级措辞")
    a = ap.parse_args()
    if not os.path.isfile(a.protocol) or not os.path.isfile(a.results):
        print("NUMERIC_V2_FAIL 缺文件")
        return 2

    pr = load_protocol(a.protocol)
    fields, first, last, thr = pr["fields"], pr["first"], pr["last"], pr["thr"]
    print("== 协议（%s）== schema=%s 窗口=%d..%d 阈值=%.1f%% 字段=%s 声明步数=%s"
          % (pr["source"], pr["schema"], first, last, thr, fields, pr["steps"]))

    runs = json.load(open(a.results, encoding="utf-8"))
    if not isinstance(runs, list) or not runs:
        print("NUMERIC_V2_FAIL results.json 不是非空列表")
        return 2

    steps_bad, identity_bad = [], []
    for r in runs:
        ok, why, ident = validate_run(r, pr["steps"])
        r["_ident"] = ident
        if ident["ok"]:
            print("  运行 %-12s 身份：%s" % (r.get("tag"), "；".join(ident["seen"].values())))
        else:
            identity_bad.append(r.get("tag"))
        r["_steps_ok"] = not any(w.startswith(("steps_unverified", "run_invalid")) for w in why)
        r["_steps_bad"] = why
        if why:
            steps_bad.append(r.get("tag"))
            print("  运行 %-12s 认证问题：%s" % (r.get("tag"), why))

    groups = {}
    for r in runs:
        groups.setdefault(r.get("variant"), []).append(r)
    same_pairs = [(v, x, y) for v, rs in sorted(groups.items()) for x, y in itertools.combinations(rs, 2)]
    cross_pairs = ([(x, y) for x, y in itertools.product(groups["A"], groups["B"])]
                   if ("A" in groups and "B" in groups) else [])

    invalid_pairs, undec_total = [], 0

    def scan(title, pairs, labelled):
        print("\n== %s ==" % title)
        if not pairs:
            print("  （无可用配对）")
            return
        for item in pairs:
            if labelled:
                v, x, y = item
                tag = "%s: %s vs %s" % (v, x.get("tag"), y.get("tag"))
            else:
                x, y = item
                tag = "%s vs %s" % (x.get("tag"), y.get("tag"))
            cells = []
            for f in fields:
                st = pair_compare(x, y, f, first, last, thr)
                if st["status"] != "OK":
                    cells.append("%s=%s(%s)%s" % (f, st["status"], st["why"],
                                                  (" " + st["detail"]) if st.get("detail") else ""))
                    invalid_pairs.append((tag, f, st.get("why"), st.get("detail", "")))
                else:
                    cells.append("%s 超阈 %d/%d 零分母 %d 首次超阈步 %s max=%.2f%%"
                                 % (f, st["over"], st["compared"], st["undecidable"],
                                    st["first_over_step"], st["max_pct"]))
            extra = "" if (x.get("_steps_ok") and y.get("_steps_ok")) else "  [步号未认证]"
            print("  %-26s %s%s" % (tag, " | ".join(cells), extra))

    scan("同臂（A/A、B/B）", same_pairs, True)
    scan("跨臂（A/B）", cross_pairs, False)

    # ---- 聚合（**这一版的核心修复**：无效/不可判定必须把结论从"满足"上拉下来）----
    def agg(pairs, labelled):
        tot_over = tot_cmp = tot_undec = tot_invalid = 0
        for item in pairs:
            x, y = (item[1], item[2]) if labelled else (item[0], item[1])
            for f in fields:
                st = pair_compare(x, y, f, first, last, thr)
                if st["status"] == "OK":
                    tot_over += st["over"]
                    tot_cmp += st["compared"]
                    tot_undec += st["undecidable"]
                else:
                    tot_invalid += 1
        return tot_over, tot_cmp, tot_undec, tot_invalid

    s_over, s_cmp, s_und, s_inv = agg(same_pairs, True)
    c_over, c_cmp, c_und, c_inv = agg(cross_pairs, False)
    undec_total = s_und + c_und

    print("\n== 结论（按 v2 归因纪律 + 本轮加固）==")
    # ★★ 自检坏例 A5 当场暴露的**输出缺陷**：先打印"重复性要求…：满足"，再在下面说"身份未认证 ⇒ 结论
    #    未验证" —— 读者第一眼看到的是"满足"。**判据的最终状态必须体现在判定行本身**，
    #    所以未认证时判定行**不得出现"满足"**（降级为"未认证…不得读作满足"）。
    # ★ 「重复性」只由**同臂**配对裁决；跨臂的无效**不得**拖垮重复性结论（否则一个跨臂坏例
    #   会把"同臂完全重复"这件已成立的事说成"无法裁决"——过宽与过松同样是错判）。
    if s_inv:
        verdict = "无法裁决（同臂有 %d 个配对的指标判据无效 ⇒ **不得**读作『满足』）" % s_inv
    elif not same_pairs:
        verdict = "无法裁决（无同臂配对 ⇒ 无从判断重复性）"
    elif s_und:
        verdict = "不可判定（同臂存在 %d 个零分母点 ⇒ **不得**判满足）" % s_und
    elif s_over == 0:
        verdict = "满足"
    else:
        verdict = "未满足"
    steps_ok_all = all(r.get("_steps_ok") for r in runs)
    if not steps_ok_all:
        display = "**未认证（有运行步号未认证）——不得读作『满足』**"
    elif identity_bad and not a.allow_missing_identity:
        display = "**未认证（缺 运行状态/配置身份/数据身份）——不得读作『满足』**"
    elif identity_bad and a.allow_missing_identity and verdict == "满足":
        display = "满足（**事后重析：不含身份认证**）"
    else:
        display = verdict
    print("  重复性要求（完整窗口 %d..%d，阈值 %.1f%%）：%s" % (first, last, thr, display))
    print("      同臂 超阈 %d/%d 比较点、零分母 %d、无效配对 %d；跨臂 超阈 %d/%d、零分母 %d、无效 %d"
          % (s_over, s_cmp, s_und, s_inv, c_over, c_cmp, c_und, c_inv))
    if c_inv or c_und:
        print("      ⚠ 跨臂侧：无效配对 %d、零分母 %d ⇒ **跨臂比较本轮无法裁决**"
              "（不影响上面的重复性结论，但不得据此说跨臂差异）" % (c_inv, c_und))
    print("      ★ 注：比较点不是独立实验数（相邻步相关、同一运行被多次配对）——不得当成样本量")
    print("  首次超阈位置：见上（口径=**打印精度下的相对差**超阈；"
          "原始张量比特级差异**本工具未做**，需张量级取证）")
    print("  原因：未定位（本工具不做归因）")
    print("  配置效应：尚不能由当前证据确定" if s_over else
          "见上（同臂未超阈/不可判定时才可读跨臂）")
    print("  官方验收：未闭合（mock 数据 ⇒ official_comparable=false）")

    if invalid_pairs:
        print("  -- 无效配对明细 --")
        for tag, f, why, det in invalid_pairs[:10]:
            print("     %-26s %-10s %s %s" % (tag, f, why, det))

    # ---- 三层状态：认证 → 身份 → 裁决 ----
    if not all(r.get("_steps_ok") for r in runs):
        print("  证据认证：**有运行步号未认证** ⇒ 上述统计不得作为裁决依据")
        print("NUMERIC_V2_UNVERIFIED_STEPS")
        return 3
    if identity_bad and not a.allow_missing_identity:
        print("  身份认证：**下列运行缺 运行状态/配置身份/数据身份 之一** ⇒ 结论未验证：%s" % identity_bad)
        print("NUMERIC_V2_UNVERIFIED_IDENTITY")
        return 4
    if identity_bad and a.allow_missing_identity:
        print("  身份认证：%s 缺身份字段 —— 本次为**事后重析**："
              "该结论**不含身份认证**（不得当作『该身份下复现』的证据）" % identity_bad)
        print("NUMERIC_V2_POSTHOC_NO_IDENTITY")
    if s_inv or s_und or not same_pairs:
        # 退出码 5 = **重复性裁决无法作出**（注意：跨臂无效不在这里 —— 它单独提示，见上）
        print("NUMERIC_V2_INCONCLUSIVE verdict=%s" % verdict)
        return 5
    print("NUMERIC_V2_DONE verdict=%s" % verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
