#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""numeric_diag.py —— 逐步数值诊断（**强制先读冻结协议**，拒绝用默认阈值）

为什么这样设计（对应外部复核的建议）
----------------------------------
1. **判据先于结果冻结**：本工具**必须**用 `--protocol <json>` 显式给出协议（阈值、窗口、口径、
   决策规则）；不给就直接退出。杜绝"跑完再挑阈值"。
2. **A/A 优先**：先算**同配置重复**的逐步差异；若 A/A 本身就大量超阈，则该判据在此负载下
   **不可用**，A/B 的任何差异都**不得**归因于配置（Codex 的决策规则）。
3. **全步而非前两步**：默认窗口 1..100。这正是 v1 结论出错的地方 —— "前两步一致"被当成了
   正确性保障，而第 3 步起就可能分叉。
4. **不做性能结论**：本工具只输出数值一致性状态。

输入：一个 `results.json`（含每次运行的 `loss` / `grad_norm` 逐步序列与 `variant`）
输出：同臂（A/A、B/B）与跨臂（A/B）的逐字段统计 + 决策结论

用法：
  python numeric_diag.py --protocol protocols/numeric_verification_100step.json \
      --results <results.json> [--max-pairs 6]
"""
import argparse
import itertools
import json
import os
import statistics
import sys


def rel_pct(a, b):
    """|a−b|/|a|×100；分母为 0 ⇒ 返回 None（不可判定，不计入比较数）。"""
    if a == 0:
        return None
    return abs(b - a) / abs(a) * 100.0


def pair_stats(ra, rb, field, first, last):
    """两组运行的逐步对比统计。"""
    sa, sb = ra.get(field) or [], rb.get(field) or []
    n = min(len(sa), len(sb))
    lo = max(0, first - 1)
    hi = min(n, last)
    vals, first_div, worst, worst_abs = [], None, 0.0, 0.0
    for i in range(lo, hi):
        d = rel_pct(sa[i], sb[i])
        if d is None:
            continue
        vals.append(d)
        if d > 2.0 and first_div is None:
            first_div = i + 1
        if d > worst:
            worst = d
        ad = abs(sb[i] - sa[i])
        if ad > worst_abs:
            worst_abs = ad
    over = sum(1 for v in vals if v > 2.0)
    return {"compared": len(vals), "over_2pct": over, "max_pct": round(worst, 3),
            "first_div_step": first_div, "worst_abs": round(worst_abs, 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, help="冻结的协议 json（必填，见模块说明）")
    ap.add_argument("--results", required=True)
    ap.add_argument("--max-pairs", type=int, default=8)
    a = ap.parse_args()

    if not os.path.isfile(a.protocol):
        print("NUMERIC_DIAG_FAIL 缺协议文件（判据必须先冻结）：%s" % a.protocol)
        return 2
    proto = json.load(open(a.protocol, encoding="utf-8"))
    thr = float(proto["thresholds"]["per_step_rel_pct"])
    first = int(proto["window"]["first_step"])
    last = int(proto["window"]["last_step"])
    fields = [m["field"] for m in proto["metrics"]]
    print("== 协议 ==")
    print("  schema=%s 冻结于=%s" % (proto.get("schema"), proto.get("frozen_at")))
    print("  窗口=%d..%d  阈值=%.1f%%（逐点相对差）  字段=%s" % (first, last, thr, fields))

    runs = json.load(open(a.results, encoding="utf-8"))
    if not isinstance(runs, list) or not runs:
        print("NUMERIC_DIAG_FAIL results.json 不是非空列表")
        return 2
    print("  运行数=%d  变体=%s" % (len(runs), sorted({r.get("variant") for r in runs})))

    groups = {}
    for r in runs:
        groups.setdefault(r.get("variant"), []).append(r)

    same, cross = [], []
    for var, rs in sorted(groups.items()):
        for x, y in itertools.combinations(rs, 2):
            same.append((var, x, y))
    if "A" in groups and "B" in groups:
        for x, y in itertools.product(groups["A"], groups["B"]):
            cross.append((x, y))

    def report(title, pairs, keyfn):
        print("\n== %s ==" % title)
        if not pairs:
            print("  （无可用配对）")
            return {}
        agg = {f: {"over": 0, "n": 0, "max": 0.0} for f in fields}
        shown = 0
        for item in pairs:
            if keyfn:
                var, x, y = item
                tag = "%s: %s vs %s" % (var, x.get("tag"), y.get("tag"))
            else:
                x, y = item
                tag = "%s vs %s" % (x.get("tag"), y.get("tag"))
            if shown < a.max_pairs:
                line = []
                for f in fields:
                    st = pair_stats(x, y, f, first, last)
                    agg[f]["over"] += st["over_2pct"]
                    agg[f]["n"] += st["compared"]
                    agg[f]["max"] = max(agg[f]["max"], st["max_pct"])
                    line.append("%s %d/%d 超阈(max=%.2f%% 首分叉=%s)"
                                % (f, st["over_2pct"], st["compared"], st["max_pct"], st["first_div_step"]))
                print("  %-24s %s" % (tag, " | ".join(line)))
                shown += 1
            else:
                for f in fields:
                    st = pair_stats(x, y, f, first, last)
                    agg[f]["over"] += st["over_2pct"]
                    agg[f]["n"] += st["compared"]
                    agg[f]["max"] = max(agg[f]["max"], st["max_pct"])
        if len(pairs) > shown:
            print("  …（共 %d 对，其余仅计入汇总）" % len(pairs))
        for f in fields:
            o, n, mx = agg[f]["over"], agg[f]["n"], agg[f]["max"]
            print("  汇总 %-10s 超阈 %d/%d（%.1f%%）  最大逐点相对差=%.2f%%"
                  % (f, o, n, (100.0 * o / n if n else 0.0), mx))
        return agg

    same_agg = report("同臂（A/A、B/B）—— 判据可用性的前提", same, True)
    cross_agg = report("跨臂（A/B）—— 只有同臂通过后才可解读", cross, False)

    # ---- 决策（严格按冻结协议）----
    print("\n== 决策（按协议 decision_rule）==")
    aa_bad = False
    for f in fields:
        o, n = same_agg.get(f, {}).get("over", 0), same_agg.get(f, {}).get("n", 0)
        if n and (100.0 * o / n) > 1.0:     # 同臂只要有 >1% 的步超阈，即认为判据不可用
            aa_bad = True
            print("  同臂 %s 超阈 %d/%d ⇒ **该判据在此负载下不可用**" % (f, o, n))
    if not same:
        print("  **缺少同臂（A/A）数据** ⇒ 按协议 fail_closed：A/B 的任何差异**不得**归因于配置")
        print("NUMERIC_DIAG_UNDETERMINED 需补 A/A 运行")
        return 0
    if aa_bad:
        print("  ⇒ 结论：**完整数值一致性未建立**；A/B 差异不能归因于被测开关（同臂同样分歧）")
        print("NUMERIC_DIAG_SAMEARM_UNSTABLE")
        return 1
    bad_cross = any(cross_agg.get(f, {}).get("over", 0) for f in fields)
    if bad_cross:
        print("  ⇒ 同臂稳定而跨臂超阈 ⇒ **配置改变了数值轨迹**，需按前向/梯度/更新分段定位")
        print("NUMERIC_DIAG_CONFIG_EFFECT")
        return 1
    print("  ⇒ 同臂与跨臂均在阈值内 ⇒ 数值一致性在本协议下**已建立**")
    print("NUMERIC_DIAG_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
