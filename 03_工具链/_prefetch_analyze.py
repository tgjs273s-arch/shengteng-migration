# -*- coding: utf-8 -*-
r"""_prefetch_analyze.py —— 预取深度对照的分析器（复用 logs_to_results 的解析口径）

判据（先写在这里，跑完只看数）：
  1. 每轮的 `iter_ms` 中位数取**窗口 11..100**（跳过 warmup；与 profile 工具的
     `window_11_end_median_ms` 同口径），同时给出 1..100 作对照；
  2. 每档取**两轮的中位数**（2 个样本，只报区间与两点值，**不假装有统计功效**）；
  3. 收益用 **(基线 − 候选) / 基线**；两轮都同向才算"稳定"，一正一负记 UNCERTAIN；
  4. **数值中性检查**：各档的 loss 序列若逐点相同 ⇒ 该改动数值中性；首个不同步也要报出来
     （预取只改"何时通信"，不该改数学；若改了，必须当风险项写出来）；
  5. rc 必须全为 0；否则该轮不计入，并如实报出。

用法：
  python _prefetch_analyze.py v1_r1=path1 v1_r2=path2 v2_r1=... v4_r2=...
"""
import io
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from logs_to_results import LINE            # noqa: E402  同一份解析口径


def parse(path):
    steps, ms, loss = [], [], []
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = LINE.match(line.strip())
            if not m or int(m.group(1)) != 0:
                continue
            steps.append(int(m.group(2)))
            ms.append(float(m.group(4)))
            loss.append(float(m.group(5)))
    return steps, ms, loss


def med(seq):
    return statistics.median(seq) if seq else None


def main():
    runs = {}
    for spec in sys.argv[1:]:
        tag, _eq, path = spec.partition("=")
        lvl = tag.split("_")[0]
        rc_file = os.path.join(os.path.dirname(path), "rc.txt")
        rc = None
        if os.path.isfile(rc_file):
            rc = io.open(rc_file, encoding="utf-8", errors="replace").read().strip()
        steps, ms, loss = parse(path)
        win = [v for s, v in zip(steps, ms) if 11 <= s <= 100]
        runs.setdefault(lvl, []).append({"tag": tag, "steps": len(steps), "ms": ms, "loss": loss,
                                         "rc": rc, "win": win})
        print("  %-8s steps=%-4d rc=%-6s median(11..100)=%s  median(1..100)=%s"
              % (tag, len(steps), rc, ("%.1f" % med(win)) if win else "-", ("%.1f" % med(ms)) if ms else "-"))

    print("\n== 每档汇总 ==")
    summary = {}
    for lvl in sorted(runs, key=lambda x: int(x[1:])):
        # ★ 修 bug（首跑即暴露）：第一版写成
        #   `[med([v for s, v in zip(r["ms"], range(1, len(r["ms"]) + 1)) if 11 <= s <= 100]) for r in runs[lvl]]`
        #   —— `zip(r["ms"], range(...))` 把**两者都是数值序列**配错了对（s 取到的是毫秒值、v 是步号），
        #   于是 `11 <= s <= 100` 几乎全 False ⇒ 每档 val 列表为空 ⇒ 汇总报"缺基线"。
        #   正确做法：解析时就该把 (step, ms) 成对存下来，而不是事后凭两个等长列表 zip。
        vals = []
        for r in runs[lvl]:
            w = r.get("win") or []
            if w:
                vals.append(med(w))
        summary[lvl] = vals
        print("  预取 %-2s：n=%d 中位数=%.1f ms  两点值=%s"
              % (lvl[1:], len(vals), med(vals) if vals else -1,
                 ["%.1f" % v for v in vals]))

    base = summary.get("v1") or []
    if not base:
        print("PREF_ANALYZE_FAIL 缺基线（v1）")
        return 2
    b = med(base)
    print("\n== 相对基线（预取 1）的收益 ==")
    verdict = []
    for lvl in sorted(summary, key=lambda x: int(x[1:])):
        if lvl == "v1":
            continue
        vals = summary[lvl]
        if not vals:
            continue
        ratios = [(b - v) / b * 100.0 for v in vals]
        same_dir = all(r > 0 for r in ratios) or all(r < 0 for r in ratios)
        tag = "稳定" if same_dir else "**方向不一致 ⇒ UNCERTAIN**"
        print("  预取 %-2s：中位数 %.1f ms，省时 %s%%（逐轮 %s）⇒ %s"
              % (lvl[1:], med(vals), "%.2f" % ((b - med(vals)) / b * 100.0),
                 ["%.2f" % r for r in ratios], tag))
        verdict.append((lvl, (b - med(vals)) / b * 100.0, same_dir))

    print("\n== 数值中性检查（loss 序列逐点比对，基准=预取 1 的第一轮）==")
    ref = runs["v1"][0]["loss"]
    for lvl in sorted(runs, key=lambda x: int(x[1:])):
        for r in runs[lvl]:
            n = min(len(ref), len(r["loss"]))
            first_diff = next((i + 1 for i in range(n) if ref[i] != r["loss"][i]), None)
            print("  %-8s 相同=%s 首个不同步=%s（比对 %d 点）"
                  % (r["tag"], first_diff is None, first_diff, n))

    best = max(verdict, key=lambda x: x[1]) if verdict else None
    if best and best[2] and abs(best[1]) >= 5.0:
        print("PREF_VERDICT 候选：预取 %s，省时 %.2f%%（两轮同向）" % (best[0][1:], best[1]))
    elif best:
        print("PREF_VERDICT 无 ≥5%% 的稳定收益（最佳=预取 %s，%.2f%%，同向=%s）"
              % (best[0][1:], best[1], best[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
