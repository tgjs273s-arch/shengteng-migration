# -*- coding: utf-8 -*-
r"""_batch4_analyze.py —— 四臂对照分析（基线 a / 预取4 b / log10 c / 组合 d）

判据（沿用 `protocols/prefetch_depth_20260922.json` 的 criteria，不新造）：
  · 每轮取 iter_ms 中位数（窗口 11..100）；每档取两轮中位数并报原始两点值；
  · 收益 =(基线−候选)/基线；两轮同向且 ≥5% ⇒ 候选收益；方向不一致 ⇒ UNCERTAIN；同向 <5% ⇒ 无可报告收益；
  · **口径差异必须声明**：`log_interval>1` 时 `iter_ms` 是该区间内的平均（语义与逐点不同），
    且 loss 可观测点变稀 ⇒ 只用于性能对照，**不得**当成数值协议的等价产物。

用法：python _batch4_analyze.py a_r1=path a_r2=path b_r1=... d_r2=...
"""
import io
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from logs_to_results import LINE            # noqa: E402

LABEL = {"a": "基线(预取1,log1)", "b": "预取4(log1)", "c": "log_interval=10(预取1)",
         "d": "预取4+log10"}


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
        arm = tag.split("_")[0]
        if not os.path.isfile(path):
            print("  跳过 %s（本地无文件 %s）" % (tag, path))
            continue
        steps, ms, loss = parse(path)
        win = [v for s, v in zip(steps, ms) if 11 <= s <= 100]
        runs.setdefault(arm, []).append({"tag": tag, "steps": len(steps), "win": win,
                                         "ms": ms, "loss": loss, "points": len(win)})
        print("  %-8s steps=%-4d 点数(11..100)=%-3d median(11..100)=%s"
              % (tag, len(steps), len(win), ("%.1f" % med(win)) if win else "-"))

    print("\n== 每臂汇总 ==")
    summary = {}
    for arm in sorted(runs):
        vals = [med(r["win"]) for r in runs[arm] if r["win"]]
        summary[arm] = vals
        print("  %-24s n=%d 中位数=%.1f ms 两点值=%s"
              % (LABEL.get(arm, arm), len(vals), med(vals) if vals else -1,
                 ["%.1f" % v for v in vals]))

    base = summary.get("a") or []
    if not base:
        print("BATCH4_FAIL 缺基线臂 a")
        return 2
    b = med(base)
    print("\n== 相对基线 a 的收益 ==")
    best = None
    for arm in sorted(summary):
        if arm == "a" or not summary[arm]:
            continue
        vals = summary[arm]
        ratios = [(b - v) / b * 100.0 for v in vals]
        same = all(r > 0 for r in ratios) or all(r < 0 for r in ratios)
        m = (b - med(vals)) / b * 100.0
        tag = "稳定" if same else "**方向不一致 ⇒ UNCERTAIN**"
        print("  %-24s 省时 %.2f%%（逐轮 %s）⇒ %s" % (LABEL.get(arm, arm), m,
                                                     ["%.2f" % r for r in ratios], tag))
        if same and (best is None or m > best[1]):
            best = (arm, m)
    if best and best[1] >= 5.0:
        print("BATCH4_VERDICT 候选：%s，省时 %.2f%%（两轮同向，≥5%%）" % (LABEL.get(best[0]), best[1]))
    elif best:
        print("BATCH4_VERDICT 无 ≥5%% 的稳定收益（最佳=%s，%.2f%%）" % (LABEL.get(best[0]), best[1]))
    else:
        print("BATCH4_VERDICT 无可比候选")

    print("\n== loss 可观测点与首点一致性（口径检查，不是数值判定）==")
    for arm in sorted(runs):
        for r in runs[arm]:
            first = r["loss"][0] if r["loss"] else None
            print("  %-8s 点数=%d 首点=%s" % (r["tag"], len(r["loss"]),
                                              ("%.7f" % first) if first else "-"))
    ref = (runs.get("a") or [{}])[0].get("loss") or []
    if ref:
        print("  说明：基线首点=%.7f；本检查只确认『log_interval>1 时点数变稀』这一口径差异，"
              "不做数值中性判定（同臂两轮本身自第 3 步就不同，该判据在本栈无区分力）" % ref[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
