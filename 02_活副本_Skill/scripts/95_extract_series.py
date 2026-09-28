#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
95_extract_series.py —— 从训练日志提取逐点序列（iter / ms / loss / grad_norm）

用途：把 train.log 变成可复核的 CSV，供：
  · 与官方基线的逐点比对
  · 项目保存脚本的前 200 条中后 100 条均值复算，及独立的 50–100 诊断窗口均值/中位
  · 复算"超 2% 步数"等判据

★ 为什么独立成脚本（而不是在 SSH 命令行里用 heredoc）：
  实测把 heredoc 拼进远程命令字符串时，**定界符会被外层引号处理破坏**，
  报 `warning: here-document delimited by end-of-file` 并 SyntaxError。
  **凡多行脚本，一律落盘成文件再执行**（与坑 21 的教训一致）。

用法：
  python3 scripts/95_extract_series.py --log out/train/train.log \
      --out out/train/loss_series.csv [--baseline-csv <官方CSV>]
"""
import argparse
import csv
import os
import statistics as st
import sys
import json
from _train_log import read_log, integrity

def parse(log_path):
    result = read_log(log_path)
    return result["rows"], len(result["issues"])


def metrics(rows, gbs):
    times = [row["ms"] for row in rows]
    mean_ms, median_ms = st.mean(times), st.median(times)
    return {"steps": [row["iter"] for row in rows], "points": len(rows),
            "mean_ms": mean_ms, "median_ms": median_ms,
            "min_ms": min(times), "max_ms": max(times),
            "mean_throughput_samples_per_s": gbs * 1000 / mean_ms,
            "median_step_rate_samples_per_s": gbs * 1000 / median_ms}


def main():
    ap = argparse.ArgumentParser(description="从训练日志提取逐点序列")
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default=None, help="输出 CSV 路径（缺省 <log>.series.csv）")
    ap.add_argument("--baseline-csv", default=None,
                    help="官方基线 CSV（含 iter,loss[,grad_norm] 列）→ 顺带算偏差")
    ap.add_argument("--window", default="50,100", help="诊断窗口（默认 50,100，含端点）")
    ap.add_argument("--summary-json", default=None,
                    help="★ 把窗口指标落盘成 JSON（供驱动/判定链断言，见坑 119）")
    ap.add_argument("--expected-gbs", type=int, default=None,
                    help="可选：与有效配置核对的 GBS；缺省仅核对日志内部一致")
    args = ap.parse_args()

    if not os.path.isfile(args.log):
        print("FATAL 日志不存在: %s" % args.log, file=sys.stderr)
        return 2
    try:
        lo, hi = (int(x) for x in args.window.split(","))
        if lo < 1 or hi < lo or (args.expected_gbs is not None and args.expected_gbs < 1):
            raise ValueError("range or GBS")
    except ValueError:
        print("FATAL 无效窗口或 GBS", file=sys.stderr)
        return 2
    parsed = read_log(args.log)
    rows, bad = parsed["rows"], len(parsed["issues"])
    totals = sorted({r["total"] for r in rows})
    total = totals[0] if len(totals) == 1 else None
    check = integrity(parsed, expected_end=total, expected_gbs=args.expected_gbs)
    rank_rows = [r for r in rows if r["rank"] == 0]
    gbs = check["gbs_values"][0] if len(check["gbs_values"]) == 1 else None
    formal = check["state"] == "COMPLETE" and gbs is not None

    out = args.out or (args.log + ".series.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["iter", "total", "ms", "loss", "grad_norm",
                                        "rank", "local_rank", "line", "consumed_samples", "gbs",
                                        "schema"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "schema": "train_series.v2"})

    iters = [r["iter"] for r in rank_rows]
    print("EXTRACT_%s rows=%d  iter=%d..%d  unparsed_iteration_lines=%d"
          % ("OK" if formal else "INCOMPLETE", len(rows),
             min(iters) if iters else 0, max(iters) if iters else 0, bad))
    print("  产物: %s" % out)

    win = [r for r in rank_rows if lo <= r["iter"] <= hi]
    window_complete = (formal and total is not None and hi <= total
                       and len(win) == hi - lo + 1 and len(win) >= 2
                       and {r["iter"] for r in win} == set(range(lo, hi + 1)))
    official = rank_rows[:200][-100:]
    official_complete = formal and len(official) == 100
    window_metrics = metrics(win, gbs) if window_complete else None
    official_metrics = metrics(official, gbs) if official_complete else None
    problems = list(check["problems"])
    if not window_complete:
        problems.append("window_incomplete")
    if not official_complete:
        problems.append("official_selection_incomplete")
    if window_complete:
        mss = [r["ms"] for r in win]
        print("  窗口 %d-%d: n=%d  中位 ms=%.1f  均值 ms=%.1f  min=%.1f  max=%.1f"
              % (lo, hi, len(win), st.median(mss), sum(mss) / len(mss), min(mss), max(mss)))
        print("  中位步时折算速度(GBS=%d): %.2f samples/s"
              % (gbs, gbs * 1000.0 / st.median(mss)))
    else:
        print("  窗口 %d-%d: 不完整，不能用于正式统计" % (lo, hi))

    if official_metrics:
        print("  保存脚本口径 first_200_then_last_100: n=100 均值 %.6f ms 吞吐 %.6f samples/s"
              % (official_metrics["mean_ms"], official_metrics["mean_throughput_samples_per_s"]))

    if rank_rows:
        print("  首记录: %s" % ({k: rank_rows[0][k] for k in ("iter", "ms", "loss", "grad_norm")}))
        print("  末记录: %s" % ({k: rank_rows[-1][k] for k in ("iter", "ms", "loss", "grad_norm")}))

    # ---- ★ 窗口指标落盘（坑 119）：P6 阶段的判据必须能对"50-100 口径"做断言，
    #   而不是只查"某个 json 存在"。这里把可断言的量显式产出。
    if args.summary_json:
        s = {
            "schema": "train_performance.v2",
            "state": "COMPLETE" if not problems else "INCOMPLETE",
            "problems": problems,
            "source_type": "training_iteration_log",
            "log_sha256": parsed["log_sha256"],
            "rank": 0,
            "gbs": gbs,
            "gbs_source": "log_checked" if args.expected_gbs is None else "log_and_config_checked",
            "integrity": check,
            "official_selection": {"selection": "first_200_then_last_100",
                                   "source": "project_saved_official_script_interpretation",
                                   "official_rule_status": "UNVERIFIED_FOR_CURRENT_CONTEST",
                                   "aggregation": "arithmetic_mean_iteration_ms",
                                   "metrics": official_metrics},
            "window_selection": {"selection": "inclusive_step_range",
                                 "range": [lo, hi],
                                 "aggregation": "mean_and_median_iteration_ms",
                                 "metrics": window_metrics},
            "log": os.path.abspath(args.log),
            "csv": os.path.abspath(out),
            "window": args.window,
            "rows": len(rows),
            "iter_first": min(iters) if iters else None,
            "iter_last": max(iters) if iters else None,
            "unparsed_iteration_lines": bad,
            "steps_used": len(win) if window_complete else 0,
            "median_ms": window_metrics["median_ms"] if window_metrics else None,
            "mean_ms": window_metrics["mean_ms"] if window_metrics else None,
            "min_ms": window_metrics["min_ms"] if window_metrics else None,
            "max_ms": window_metrics["max_ms"] if window_metrics else None,
            # v1 compatibility: samples_per_s was median-step-rate, not mean throughput.
            "samples_per_s": (window_metrics["median_step_rate_samples_per_s"]
                              if window_metrics else None),
            "loss_median": st.median([r["loss"] for r in win]) if window_complete else None,
            "gn_median": st.median([r["grad_norm"] for r in win]) if window_complete else None,
            "step1_loss": next((r["loss"] for r in rank_rows if r["iter"] == 1), None),
            "step1_grad_norm": next((r["grad_norm"] for r in rank_rows if r["iter"] == 1), None),
            "last_loss": rank_rows[-1]["loss"] if rank_rows else None,
            "last_grad_norm": rank_rows[-1]["grad_norm"] if rank_rows else None,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)) or ".", exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=1)
        print("WINDOW_JSON_%s path=%s steps_used=%d median_ms=%s samples_per_s=%s"
              % ("OK" if window_complete else "INCOMPLETE", args.summary_json, s["steps_used"],
                 ("%.1f" % s["median_ms"]) if s["median_ms"] else "None",
                 ("%.2f" % s["samples_per_s"]) if s["samples_per_s"] else "None"))

    # 与官方基线逐点比对（若提供）
    if args.baseline_csv and not os.path.isfile(args.baseline_csv):
        print("FATAL 基线 CSV 不存在: %s" % args.baseline_csv, file=sys.stderr)
        return 2
    if args.baseline_csv and formal:
        base = {}
        with open(args.baseline_csv, encoding="utf-8", errors="replace") as f:
            rd = csv.DictReader(f)
            for r0 in rd:
                try:
                    it = int(float(r0.get("iter") or r0.get("iteration")))
                except Exception:
                    continue
                try:
                    base[it] = {
                        "loss": float(r0.get("loss")) if r0.get("loss") else None,
                        "grad_norm": float(r0.get("grad_norm") or r0.get("gn") or 0) or None,
                    }
                except Exception:
                    pass
        dl, dg, over = [], [], 0
        for r in rows:
            b = base.get(r["iter"])
            if not b or not b["loss"]:
                continue
            e = abs(r["loss"] - b["loss"]) / abs(b["loss"]) * 100.0
            dl.append(e)
            if e > 2.0:
                over += 1
        if dl:
            print("  对比基线(%s): n=%d  loss 偏差 Mean=%.4f%% Max=%.4f%%  超2%%步数=%d"
                  % (args.baseline_csv, len(dl), sum(dl) / len(dl), max(dl), over))
    return 0 if not problems else 3


if __name__ == "__main__":
    sys.exit(main())
