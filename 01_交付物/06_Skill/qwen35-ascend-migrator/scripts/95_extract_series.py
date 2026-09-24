#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
95_extract_series.py —— 从训练日志提取逐点序列（iter / ms / loss / grad_norm）

用途：把 train.log 变成可复核的 CSV，供：
  · 与官方基线的逐点比对
  · 50–100 窗口的均值/中位（官方性能口径）
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
import re
import statistics as st
import sys

# 兼容 MSMM 的日志格式：
# [Rank 0 | Local Rank 0] 2026-... INFO [...train_engine:48] =>  [..] iteration  1/  100 \
#   | consumed samples: 8 | elapsed time per iteration (ms): 7137.9 | learning rate: ... \
#   | global batch size: 8 | loss: 1.924620E+00 | grad norm: 134.300 |
PAT = re.compile(
    r"iteration\s+(\d+)\s*/\s*(\d+).*?"
    r"elapsed time per iteration \(ms\):\s*([\d.]+).*?"
    r"loss:\s*([\d.eE+-]+).*?"
    r"grad norm:\s*([\d.eE+-]+)",
    re.S)


def parse(log_path):
    rows = []
    bad = 0
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if "iteration" not in ln:
                continue
            m = PAT.search(ln)
            if not m:
                bad += 1
                continue
            rows.append({
                "iter": int(m.group(1)),
                "total": int(m.group(2)),
                "ms": float(m.group(3)),
                "loss": float(m.group(4)),
                "grad_norm": float(m.group(5)),
            })
    return rows, bad


def main():
    ap = argparse.ArgumentParser(description="从训练日志提取逐点序列")
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default=None, help="输出 CSV 路径（缺省 <log>.series.csv）")
    ap.add_argument("--baseline-csv", default=None,
                    help="官方基线 CSV（含 iter,loss[,grad_norm] 列）→ 顺带算偏差")
    ap.add_argument("--window", default="50,100", help="窗口（默认 50,100，官方性能口径）")
    ap.add_argument("--summary-json", default=None,
                    help="★ 把窗口指标落盘成 JSON（供驱动/判定链断言，见坑 119）")
    args = ap.parse_args()

    if not os.path.isfile(args.log):
        print("FATAL 日志不存在: %s" % args.log, file=sys.stderr)
        return 2
    rows, bad = parse(args.log)
    if not rows:
        print("FATAL 未解析到任何 iteration 行（检查是否为 MSMM trainer 日志）", file=sys.stderr)
        return 2

    out = args.out or (args.log + ".series.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["iter", "total", "ms", "loss", "grad_norm"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    iters = [r["iter"] for r in rows]
    print("EXTRACT_OK rows=%d  iter=%d..%d  unparsed_iteration_lines=%d"
          % (len(rows), min(iters), max(iters), bad))
    print("  产物: %s" % out)

    lo, hi = (int(x) for x in args.window.split(","))
    win = [r for r in rows if lo <= r["iter"] <= hi]
    if win:
        mss = [r["ms"] for r in win]
        print("  窗口 %d-%d: n=%d  中位 ms=%.1f  均值 ms=%.1f  min=%.1f  max=%.1f"
              % (lo, hi, len(win), st.median(mss), sum(mss) / len(mss), min(mss), max(mss)))
        print("  吞吐(GBS=8): 中位 %.2f samples/s"
              % (8 * 1000.0 / st.median(mss)))
    else:
        print("  窗口 %d-%d: 无数据（训练步数不足）" % (lo, hi))

    print("  首步: %s" % ({k: rows[0][k] for k in ("iter", "ms", "loss", "grad_norm")}))
    print("  末步: %s" % ({k: rows[-1][k] for k in ("iter", "ms", "loss", "grad_norm")}))

    # ---- ★ 窗口指标落盘（坑 119）：P6 阶段的判据必须能对"50-100 口径"做断言，
    #   而不是只查"某个 json 存在"。这里把可断言的量显式产出。
    if args.summary_json:
        s = {
            "schema": "loss_series_window.v1",
            "log": os.path.abspath(args.log),
            "csv": os.path.abspath(out),
            "window": args.window,
            "rows": len(rows),
            "iter_first": min(iters),
            "iter_last": max(iters),
            "unparsed_iteration_lines": bad,
            "steps_used": len(win),
            "median_ms": st.median(mss) if win else None,
            "mean_ms": (sum(mss) / len(mss)) if win else None,
            "min_ms": min(mss) if win else None,
            "max_ms": max(mss) if win else None,
            # 吞吐口径：GBS=8（官方几何）→ samples/s
            "samples_per_s": (8 * 1000.0 / st.median(mss)) if win else None,
            "loss_median": st.median([r["loss"] for r in win]) if win else None,
            "gn_median": st.median([r["grad_norm"] for r in win]) if win else None,
            "step1_loss": rows[0]["loss"],
            "step1_grad_norm": rows[0]["grad_norm"],
            "last_loss": rows[-1]["loss"],
            "last_grad_norm": rows[-1]["grad_norm"],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)) or ".", exist_ok=True)
        import json as _json
        with open(args.summary_json, "w", encoding="utf-8") as f:
            _json.dump(s, f, ensure_ascii=False, indent=1)
        print("WINDOW_JSON_OK path=%s steps_used=%d median_ms=%s samples_per_s=%s"
              % (args.summary_json, s["steps_used"],
                 ("%.1f" % s["median_ms"]) if s["median_ms"] else "None",
                 ("%.2f" % s["samples_per_s"]) if s["samples_per_s"] else "None"))

    # 与官方基线逐点比对（若提供）
    if args.baseline_csv and os.path.isfile(args.baseline_csv):
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
