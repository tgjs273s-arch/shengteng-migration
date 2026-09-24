#!/bin/bash
# 远端工作㉓：从 kernel_details.csv 聚合"算子耗时占比"（回答：是否单个算子占据关键路径）
# 纪律：只看**比例**；被 profiling 拖慢的绝对时间不用于对标。
set -u
OUT=/root/ops/prof_20260922_112113
K=$(find "$OUT/prof" -name 'kernel_details.csv' | head -1)
echo "KERNEL_CSV=$K"
[ -n "$K" ] || { echo "PROF_KERNEL_FAIL 找不到 kernel_details.csv"; exit 1; }
python3 - "$K" <<'PY'
import csv, io, sys, collections
p = sys.argv[1]
rows = list(csv.DictReader(io.open(p, encoding="utf-8", errors="replace")))
if not rows:
    print("PROF_KERNEL_FAIL CSV 无数据行"); raise SystemExit(1)
cols = list(rows[0].keys())
name_col = next((c for c in cols if "Name" in c), None)
dur_col = next((c for c in cols if "Duration" in c and "us" in c.lower()), None) or \
          next((c for c in cols if "Duration" in c), None)
print("列名:", cols[:8])
print("用列: name=%r duration=%r" % (name_col, dur_col))
agg = collections.Counter()
n_by = collections.Counter()
tot = 0.0
for r in rows:
    try:
        d = float(r.get(dur_col) or 0)
    except ValueError:
        continue
    n = (r.get(name_col) or "?").strip()
    agg[n] += d
    n_by[n] += 1
    tot += d
print("kernel 行数=%d 总耗时=%.1f us（仅比例可用）" % (len(rows), tot))
print("--- 耗时占比 Top 15 ---")
for i, (n, d) in enumerate(agg.most_common(15), 1):
    print("  %2d. %-58s %9.1f us  %5.2f%%  调用 %d 次" % (i, n[:58], d, d / tot * 100.0, n_by[n]))
print("--- 按调用次数 Top 8 ---")
for i, (n, c) in enumerate(n_by.most_common(8), 1):
    print("  %2d. %-58s %d 次  %.1f us  占时 %.2f%%" % (i, n[:58], c, agg[n], agg[n] / tot * 100.0))
top1 = agg.most_common(1)[0]
print("PROF_KERNEL_TOP1 %s = %.2f%%" % (top1[0][:60], top1[1] / tot * 100.0))
print("PROF_KERNEL_DONE")
PY
