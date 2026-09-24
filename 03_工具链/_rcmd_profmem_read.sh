#!/bin/bash
# 远端工作㉗：等分析结束 → 读桶与显存产物（含 operator_memory 聚合，若存在）
set -u
B=$(ls -1dt /root/ops/profmem3_* 2>/dev/null | head -1)
OUT=$B/run
for i in $(seq 1 30); do
  [ -f "$OUT/result.json" ] && break
  sleep 3
done
echo "=== 完成 ==="
[ -f "$OUT/result.json" ] && echo "RESULT_JSON 存在" || echo "仍无 result.json（可能分析很慢）"
grep -E 'P2PROF_' "$B/launch.log" 2>/dev/null | tail -3 || true
echo "=== 桶（若已完成）==="
python3 - "$OUT/result.json" <<'PY' 2>/dev/null || echo "  （result.json 尚不可读）"
import json, io, sys
d = json.load(io.open(sys.argv[1], encoding="utf-8"))
sp = d.get("shares_pct") or {}
print("  shares_pct:", {k: round(v, 2) for k, v in sp.items()})
print("  profiled_steps=%s wall=%s kernel_launches/step=%s" % (d.get("step_trace_rows"), d.get("wall_seconds"),
      d.get("kernel_launches_per_profiled_step")))
b = d.get("buckets") or {}
for k in ("Computing", "Free", "Communication", "Communication(Not Overlapped)"):
    if k in b: print("  %-32s mean=%.0f us" % (k, b[k]["mean"]))
PY
echo "=== 显存相关文件 ==="
find "$OUT/prof" -name '*memory*' -o -name '*Memory*' 2>/dev/null | head -8 || echo "（无）"
echo "=== operator_memory 聚合（若存在，按 Size 总和 Top 10）==="
M=$(find "$OUT/prof" -name 'operator_memory.csv' | head -1)
if [ -n "$M" ]; then
python3 - "$M" <<'PY'
import csv, io, sys, collections
rows = list(csv.DictReader(io.open(sys.argv[1], encoding="utf-8", errors="replace")))
print("  行数=%d 列=%s" % (len(rows), list(rows[0].keys())[:8] if rows else []))
size_col = next((c for c in (rows[0] or {}) if "Size" in c), None)
name_col = next((c for c in (rows[0] or {}) if "Name" in c or "name" in c), None)
agg = collections.Counter()
for r in rows:
    try:
        s = float(r.get(size_col) or 0)
    except ValueError:
        continue
    agg[(r.get(name_col) or "?")[:56]] = max(agg[(r.get(name_col) or "?")[:56]], s)
print("  --- 峰值占用 Top 10（同名列取最大）---")
for n, s in agg.most_common(10):
    print("   %-56s %.1f" % (n, s))
PY
else
  echo "  （无 operator_memory.csv）"
fi
echo "PROFMEM_READ_DONE"
