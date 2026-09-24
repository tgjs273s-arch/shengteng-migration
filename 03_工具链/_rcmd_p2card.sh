#!/bin/bash
# P2 候选卡：拷贝/转置/类型往返的 API 级证据（复用现有 profile）
set -u
ROOT=$(ls -dt /root/ops/prof62_*/ 2>/dev/null | head -1); ROOT=${ROOT%/}
AP=$(find "$ROOT/prof" -type d -name "ASCEND_PROFILER_OUTPUT" | head -1)
echo "=== api_statistic：与拷贝/转置/类型往返相关的行 ==="
grep -iE "contiguous|cast|transpose|copy|to_|_to|view|reshape|cat|mul|add" "$AP/api_statistic.csv" 2>/dev/null | sed '/^$/d' | head -40
echo
echo "=== operator_details.csv 表头（看有没有调用栈列） ==="
head -1 "$AP/operator_details.csv" 2>/dev/null | tr ',' '\n' | head -30
echo "行数: $(wc -l < "$AP/operator_details.csv" 2>/dev/null)"
echo "=== operator_details 里与 contiguous 相关的聚合（若有 Name 列） ==="
python3 - "$AP/operator_details.csv" <<'PY'
import csv, sys, collections
p = sys.argv[1]
agg = collections.defaultdict(lambda: [0, 0.0])
n = 0
with open(p, newline="", encoding="utf-8", errors="replace") as fh:
    r = csv.DictReader(fh)
    cols = r.fieldnames or []
    ncol = next((c for c in cols if c and c.lower() in ("name", "op name", "api name", "opname")), None)
    tcol = next((c for c in cols if c and "duration" in c.lower()), None)
    for row in r:
        n += 1
        if n > 400000:
            break
        nm = (row.get(ncol) or "") if ncol else ""
        if any(k in nm.lower() for k in ("contiguous", "cast", "transpose", "copy")):
            try: d = float(row.get(tcol) or 0)
            except Exception: d = 0.0
            e = agg[nm]; e[0] += 1; e[1] += d
print("扫描行数=%d  Name列=%s  Dur列=%s" % (n, ncol, tcol))
for k, (c, d) in sorted(agg.items(), key=lambda x: -x[1][1])[:15]:
    print("  %-52s calls=%6d  dur=%12.0f us" % (k[:52], c, d))
PY
echo "P2_CARD_DONE"
