#!/bin/bash
# P2：看清 Call Stack 的真实格式 + Input Shapes（只读，复用现有 profile）
set -u
ROOT=$(ls -dt /root/ops/prof62_*/ 2>/dev/null | head -1); ROOT=${ROOT%/}
AP=$(find "$ROOT/prof" -type d -name "ASCEND_PROFILER_OUTPUT" | head -1)
python3 - "$AP/operator_details.csv" <<'PY'
import csv, sys
p = sys.argv[1]
want = {"aten::transpose", "aten::contiguous", "aten::_to_copy", "aten::copy_"}
seen = {}
n = 0
with open(p, newline="", encoding="utf-8", errors="replace") as fh:
    r = csv.DictReader(fh)
    for row in r:
        n += 1
        if n > 400000:
            break
        nm = (row.get("Name") or "").strip()
        if nm in want and nm not in seen:
            seen[nm] = row
        if len(seen) == len(want):
            break
print("扫描记录数=%d  命中=%s" % (n, sorted(seen)))
for nm, row in seen.items():
    print("=" * 78)
    print("NAME: %s" % nm)
    print("Input Shapes: %r" % (row.get("Input Shapes") or "")[:400])
    cs = row.get("Call Stack") or ""
    print("Call Stack 长度=%d" % len(cs))
    print("--- 前 1200 字符（repr，看清分隔符与帧格式） ---")
    print(repr(cs[:1200]))
PY
echo "P2_FMT_DONE"
