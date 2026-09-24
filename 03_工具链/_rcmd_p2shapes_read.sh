#!/bin/bash
# P2：补 shape/dtype/stride（若 p2shapes 已完成）+ 读 hotspot 源码行
set -u
echo "=== p2shapes 进度 ==="
R=$(ls -dt /root/ops/p2shapes_*/ 2>/dev/null | head -1); R=${R%/}
echo "ROOT=$R"
tail -n 5 "$R.launch.log" 2>/dev/null || echo "(无 launch.log)"
[ -f "$R/result.json" ] && echo "RESULT_READY" || echo "not_ready"

echo
echo "=== hotspot 源码：chunk_o.py（找 chunk_bwd_dqkwg 第 443 行附近） ==="
F=$(find /root/MindSpeed-MM -name "chunk_o.py" 2>/dev/null | head -1)
echo "FILE=$F"
[ -n "$F" ] && sed -n '425,455p' "$F"

echo
echo "=== hotspot 源码：fsdp/utils/utils.py L88-108（move_to_device） ==="
sed -n '88,108p' /root/MindSpeed-MM/mindspeed_mm/fsdp/utils/utils.py

echo
echo "=== 若 p2shapes 就绪：取出这两个算子的 Input Shapes ==="
AP=$(find "$R/prof" -type d -name "ASCEND_PROFILER_OUTPUT" 2>/dev/null | head -1)
if [ -n "${AP:-}" ] && [ -f "$AP/operator_details.csv" ]; then
python3 - "$AP/operator_details.csv" <<'PY'
import csv, sys, re, collections
p = sys.argv[1]
FRAME = re.compile(r"^(.+?)\((\d+)\):\s*(.+)$")
want = ("chunk_o.py", "utils.py")
seen = collections.Counter()
n = 0
with open(p, newline="", encoding="utf-8", errors="replace") as fh:
    for row in csv.DictReader(fh):
        n += 1
        if n > 200000:
            break
        nm = (row.get("Name") or "").strip()
        if nm not in ("aten::copy_", "aten::contiguous", "aten::transpose", "aten::_to_copy"):
            continue
        cs = row.get("Call Stack") or ""
        for part in cs.replace("\r\n", "\n").split(";"):
            part = part.strip()
            if not part:
                continue
            m = FRAME.match(part)
            if m and any(w in m.group(1) for w in want):
                k = (nm, "%s(%s): %s" % (m.group(1).split("/")[-1], m.group(2), m.group(3).strip()))
                if seen[k] < 3:
                    seen[k] += 1
                    print("  %-22s %-52s shapes=%r" % (k[0], k[1][:52], (row.get("Input Shapes") or "")[:160]))
print("扫描=%d" % n)
PY
else
  echo "(p2shapes 产物还没就绪)"
fi
echo "P2_SHAPES_DONE"
