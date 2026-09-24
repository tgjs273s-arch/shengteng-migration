#!/bin/bash
# P2 候选卡：把 transpose/contiguous/_to_copy/copy_ 归因到**调用栈**
set -u
ROOT=$(ls -dt /root/ops/prof62_*/ 2>/dev/null | head -1); ROOT=${ROOT%/}
AP=$(find "$ROOT/prof" -type d -name "ASCEND_PROFILER_OUTPUT" | head -1)
python3 - "$AP/operator_details.csv" <<'PY'
import csv, sys, collections, re
p = sys.argv[1]
TARGETS = ("aten::transpose", "aten::contiguous", "aten::_to_copy", "aten::copy_")
agg = collections.defaultdict(lambda: [0, 0.0])
frames = collections.defaultdict(lambda: collections.Counter())
n = 0
with open(p, newline="", encoding="utf-8", errors="replace") as fh:
    r = csv.DictReader(fh)
    for row in r:
        n += 1
        if n > 300000:
            break
        nm = (row.get("Name") or "").strip()
        if nm not in TARGETS:
            continue
        try:
            d = float(row.get("Host Self Duration(us)") or 0)
        except Exception:
            d = 0.0
        e = agg[nm]; e[0] += 1; e[1] += d
        cs = row.get("Call Stack") or ""
        # 取调用栈里最后出现的本项目/框架源码帧
        cands = re.findall(r"([\w./\-]+\.py):(\d+)\((\w+)\)", cs)
        keep = [c for c in cands if any(k in c[0] for k in ("MindSpeed", "mindspeed_mm", "qwen35"))]
        if keep:
            f = keep[-1]
            frames[nm]["%s:%s(%s)" % (f[0].split("/")[-1], f[1], f[2])] += 1
        elif cands:
            f = cands[-1]
            frames[nm]["%s:%s(%s)" % (f[0].split("/")[-1], f[1], f[2])] += 1
print("扫描记录数=%d" % n)
print("=== 命中统计 ===")
for k, (c, d) in sorted(agg.items(), key=lambda x: -x[1][1]):
    print("  %-20s calls=%6d  host_self=%10.0f us   (≈%.0f 次/步, 共 5 步)" % (k, c, d, c / 5.0))
print("=== 每个算子的 top 调用点（末帧） ===")
for nm in TARGETS:
    print("-- %s" % nm)
    for frame, c in frames[nm].most_common(6):
        print("     %6d  %s" % (c, frame))
PY
echo "P2_STACK_DONE"
