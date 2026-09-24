#!/bin/bash
# P2 候选卡：**修正取帧逻辑**（取最内层项目帧）+ 分调用点计数
set -u
ROOT=$(ls -dt /root/ops/prof62_*/ 2>/dev/null | head -1); ROOT=${ROOT%/}
AP=$(find "$ROOT/prof" -type d -name "ASCEND_PROFILER_OUTPUT" | head -1)
python3 - "$AP/operator_details.csv" <<'PY'
import csv, sys, collections, re
p = sys.argv[1]
TARGETS = ("aten::copy_", "aten::transpose", "aten::contiguous", "aten::_to_copy")
FRAME = re.compile(r"^(.+?)\((\d+)\):\s*(.+)$")

def frames_of(cs):
    out = []
    for part in (cs or "").replace("\r\n", "\n").split(";"):
        part = part.strip()
        if not part:
            continue
        m = FRAME.match(part)
        if m:
            out.append((m.group(1), m.group(2), m.group(3).strip()))
    return out

def short(f):
    return "%s(%s): %s" % (f[0].split("/")[-1], f[1], f[2])

# 栈是**内层在前** ⇒ 最内层项目帧 = 第一个含 MindSpeed-MM 的帧（修正上一版取成最外层的错误）
per_op = collections.defaultdict(lambda: [0, 0.0])
site = collections.defaultdict(lambda: [0, 0.0])
inner_any = collections.defaultdict(lambda: [0, 0.0])
n = 0
with open(p, newline="", encoding="utf-8", errors="replace") as fh:
    for row in csv.DictReader(fh):
        n += 1
        if n > 400000:
            break
        nm = (row.get("Name") or "").strip()
        if nm not in TARGETS:
            continue
        try:
            d = float(row.get("Host Self Duration(us)") or 0)
        except Exception:
            d = 0.0
        per_op[nm][0] += 1; per_op[nm][1] += d
        fs = frames_of(row.get("Call Stack"))
        if fs:
            inner_any[short(fs[0])][0] += 1; inner_any[short(fs[0])][1] += d
        proj = next((f for f in fs if "MindSpeed-MM" in f[0] or "qwen35" in f[0]), None)
        if proj:
            k = "%s | %s" % (nm, short(proj))
            site[k][0] += 1; site[k][1] += d
print("扫描记录数=%d" % n)
print("=== per-op 总量（host self） ===")
for k, (c, d) in sorted(per_op.items(), key=lambda x: -x[1][1]):
    print("  %-20s calls=%6d  host_self=%10.0f us  (≈%.0f 次/步)" % (k, c, d, c / 5.0))
print("=== 最内层**任意**帧 top（说明算子由谁发出） ===")
for k, (c, d) in sorted(inner_any.items(), key=lambda x: -x[1][1])[:10]:
    print("  %-56s calls=%6d host_self=%10.0f" % (k[:56], c, d))
print("=== 最内层**项目代码**帧 top（= 可改的源码位置） ===")
for k, (c, d) in sorted(site.items(), key=lambda x: -x[1][1])[:18]:
    print("  %-72s calls=%6d host_self=%10.0f" % (k[:72], c, d))
PY
echo "P2_ATTRIB2_DONE"
