#!/bin/bash
# P2 候选卡：修正后的调用点归因（格式：path(line): func;  以 ';\r\n' 分隔）
set -u
ROOT=$(ls -dt /root/ops/prof62_*/ 2>/dev/null | head -1); ROOT=${ROOT%/}
AP=$(find "$ROOT/prof" -type d -name "ASCEND_PROFILER_OUTPUT" | head -1)
python3 - "$AP/operator_details.csv" <<'PY'
import csv, sys, collections, re
p = sys.argv[1]
TARGETS = ("aten::copy_", "aten::transpose", "aten::contiguous", "aten::_to_copy")
FRAME = re.compile(r"^(.+?)\((\d+)\):\s*(.+)$")
agg = collections.defaultdict(lambda: [0, 0.0])
per_op = collections.defaultdict(lambda: [0, 0.0])
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
        cs = row.get("Call Stack") or ""
        frames = []
        for part in cs.replace("\r\n", "\n").split(";"):
            part = part.strip()
            if not part:
                continue
            m = FRAME.match(part)
            if m:
                frames.append((m.group(1), m.group(2), m.group(3).strip()))
        # 取**最后一个**落在项目代码（MindSpeed-MM）里的帧作为归因点
        site = None
        for f in frames:
            if "MindSpeed-MM" in f[0] or "qwen35" in f[0]:
                site = f
        if site is None and frames:
            site = frames[-1]
        if site:
            key = "%s(%s): %s" % (site[0].split("/")[-1], site[1], site[2])
            agg[key][0] += 1; agg[key][1] += d
print("扫描记录数=%d" % n)
print("=== 每个算子的总量（host self） ===")
for k, (c, d) in sorted(per_op.items(), key=lambda x: -x[1][1]):
    print("  %-20s calls=%6d host_self=%10.0f us  (≈%.0f 次/步)" % (k, c, d, c / 5.0))
print("=== top 调用点（项目代码帧；host self） ===")
for k, (c, d) in sorted(agg.items(), key=lambda x: -x[1][1])[:18]:
    print("  %-58s calls=%6d  host_self=%10.0f us" % (k[:58], c, d))
PY
echo "P2_ATTRIB_DONE"
