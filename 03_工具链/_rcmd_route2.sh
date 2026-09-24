#!/bin/bash
# 路线 2 候选卡取证：GDN 反向 kernel 的**设备侧**耗时 + 调用点 + shape
set -u
FRAME_RE_DOC=''
P1=$(ls -dt /root/ops/prof62_*/ 2>/dev/null | head -1); P1=${P1%/}
P2D=$(ls -dt /root/ops/p2shapes_*/ 2>/dev/null | head -1); P2D=${P2D%/}
A1=$(find "$P1/prof" -type d -name ASCEND_PROFILER_OUTPUT 2>/dev/null | head -1)
A2=$(find "$P2D/prof" -type d -name ASCEND_PROFILER_OUTPUT 2>/dev/null | head -1)
echo "prof62=$A1"
echo "p2shapes=$A2"
python3 - "$A1/operator_details.csv" "$A2/operator_details.csv" <<'PY'
import csv, sys, re, collections, os
p1, p2 = sys.argv[1], sys.argv[2]
FRAME = re.compile(r"^(.+?)\((\d+)\):\s*(.+)$")
TARGETS = ("wy_repr", "causal_conv1d", "chunk_bwd", "chunk_gated_delta")
def scan(path, want_shapes, limit=400000):
    agg = collections.defaultdict(lambda: [0, 0.0, 0.0, None])
    n = 0
    if not os.path.isfile(path):
        return agg, 0
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            n += 1
            if n > limit:
                break
            nm = (row.get("Name") or "").strip()
            if not any(t in nm for t in TARGETS):
                continue
            cs = row.get("Call Stack") or ""
            proj = None
            for part in cs.replace("\r\n", "\n").split(";"):
                part = part.strip()
                if not part:
                    continue
                m = FRAME.match(part)
                if m and ("MindSpeed-MM" in m.group(1) or "qwen35" in m.group(1)):
                    proj = "%s(%s): %s" % (m.group(1).split("/")[-1], m.group(2), m.group(3).strip())
                    break
            try:
                ds = float(row.get("Device Self Duration(us)") or 0)
                dt = float(row.get("Device Total Duration(us)") or 0)
            except Exception:
                ds = dt = 0.0
            k = "%s | %s" % (nm, proj or "?")
            e = agg[k]; e[0] += 1; e[1] += ds; e[2] += dt
            if want_shapes and e[3] is None and (row.get("Input Shapes") or "").strip():
                e[3] = (row.get("Input Shapes") or "")[:180]
    return agg, n
agg, n1 = scan(p1, False)
print("=== prof62 设备侧（扫描 %d 条） ===")
for k, (c, ds, dt, _s) in sorted(agg.items(), key=lambda x: -x[1][1])[:14]:
    print("  %-74s calls=%5d dev_self=%10.0f dev_total=%10.0f  (≈%.0f 次/步)"
          % (k[:74], c, ds, dt, c / 5.0))
agg2, n2 = scan(p2, True)
print("=== p2shapes 的 Input Shapes（扫描 %d 条） ===")
for k, (c, ds, dt, s) in sorted(agg2.items(), key=lambda x: -x[1][1])[:10]:
    print("  %-64s calls=%5d dev_self=%9.0f" % (k[:64], c, ds))
    print("      shapes=%r" % (s,))
PY
echo "P2_ROUTE2_DONE"
