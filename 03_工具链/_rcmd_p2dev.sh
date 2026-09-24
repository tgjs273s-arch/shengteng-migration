#!/bin/bash
# P2：chunk_o.py 那个算子的**设备侧**耗时 + 执行分支判定（只读）
set -u
ROOT=$(ls -dt /root/ops/prof62_*/ 2>/dev/null | head -1); ROOT=${ROOT%/}
AP=$(find "$ROOT/prof" -type d -name "ASCEND_PROFILER_OUTPUT" | head -1)
echo "AP=$AP"
python3 - "$AP/operator_details.csv" <<'PY'
import csv, sys, re, collections
p = sys.argv[1]
FRAME = re.compile(r"^(.+?)\((\d+)\):\s*(.+)$")
hits = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0])
branch = collections.Counter()
n = 0
with open(p, newline="", encoding="utf-8", errors="replace") as fh:
    for row in csv.DictReader(fh):
        n += 1
        if n > 400000:
            break
        cs = row.get("Call Stack") or ""
        if "chunk_o.py" not in cs:
            continue
        nm = (row.get("Name") or "").strip()
        # 分支判定：哪个分支的构造算子出现在 chunk_o 的栈里
        if nm in ("aten::arange",): branch["else分支(arange/bias)"] += 1
        if nm in ("aten::cat",): branch["else分支(cat/zeros)"] += 1
        if nm in ("aten::tril",): branch["if分支(tril掩码)"] += 1
        if nm in ("aten::zeros", "aten::empty"): branch["zeros/empty"] += 1
        try:
            hs = float(row.get("Host Self Duration(us)") or 0)
            ds = float(row.get("Device Self Duration(us)") or 0)
            dt = float(row.get("Device Total Duration(us)") or 0)
        except Exception:
            hs = ds = dt = 0.0
        proj = None
        for part in cs.replace("\r\n", "\n").split(";"):
            part = part.strip()
            if not part:
                continue
            m = FRAME.match(part)
            if m and "chunk_o.py" in m.group(1):
                proj = "%s(%s): %s" % (m.group(1).split("/")[-1], m.group(2), m.group(3).strip())
                break
        k = "%s | %s" % (nm, proj or "?")
        e = hits[k]; e[0] += 1; e[1] += hs; e[2] += ds; e[3] += dt
print("扫描=%d  含 chunk_o.py 的记录=%d" % (n, sum(v[0] for v in hits.values())))
print("=== 分支线索（构造算子出现次数） ===")
for k, v in branch.most_common():
    print("   %-28s %d" % (k, v))
print("=== top（host vs device，单位 us，5 步合计） ===")
for k, (c, hs, ds, dt) in sorted(hits.items(), key=lambda x: -x[1][1])[:12]:
    print("  %-58s calls=%5d host_self=%10.0f dev_self=%9.0f dev_total=%9.0f"
          % (k[:58], c, hs, ds, dt))
PY
echo "=== p2shapes 状态 ==="
R=$(ls -dt /root/ops/p2shapes_*/ 2>/dev/null | head -1); R=${R%/}
[ -f "$R/result.json" ] && echo "P2SHAPES_READY" || { echo "not_ready"; tail -n 3 "$R.launch.log" 2>/dev/null; }
echo "P2_DEV_DONE"
