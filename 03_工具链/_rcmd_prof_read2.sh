set -u
OUT=/root/ops/prof_20260922_112113
python3 - <<'PY'
import json, io
d = json.load(io.open("/root/ops/prof_20260922_112113/result.json", encoding="utf-8"))
print("顶层键:", sorted(d.keys()))
for k in sorted(d):
    v = d[k]
    if isinstance(v, dict) and any(s in k.lower() for s in ("bucket", "trace", "kernel", "mean", "stage", "prof")):
        print("---", k)
        for kk, vv in v.items():
            print("   %-34s %s" % (kk, vv))
    elif not isinstance(v, (dict, list)) and k not in ("identity",):
        print("%-30s %s" % (k, v))
PY
echo "=== step_trace_time.csv 表头 ==="
C=$(find "$OUT/prof" -name step_trace_time.csv | head -1)
head -1 "$C" | tr ',' '\n' | head -14 | nl
echo "PROF_READ2_DONE"
