#!/bin/bash
# §4-2 归因分析：kernel_details 明细 → 找 grad-norm 与 all_reduce 的占比（含 Wait Time）
set -u
OUT=$(ls -dt /root/ops/prof62_*/ 2>/dev/null | head -1); OUT=${OUT%/}
AP=$(find "$OUT/prof" -type d -name "ASCEND_PROFILER_OUTPUT" | head -1)
echo "AP=$AP"
echo "=== ASCEND_PROFILER_OUTPUT 文件清单 ==="
ls -la "$AP" 2>/dev/null
echo "=== 归因分析 ==="
python3 - "$AP" <<'PY'
import csv, sys, os, collections
ap = sys.argv[1]
kd = os.path.join(ap, "kernel_details.csv")
tot_dur = 0.0; tot_wait = 0.0; n = 0
by = collections.defaultdict(lambda: [0, 0.0, 0.0])
with open(kd, newline='', encoding='utf-8', errors='replace') as fh:
    for row in csv.DictReader(fh):
        n += 1
        try: d = float(row.get('Duration(us)') or 0)
        except Exception: d = 0.0
        try: w = float(row.get('Wait Time(us)') or 0)
        except Exception: w = 0.0
        tot_dur += d; tot_wait += w
        e = by[row.get('Name') or '']; e[0] += 1; e[1] += d; e[2] += w
print("rows=%d  total_duration_us=%.0f  total_wait_us=%.0f" % (n, tot_dur, tot_wait))
print()
print("---- top 12 by Duration ----")
for name, (c, d, w) in sorted(by.items(), key=lambda x: -x[1][1])[:12]:
    print("  %6.2f%%  dur=%10.0f wait=%9.0f calls=%5d  %s" % (d/tot_dur*100, d, w, c, name[:64]))
print()
def agg(label, keys):
    c = sum(by[k][0] for k in keys); d = sum(by[k][1] for k in keys); w = sum(by[k][2] for k in keys)
    print("  %-26s calls=%5d  dur=%10.0f (%5.2f%%)  wait=%10.0f (%5.2f%%)"
          % (label, c, d, d/tot_dur*100, w, w/tot_wait*100 if tot_wait else 0))
    for k in sorted(keys, key=lambda k: -by[k][1])[:6]:
        print("        - %-46s dur=%9.0f wait=%9.0f calls=%d" % (k[:46], by[k][1], by[k][2], by[k][0]))
print("---- 目标算子族 ----")
agg("LpNorm 族（grad norm）", [k for k in by if 'LpNorm' in k or 'lpnorm' in k.lower()])
agg("AllReduce/Hccl 通信", [k for k in by if 'AllReduce' in k or 'allreduce' in k.lower() or 'Hccl' in k])
agg("AdamW（对照：确实在算）", [k for k in by if 'AdamW' in k or 'ApplyAdam' in k])
agg("MatMul 族（对照）", [k for k in by if 'MatMul' in k])
PY
echo "ANALYZE62_DONE"
