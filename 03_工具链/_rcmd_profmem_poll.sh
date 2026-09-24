#!/bin/bash
# 远端工作㉖：轮询带显存的 profile；统计 HBM 曲线（是否跨步增长）
set -u
B=$(ls -1dt /root/ops/profmem3_* 2>/dev/null | head -1)
OUT=$B/run
echo "BASE=$B"
for i in $(seq 1 30); do
  [ -f "$OUT/result.json" ] && break
  grep -qE 'P2PROF_FAIL|FATAL' "$B/launch.log" 2>/dev/null && break
  sleep 3
done
echo "=== 完成判定 ==="
[ -f "$OUT/result.json" ] && echo "RESULT_JSON 存在" || echo "尚无 result.json"
grep -oE 'iteration +[0-9]+/ *100' "$OUT/train.log" 2>/dev/null | tail -1 || true
grep -E 'P2PROF_' "$B/launch.log" 2>/dev/null | tail -3 || true
echo "=== HBM 曲线统计（chip0/chip1，单位 MB）==="
python3 - "$B/hbm.csv" <<'PY'
import csv, io, sys
rows = [r for r in csv.DictReader(io.open(sys.argv[1], encoding="utf-8")) if r.get("chip0_hbm_mb")]
if not rows:
    print("  （无采样）"); raise SystemExit
def col(k):
    return [int(r[k]) for r in rows if r.get(k, "").isdigit()]
for k in ("chip0_hbm_mb", "chip1_hbm_mb"):
    v = col(k)
    if not v: continue
    print("  %s: n=%d 首=%d 末=%d 最小=%d 最大=%d 变化=%+d MB" % (k, len(v), v[0], v[-1], min(v), max(v), v[-1]-v[0]))
t = [int(r["t_epoch"]) for r in rows if r.get("t_epoch", "").isdigit()]
if t: print("  采样时长 = %d 秒" % (t[-1] - t[0]))
PY
echo "=== 显存相关产物 ==="
find "$OUT/prof" -maxdepth 2 -name '*memory*' 2>/dev/null | head -6 || echo "（无 memory 文件）"
echo "PROFMEM_POLL_DONE"
