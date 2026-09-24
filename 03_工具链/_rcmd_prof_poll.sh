#!/bin/bash
# 远端工作㉑：轮询 profile 结果（每次最多等 ~95 秒；结果以 result.json / P2PROF_OK 为准）
set -u
L=$(ls -1t /root/ops/prof_*.launch.log 2>/dev/null | head -1)
OUT=${L%.launch.log}
echo "PROF_OUT=$OUT"
for i in $(seq 1 30); do
  if [ -f "$OUT/result.json" ]; then break; fi
  if grep -qE 'P2PROF_FAIL|FATAL|Traceback' "$L" 2>/dev/null; then break; fi
  sleep 3
done
echo "=== 是否完成 ==="
[ -f "$OUT/result.json" ] && echo "RESULT_JSON 存在" || echo "尚无 result.json"
echo "=== launch.log 关键行 ==="
grep -E 'PROF_|P2PROF_' "$L" 2>/dev/null | tail -12 | cut -c1-180
echo "=== launch.log 尾部（未完成时看进度）==="
tail -n 5 "$L" 2>/dev/null | cut -c1-180
echo "=== 产物目录 ==="
ls -1 "$OUT" 2>/dev/null | head -12
echo "PROF_POLL_DONE"
