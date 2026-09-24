#!/bin/bash
# 四臂批次探测（给本地后台作业用；输出保持极小）
set -u
B=$(ls -1dt /root/ops/batch4_* 2>/dev/null | head -1)
[ -n "$B" ] || { echo "BATCH4_NO_DIR"; exit 0; }
echo "BATCH4_BASE=$B"
if [ -f "$B/all.done" ]; then
  echo "BATCH4_ALL_DONE"
else
  echo "BATCH4_DONE_COUNT=$(grep -cE 'RUN_DONE' "$B/all.out" 2>/dev/null || echo 0)/8"
fi
