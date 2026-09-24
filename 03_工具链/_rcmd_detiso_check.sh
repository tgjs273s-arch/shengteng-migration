#!/bin/bash
# 分离实验探测（输出极小）
set -u
B=$(ls -1dt /root/ops/detiso_* 2>/dev/null | head -1)
[ -n "$B" ] || { echo "DETISO_NO_DIR"; exit 0; }
echo "DETISO_BASE=$B"
if [ -f "$B/all.done" ]; then echo "DETISO_ALL_DONE"; else
  echo "DETISO_DONE_COUNT=$(grep -cE 'RUN_DONE' "$B/all.out" 2>/dev/null || echo 0)/4"
fi
