#!/bin/bash
set -u
B=$(ls -1dt /root/ops/detiso3_* 2>/dev/null | head -1)
[ -n "$B" ] || { echo "DETISO3_NO_DIR"; exit 0; }
echo "DETISO3_BASE=$B"
if [ -f "$B/all.done" ]; then echo "DETISO3_ALL_DONE"; else
  echo "DETISO3_DONE_COUNT=$(grep -cE 'RUN_DONE' "$B/all.out" 2>/dev/null || echo 0)/4"
fi
