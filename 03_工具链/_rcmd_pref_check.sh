#!/bin/bash
# 实验完成探测（给本地后台作业用；只输出一行标记，保持输出极小）
set -u
B=$(ls -1dt /root/ops/pref_* | head -1)
if [ -f "$B/all.done" ]; then
  echo "ALL_DONE_YES $B"
  grep -E 'RUN_DONE' "$B/all.out" 2>/dev/null | tail -6 || true
else
  N=$(grep -cE 'RUN_DONE' "$B/all.out" 2>/dev/null || echo 0)
  echo "ALL_DONE_NO $B 已完成=$N/6"
fi
