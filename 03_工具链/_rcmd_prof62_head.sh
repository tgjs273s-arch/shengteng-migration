#!/bin/bash
# §4-2 归因分析：检查 prof 产物结构 + kernel_details.csv 表头（找 stack 列）
set -u
OUT=$(ls -dt /root/ops/prof62_*/ 2>/dev/null | head -1); OUT=${OUT%/}
echo "PROF_DIR=$OUT/prof"
find "$OUT/prof" -maxdepth 2 -type f 2>/dev/null | head -20
echo "=== 大小 ==="
du -sh "$OUT/prof" 2>/dev/null
echo "=== kernel_details.csv 表头 ==="
KD=$(find "$OUT/prof" -name "kernel_details.csv" 2>/dev/null | head -1)
echo "KD=$KD"
if [ -n "$KD" ]; then
  head -1 "$KD"
  echo "行数: $(wc -l < "$KD")"
fi
echo "=== step_trace_time.csv ==="
ST=$(find "$OUT/prof" -name "step_trace_time.csv" 2>/dev/null | head -1)
echo "ST=$ST"
[ -n "$ST" ] && head -2 "$ST"
echo "ANALYZE_HEAD_DONE"
