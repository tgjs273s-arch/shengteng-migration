#!/bin/bash
# P2 候选卡取证：**复用现有 profile**，不重新采样
set -u
ROOT=$(ls -dt /root/ops/prof62_*/ 2>/dev/null | head -1); ROOT=${ROOT%/}
AP=$(find "$ROOT/prof" -type d -name "ASCEND_PROFILER_OUTPUT" | head -1)
echo "AP=$AP"
echo "=== api_statistic.csv 全量（torch 级 API，最接近源码层） ==="
cat "$AP/api_statistic.csv" 2>/dev/null | head -45
echo
echo "=== operator_details.csv 表头 ==="
head -1 "$AP/operator_details.csv" 2>/dev/null
echo "行数: $(wc -l < "$AP/operator_details.csv" 2>/dev/null)"
echo "P2_RECON_DONE"
