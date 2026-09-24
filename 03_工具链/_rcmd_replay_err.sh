#!/bin/bash
# 远端工作⑰：取失败原因（回放入口跑挂，看 traceback 原文）
set -u
R=$(ls -1d /root/ops/replay_* | tail -1)
echo "=== a 日志里第一次 Traceback 前后 30 行 ==="
awk '/Traceback \(most recent call last\)/{f=1} f{print; n++} n>30{exit}' "$R/a/run.log" | cut -c1-200 | head -34
echo "=== 若无 traceback，看 ERROR/Error 行 ==="
grep -nE 'Error|error|Exception|FAIL' "$R/a/run.log" | head -12 | cut -c1-200
echo "REPLAY_ERR_READ_DONE"
