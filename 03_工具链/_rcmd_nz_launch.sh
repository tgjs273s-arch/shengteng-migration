#!/bin/bash
# ③ 非零学习率回放：后台启动 Python 编排
set -u
cd /root/ops
TS=$(date +%Y%m%d_%H%M%S)
LOG=/root/ops/_nzreplay_${TS}.launch.log
echo "LAUNCH_LOG=$LOG"
nohup python3 /root/ops/_nzreplay_run.py > "$LOG" 2>&1 &
echo "LAUNCHED_NZ pid=$!"
sleep 30
echo "=== 30 秒后 ==="
tail -n 12 "$LOG" 2>/dev/null || echo "(还没有日志)"
echo "NZ_LAUNCH_DONE"
