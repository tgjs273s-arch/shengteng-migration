#!/bin/bash
# P2 端到端确认实验：后台发车（6 轮 = 基线3 + 候选3，预先固定区组）
set -u
cp -f /root/ops/fwbackup_p2/chunk_o.py.orig /root/ops/fwbackup_p2/chunk_o.py.orig.keep 2>/dev/null || true
ls -la /root/ops/fwbackup_p2/ || { echo "ABORT 无备份目录"; exit 1; }
test -f /root/ops/fwbackup_p2/chunk_o.py.orig || { echo "ABORT 无备份文件"; exit 1; }
TS=$(date +%Y%m%d_%H%M%S)
LOG=/root/ops/_p2ab_${TS}.launch.log
echo "LAUNCH_LOG=$LOG"
nohup python3 /root/ops/_p2ab_run.py > "$LOG" 2>&1 &
echo "LAUNCHED_P2AB pid=$!"
sleep 25
tail -n 10 "$LOG" 2>/dev/null || echo "(还没有日志)"
echo "P2AB_LAUNCH_DONE"
