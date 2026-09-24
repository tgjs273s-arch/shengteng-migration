#!/bin/bash
# P1：复用已有检查点，跳过 PHASE1（不重跑训练），只跑两次回放 + 比较
set -u
ROOT=$(ls -dt /root/ops/nzreplay_*/ 2>/dev/null | head -1); ROOT=${ROOT%/}
echo "REUSE_ROOT=$ROOT"
# 校验该批次确有 iter_0000030 与 metadata
ls -d "$ROOT"/ckpt/iter_0000030 >/dev/null 2>&1 || { echo "ABORT 没有 iter_0000030"; exit 1; }
test -f "$ROOT"/ckpt/iter_0000030/.metadata || { echo "ABORT 缺 .metadata"; exit 1; }
TS=$(date +%Y%m%d_%H%M%S)
LOG=/root/ops/_nzreplay_reuse_${TS}.launch.log
echo "LAUNCH_LOG=$LOG"
nohup python3 /root/ops/_nzreplay_run.py --reuse-root "$ROOT" > "$LOG" 2>&1 &
echo "LAUNCHED_REUSE pid=$!"
sleep 30
echo "=== 30 秒后 ==="
tail -n 14 "$LOG" 2>/dev/null || echo "(还没有日志)"
echo "NZ_REUSE_LAUNCH_DONE"
