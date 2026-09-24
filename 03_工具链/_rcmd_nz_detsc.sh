#!/bin/bash
# A/A 确定性诊断对照：复用同一检查点，两次回放**都开**确定性模式
set -u
ROOT=$(ls -dt /root/ops/nzreplay_*/ 2>/dev/null | head -1); ROOT=${ROOT%/}
echo "REUSE_ROOT=$ROOT"
ls -d "$ROOT"/ckpt/iter_0000030 >/dev/null 2>&1 || { echo "ABORT 没有 iter_0000030"; exit 1; }
test -f "$ROOT"/ckpt/iter_0000030/.metadata || { echo "ABORT 缺 .metadata"; exit 1; }
echo "=== 清掉可能残留的 DETSC marker（避免旧残留当自证） ==="
rm -f /tmp/detsc62_active_rank_0 /tmp/detsc62_active_rank_1 /tmp/detsc62_active_rank_x
echo "markers now: $(ls /tmp/detsc62_active_rank_* 2>/dev/null | wc -l)"
echo "=== 注入件在位性 ==="
ls -la /root/ops/detsc62/sitecustomize.py || { echo "ABORT 缺 sitecustomize.py"; exit 1; }
TS=$(date +%Y%m%d_%H%M%S)
LOG=/root/ops/_nzreplay_detsc_${TS}.launch.log
echo "LAUNCH_LOG=$LOG"
nohup python3 /root/ops/_nzreplay_run.py --reuse-root "$ROOT" --detsc on > "$LOG" 2>&1 &
echo "LAUNCHED_DETSC pid=$!"
sleep 25
echo "=== 25 秒后 ==="
tail -n 12 "$LOG" 2>/dev/null || echo "(还没有日志)"
echo "NZ_DETSC_LAUNCH_DONE"
