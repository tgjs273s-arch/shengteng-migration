#!/bin/bash
# P0 现场核对：机器身份 / 进程 / 最新批次 / 失败原因（只读，不启动任何训练）
set -u
echo "=== 机器身份 ==="
hostname
echo "boot_id: $(cat /proc/sys/kernel/random/boot_id 2>/dev/null)"
npu-smi info 2>/dev/null | sed -n '1,4p'
echo "=== 是否有训练进程在跑（只读；不用 pkill -f，见坑 207） ==="
ps -eo pid,etime,cmd 2>/dev/null | grep -E "[t]orchrun|[5]0_train|[5]2_replay" | head -5 || echo "(无)"
echo "=== 最新批次 ==="
ls -dt /root/ops/nzreplay_*/ 2>/dev/null | head -3
R=$(ls -dt /root/ops/nzreplay_*/ 2>/dev/null | head -1); R=${R%/}
echo "LATEST=$R"
echo "--- all.done ---"; cat "$R/all.done" 2>/dev/null || echo "(无)"
echo "--- 进度行 ---"; L=$(ls -t /root/ops/_nzreplay_*.launch.log 2>/dev/null | head -1); echo "LAUNCH=$L"
grep -E "^###|NZREPLAY" "$L" 2>/dev/null | tail -14
echo "=== phase2 a/run.log 失败原因 ==="
grep -nE "Error|Traceback|raise |Exception|FAIL|assert" "$R/a/run.log" 2>/dev/null | tail -18
echo "--- 尾部 ---"
tail -14 "$R/a/run.log" 2>/dev/null
echo "P0_SITE_DONE"
