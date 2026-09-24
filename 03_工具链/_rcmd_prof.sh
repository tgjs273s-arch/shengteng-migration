#!/bin/bash
# 远端工作⑳：短 profile（保持 100 步协议；在 50-55 步稳定区采 trace；先不开调用栈）
set -u
CFG=/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml
TS=$(date +%Y%m%d_%H%M%S)
OUT=/root/ops/prof_$TS
echo "PROF_ROOT=$OUT"
cd /root/qwen35-ascend-migrator
nohup python3 scripts/56_profile_run.py \
  --skill /root/qwen35-ascend-migrator \
  --config "$CFG" \
  --steps 100 --prof-start 50 --prof-end 55 \
  --out "$OUT" --mind /root/MindSpeed-MM --timeout 1500 \
  > "$OUT.launch.log" 2>&1 &
echo "LAUNCHED_PROF pid=$!"
sleep 25
echo "=== 25 秒后 launch.log ==="
tail -n 8 "$OUT.launch.log" 2>/dev/null || echo "（还没有日志）"
echo "PROF_LAUNCH_DONE"
