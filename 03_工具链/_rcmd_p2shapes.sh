#!/bin/bash
# P2 第二步：短窗口 + record-shapes（只为补 shape/dtype/stride；**与计时分开跑**）
set -u
CFG=/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml
TS=$(date +%Y%m%d_%H%M%S)
OUT=/root/ops/p2shapes_$TS
echo "P2SHAPES_ROOT=$OUT"
cd /root/qwen35-ascend-migrator
nohup python3 scripts/56_profile_run.py \
  --skill /root/qwen35-ascend-migrator --config "$CFG" \
  --steps 25 --prof-start 15 --prof-end 20 \
  --record-shapes --with-stack \
  --out "$OUT" --mind /root/MindSpeed-MM --timeout 1500 \
  > "$OUT.launch.log" 2>&1 &
echo "LAUNCHED_P2SHAPES pid=$!"
sleep 20
tail -n 6 "$OUT.launch.log" 2>/dev/null || echo "(还没有日志)"
echo "P2SHAPES_LAUNCH_DONE"
