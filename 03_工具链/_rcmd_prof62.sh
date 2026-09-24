#!/bin/bash
# §4-2 侦察：带调用栈的 profile —— 把算子发射归因到 Python 调用点
# 目的：读出 clip_grad_norm / average_losses_across_data_parallel_group 两处的占比
# 注意：本脚本**不改任何框架代码**（候选只做归因，不改文件）
set -u
CFG=/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml
TS=$(date +%Y%m%d_%H%M%S)
OUT=/root/ops/prof62_$TS
echo "PROF62_ROOT=$OUT"
echo "=== 配置存在性 ==="
ls -la "$CFG" || echo "MISSING_CONFIG"
echo "=== 框架文件哈希（开工前，便于事后证明"没改过"） ==="
md5sum /root/MindSpeed-MM/mindspeed_mm/fsdp/optimizer/clip_grad_norm.py \
       /root/MindSpeed-MM/mindspeed_mm/fsdp/train/train_engine.py
cd /root/qwen35-ascend-migrator
nohup python3 scripts/56_profile_run.py \
  --skill /root/qwen35-ascend-migrator \
  --config "$CFG" \
  --steps 100 --prof-start 50 --prof-end 55 \
  --with-stack \
  --out "$OUT" --mind /root/MindSpeed-MM --timeout 1500 \
  > "$OUT.launch.log" 2>&1 &
echo "LAUNCHED_PROF62 pid=$!"
sleep 25
echo "=== 25 秒后 launch.log ==="
tail -n 10 "$OUT.launch.log" 2>/dev/null || echo "(还没有日志)"
echo "PROF62_LAUNCH_DONE"
