#!/bin/bash
# 远端工作㉟：**分离确定性机制**——`use_deter_comp=true` 同时设了两个环境变量，
#   本实验分别只设其中一个，看哪一个能恢复「同臂重复性」：
#     v1 臂：HCCL_DETERMINISTIC=True（集合通信算子级确定性），不动 matmul K-shift
#     v2 臂：CLOSE_MATMUL_K_SHIFT=1（关掉 matmul K 轴 shift），不动 HCCL
#   每臂 2 轮（同臂配对才能判重复性）；100 步；无 profiling。
#   基线（两个都不设）已有：/root/ops/batch4_20260922_122428/{a_r1,a_r2}（今日同批次）
set -eu
CFG=/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml
BASE=/root/ops/detiso_$(date +%Y%m%d_%H%M%S)
mkdir -p "$BASE"
echo "DETISO_BASE=$BASE"
echo "=== 确认这两个变量在框架里的设置点（只读）==="
grep -rn 'HCCL_DETERMINISTIC\|CLOSE_MATMUL_K_SHIFT' /root/MindSpeed/mindspeed/fsdp/utils/random.py 2>/dev/null | head -4

cat > "$BASE/all.sh" <<EOF
set -u
cd /root/qwen35-ascend-migrator
for arm in v1 v2; do
  for r in 1 2; do
    D="$BASE/\${arm}_r\${r}"; mkdir -p "\$D"
    EXTRA=""
    if [ "\$arm" = "v1" ]; then EXTRA="export HCCL_DETERMINISTIC=True"; fi
    if [ "\$arm" = "v2" ]; then EXTRA="export CLOSE_MATMUL_K_SHIFT=1"; fi
    cat > "\$D/launch.sh" <<EOS
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:\\\$LD_LIBRARY_PATH
export NON_MEGATRON=true
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/root/triton_cache
\$EXTRA
echo "DETISO_ENV arm=\$arm EXTRA=\$EXTRA HCCL_DETERMINISTIC=\\\${HCCL_DETERMINISTIC:-unset} CLOSE_MATMUL_K_SHIFT=\\\${CLOSE_MATMUL_K_SHIFT:-unset}"
cd /root/MindSpeed-MM && PYTHONPATH=/root/MindSpeed-MM:\\\${PYTHONPATH:-} timeout 1200 torchrun \\
  --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6130 \\
  mindspeed_mm/fsdp/train/trainer.py $CFG > "$D/train.log" 2>&1
echo "train_rc=\\\$?" > "$D/rc.txt"
EOS
    echo "RUN_START arm=\$arm r=\$r \$(date +%H:%M:%S)"
    bash "\$D/launch.sh" > "\$D/launch.out" 2>&1
    echo "RUN_DONE arm=\$arm r=\$r rc=\$(cat \$D/rc.txt 2>/dev/null || echo NA)"
  done
done
echo "DETISO_ALL_DONE" > "$BASE/all.done"
EOF
nohup bash "$BASE/all.sh" > "$BASE/all.out" 2>&1 &
echo "LAUNCHED_DETISO pid=$!"
sleep 20
tail -n 4 "$BASE/all.out" 2>/dev/null || true
echo "DETISO_LAUNCH_DONE"
