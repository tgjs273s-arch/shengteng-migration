#!/bin/bash
# 远端工作㊱：分离确定性机制（重写启动方式：**引号化 heredoc + 参数传 BASE**，不再嵌套转义）
set -eu
CFG=/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml
BASE=/root/ops/detiso_$(date +%Y%m%d_%H%M%S)
mkdir -p "$BASE"
echo "DETISO_BASE=$BASE"

cat > "$BASE/all.sh" <<'EOS'
#!/bin/bash
set -u
BASE="$1"
CFG=/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
export NON_MEGATRON=true
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/root/triton_cache
cd /root/MindSpeed-MM
for arm in v1 v2; do
  for r in 1 2; do
    D="$BASE/${arm}_r${r}"
    mkdir -p "$D"
    EXTRA_NEEDED=""
    if [ "$arm" = "v1" ]; then export HCCL_DETERMINISTIC=True; unset CLOSE_MATMUL_K_SHIFT; EXTRA_NEEDED="HCCL_DETERMINISTIC=True"; fi
    if [ "$arm" = "v2" ]; then unset HCCL_DETERMINISTIC; export CLOSE_MATMUL_K_SHIFT=1; EXTRA_NEEDED="CLOSE_MATMUL_K_SHIFT=1"; fi
    echo "RUN_START arm=$arm r=$r $EXTRA_NEEDED $(date +%H:%M:%S) HCCL=${HCCL_DETERMINISTIC:-unset} KSHIFT=${CLOSE_MATMUL_K_SHIFT:-unset}"
    PYTHONPATH=/root/MindSpeed-MM:${PYTHONPATH:-} timeout 1200 torchrun \
      --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6130 \
      mindspeed_mm/fsdp/train/trainer.py "$CFG" > "$D/train.log" 2>&1
    echo "train_rc=$?" > "$D/rc.txt"
    echo "RUN_DONE arm=$arm r=$r rc=$(cat $D/rc.txt)"
  done
done
echo "DETISO_ALL_DONE" > "$BASE/all.done"
EOS
chmod +x "$BASE/all.sh"
nohup bash "$BASE/all.sh" "$BASE" > "$BASE/all.out" 2>&1 &
echo "LAUNCHED_DETISO pid=$!"
sleep 25
echo "=== 25 秒后 ==="
tail -n 4 "$BASE/all.out" 2>/dev/null || true
echo "DETISO_LAUNCH_DONE"
