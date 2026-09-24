#!/bin/bash
# ============================================================
# 最终验收训练：Qwen3.5-0.8B · 1× 昇腾 910C 双 die · 官方几何 · 100 步
# 用途：复现精度分析报告 / 性能测试报告中的全部数据
# ============================================================
set -e

MSMM_DIR=${MSMM_DIR:-/root/MindSpeed-MM}
CONFIG=${CONFIG:-$MSMM_DIR/examples/qwen3_5/qwen3_5_0_8B_optimized.yaml}
LOG=${LOG:-/root/a3_train_100steps.log}
PORT=${PORT:-6111}

# ---- CANN 环境（缺失会导致 libhccl.so 找不到）----
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

# ---- 必需环境变量（缺 NON_MEGATRON 会报 No module named 'megatron'）----
export NON_MEGATRON=true
export TASK_QUEUE_ENABLE=2          # 实测 =1 会导致周期慢步回归
export ASCEND_LAUNCH_BLOCKING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/root/triton_cache
mkdir -p "$TRITON_CACHE_DIR"

cd "$MSMM_DIR"
echo "=== 开始训练 $(date) ==="
echo "配置: $CONFIG"
echo "日志: $LOG（完整日志，从程序启动开始，请勿裁剪）"

# 双 die 并行 = 官方基线拓扑（NPUS_PER_NODE=2）
timeout 5400 torchrun --nproc_per_node 2 --nnodes 1 --node_rank 0 \
  --master_addr localhost --master_port "$PORT" \
  mindspeed_mm/fsdp/train/trainer.py "$CONFIG" 2>&1 | tee "$LOG"

echo "=== 训练结束 $(date) ==="
echo "迭代行数: $(grep -ac 'iteration' "$LOG")"
echo "预期: 100 步完成；第 1 步 loss = 1.924621"
