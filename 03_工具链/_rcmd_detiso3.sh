#!/bin/bash
# 远端工作㊲：只设 torch 层确定性开关（经 sitecustomize 于解释器启动时注入）+ 三者全设的正对照
#   v3：仅 torch.use_deterministic_algorithms(True)（不设两个 env）
#   v4：sitecustomize + HCCL_DETERMINISTIC=True + CLOSE_MATMUL_K_SHIFT=1（**正对照**，预期 0/100 超阈）
set -eu
BASE=/root/ops/detiso3_$(date +%Y%m%d_%H%M%S)
mkdir -p "$BASE/inject"
echo "DETISO3_BASE=$BASE"

cat > "$BASE/inject/sitecustomize.py" <<'PYEOF'
# 解释器启动时注入：**只**打开 PyTorch 分派层的确定性算法开关
import os
import sys
try:
    import torch
    torch.use_deterministic_algorithms(True)
    sys.stderr.write("DETSC active torch.use_deterministic_algorithms(True) pid=%d\n" % os.getpid())
    sys.stderr.flush()
except Exception as exc:                     # 注入失败必须**响亮**，不许静默
    sys.stderr.write("DETSC FAILED %s: %s\n" % (type(exc).__name__, exc))
    sys.stderr.flush()
PYEOF

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
export PYTHONPATH="$BASE/inject:/root/MindSpeed-MM:${PYTHONPATH:-}"
cd /root/MindSpeed-MM
for arm in v3 v4; do
  for r in 1 2; do
    D="$BASE/${arm}_r${r}"
    mkdir -p "$D"
    if [ "$arm" = "v3" ]; then unset HCCL_DETERMINISTIC; unset CLOSE_MATMUL_K_SHIFT; fi
    if [ "$arm" = "v4" ]; then export HCCL_DETERMINISTIC=True; export CLOSE_MATMUL_K_SHIFT=1; fi
    echo "RUN_START arm=$arm r=$r $(date +%H:%M:%S) HCCL=${HCCL_DETERMINISTIC:-unset} KSHIFT=${CLOSE_MATMUL_K_SHIFT:-unset} PYTHONPATH_head=${PYTHONPATH%%:*}"
    timeout 1200 torchrun --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost \
      --master_port 6140 mindspeed_mm/fsdp/train/trainer.py "$CFG" > "$D/train.log" 2>&1
    echo "train_rc=$?" > "$D/rc.txt"
    echo "RUN_DONE arm=$arm r=$r rc=$(cat $D/rc.txt) DETSC=$(grep -c 'DETSC active' $D/train.log || true)"
  done
done
echo "DETISO3_ALL_DONE" > "$BASE/all.done"
EOS
chmod +x "$BASE/all.sh"
nohup bash "$BASE/all.sh" "$BASE" > "$BASE/all.out" 2>&1 &
echo "LAUNCHED_DETISO3 pid=$!"
sleep 25
tail -n 4 "$BASE/all.out" 2>/dev/null || true
echo "DETISO3_LAUNCH_DONE"
