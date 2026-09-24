#!/bin/bash
# 远端工作⑪：等全窗口基线结束，再起**负向对照运行**（同一配置，仅数据集换成"改过一个真实字段"的副本）
set -u
R=$(ls -1d /root/ops/fp51full_* | tail -1)
for i in $(seq 1 30); do
  [ -f "$R/rc.txt" ] && break
  sleep 3
done
echo "基线 rc: $(cat $R/rc.txt 2>/dev/null || echo 未结束)"
echo "基线记录 yield 数 = $(grep -c BATCHFP_MICRO $R/fp.log 2>/dev/null || echo 0)"

TS=$(date +%Y%m%d_%H%M%S)
NROOT=/root/ops/fp51neg_$TS
mkdir -p "$NROOT"
cat > "$NROOT/launch.sh" <<EOF
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:\$LD_LIBRARY_PATH
export NON_MEGATRON=true
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/root/triton_cache
export BATCHFP=1
export BATCHFP_MAX=100
export BATCHFP_LOG=$NROOT/fp.log
cd /root/MindSpeed-MM && PYTHONPATH=/root/MindSpeed-MM:\${PYTHONPATH:-} timeout 1200 torchrun \\
  --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6116 \\
  /root/qwen35-ascend-migrator/scripts/51_train_fp.py /root/ops/fp_neg/cfg_perturbed.yaml > $NROOT/train.log 2>&1
echo "train_rc=\$?" > $NROOT/rc.txt
EOF
nohup bash "$NROOT/launch.sh" > "$NROOT/launch.out" 2>&1 &
echo "NEG_ROOT=$NROOT LAUNCHED pid=$!"
echo "FP51_NEG_START_DONE"
