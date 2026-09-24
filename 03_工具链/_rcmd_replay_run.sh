#!/bin/bash
# 远端工作⑮：一次更新回放 —— 连跑两轮（同配置、固定完整状态），后台执行
set -u
CFG=/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml
TS=$(date +%Y%m%d_%H%M%S)
ROOT=/root/ops/replay_$TS
mkdir -p "$ROOT/a" "$ROOT/b"
echo "REPLAY_ROOT=$ROOT"

mk_launch () {   # $1=tag  $2=port  $3=outdir
cat > "$ROOT/launch_$1.sh" <<EOF
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:\$LD_LIBRARY_PATH
export NON_MEGATRON=true
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/root/triton_cache
export REPLAY_OUT=$3
cd /root/MindSpeed-MM && PYTHONPATH=/root/MindSpeed-MM:\${PYTHONPATH:-} timeout 900 torchrun \\
  --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port $2 \\
  /root/qwen35-ascend-migrator/scripts/52_replay.py $CFG > $3/run.log 2>&1
echo "rc=\$?" > $3/rc.txt
EOF
}

cat > "$ROOT/all.sh" <<EOF
set -u
bash "$ROOT/launch_a.sh"
bash "$ROOT/launch_b.sh"
echo "REPLAY_ALL_DONE" > "$ROOT/all.done"
EOF
mk_launch a 6120 "$ROOT/a"
mk_launch b 6121 "$ROOT/b"
nohup bash "$ROOT/all.sh" > "$ROOT/all.out" 2>&1 &
echo "LAUNCHED_ALL pid=$!"
sleep 30
echo "=== 30 秒后 a 的日志尾部（看是否正常起来）==="
tail -n 6 "$ROOT/a/run.log" 2>/dev/null || echo "（还没有日志）"
echo "REPLAY_LAUNCH_DONE"
