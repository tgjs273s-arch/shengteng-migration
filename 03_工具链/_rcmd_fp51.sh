#!/bin/bash
# 远端工作⑤：跑**逐步 batch 指纹**（第一轮）
#
# 纪律（按 Codex 的要求）：
#   · **保持原 100 步 + 学习率协议**（不改 train_iters、不派生配置）；
#   · 只在前几个微批启用记录（BATCHFP_MAX=3）——记录本身会改变时序，故只用于取证；
#   · 启动命令**照 50_train.py 的 dry-run 原文**，只把入口脚本换成 51_train_fp.py
#     （不发明新的启动约定：同一套 env 导出、同一 PYTHONPATH、同一套 torchrun 参数）；
#   · 训练 **nohup 后台跑**（单条远端命令上限 110 s，训练约 140 s）⇒ 立刻返回，之后轮询日志。
set -eu
TS=$(date +%Y%m%d_%H%M%S)
ROOT=/root/ops/fp51_$TS
mkdir -p "$ROOT"
CFG=/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml
echo "FP51_ROOT=$ROOT"
echo "配置 train_iters=$(python3 -c "import yaml;print(yaml.safe_load(open('$CFG'))['training']['train_iters'])")"
echo "配置几何=$(python3 -c "import yaml;d=yaml.safe_load(open('$CFG'));t=d['training'];print('mbs=%s gas=%s world=%s' % (t['micro_batch_size'], t['gradient_accumulation_steps'], d['parallel']['data_parallel_size']))")"

cat > "$ROOT/launch.sh" <<EOF
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:\$LD_LIBRARY_PATH
export NON_MEGATRON=true
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/root/triton_cache
# ★ 指纹开关（只记前 3 个微批；日志按 rank 分文件，避免多进程互相撕裂）
export BATCHFP=1
export BATCHFP_MAX=3
export BATCHFP_LOG=$ROOT/fp.log
cd /root/MindSpeed-MM && PYTHONPATH=/root/MindSpeed-MM:\${PYTHONPATH:-} timeout 900 torchrun \\
  --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6113 \\
  /root/qwen35-ascend-migrator/scripts/51_train_fp.py $CFG > $ROOT/train.log 2>&1
rc=\$?
echo "train_rc=\$rc" > $ROOT/rc.txt
echo "FP51_DONE rc=\$rc"
EOF
chmod +x "$ROOT/launch.sh"
nohup bash "$ROOT/launch.sh" > "$ROOT/launch.out" 2>&1 &
echo "LAUNCHED pid=$! （后台；用 tail 轮询 $ROOT/train.log）"
sleep 20
echo "=== 20 秒后 train.log 头部（看是否正常起来）==="
tail -n 12 "$ROOT/train.log" 2>/dev/null || echo "（还没有日志）"
echo "FP51_START_DONE"
