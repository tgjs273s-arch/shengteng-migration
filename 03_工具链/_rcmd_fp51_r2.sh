#!/bin/bash
# 远端工作⑦：第二轮相同运行的指纹（用于跨运行 diff）+ 查清 mock 数据集是什么
set -eu
CFG=/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml
TS=$(date +%Y%m%d_%H%M%S)
ROOT=/root/ops/fp51_$TS
mkdir -p "$ROOT"
echo "FP51_R2_ROOT=$ROOT"
echo "=== 配置里的数据集相关键 ==="
python3 - <<PY
import yaml, json
d = yaml.safe_load(open("$CFG"))
sp = d["data"]["dataset_param"]
print(json.dumps(sp, ensure_ascii=False, indent=1)[:1200])
PY
echo "=== 数据集文件 ==="
python3 - <<PY
import yaml, os, glob
d = yaml.safe_load(open("$CFG"))
sp = d["data"]["dataset_param"]
cands = []
def walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, str) and ("/" in v or v.endswith((".json", ".jsonl", ".parquet"))):
                cands.append(v)
            walk(v)
    elif isinstance(o, list):
        for x in o: walk(x)
walk(sp)
for c in cands[:8]:
    if os.path.exists(c):
        st = os.stat(c)
        print("EXISTS %-70s %d B" % (c, st.st_size))
    else:
        print("MISSING %s" % c)
PY
cat > "$ROOT/launch.sh" <<EOF
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:\$LD_LIBRARY_PATH
export NON_MEGATRON=true
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/root/triton_cache
export BATCHFP=1
export BATCHFP_MAX=3
export BATCHFP_LOG=$ROOT/fp.log
cd /root/MindSpeed-MM && PYTHONPATH=/root/MindSpeed-MM:\${PYTHONPATH:-} timeout 900 torchrun \\
  --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6114 \\
  /root/qwen35-ascend-migrator/scripts/51_train_fp.py $CFG > $ROOT/train.log 2>&1
echo "train_rc=\$?" > $ROOT/rc.txt
EOF
nohup bash "$ROOT/launch.sh" > "$ROOT/launch.out" 2>&1 &
echo "LAUNCHED_R2 pid=$!"
echo "FP51_R2_START_DONE"
