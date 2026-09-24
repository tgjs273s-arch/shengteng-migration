#!/bin/bash
# 捕获 prepare_wy_repr_bwd 的真实入参（单步回放 + 只读注入件）
set -u
ROOT=$(ls -dt /root/ops/nzreplay_*/ 2>/dev/null | head -1); ROOT=${ROOT%/}
CFG="$ROOT/cfg2_replay_from_ckpt.yaml"
test -f "$CFG" || { echo "ABORT 缺 cfg2 ($CFG)"; exit 1; }
mkdir -p /root/ops/argdump62
OUT="$ROOT/argdump_run"; mkdir -p "$OUT"
rm -f /root/ops/argdump62/../argdump_out/*.json 2>/dev/null || true
mkdir -p /root/ops/argdump_out && rm -f /root/ops/argdump_out/*.json
ls -la /root/ops/argdump62/sitecustomize.py || { echo "ABORT 缺注入件"; exit 1; }
cat > "$ROOT/launch_argdump.sh" <<EOS
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:\$LD_LIBRARY_PATH
export NON_MEGATRON=true
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/root/triton_cache
export ARGDUMP_OUT=/root/ops/argdump_out
export PYTHONPATH=/root/ops/argdump62:/root/MindSpeed-MM:\${PYTHONPATH:-}
export REPLAY_OUT=$OUT
cd /root/MindSpeed-MM && timeout 1200 torchrun \
  --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6160 \
  /root/qwen35-ascend-migrator/scripts/52_replay.py $CFG > $OUT/run.log 2>&1
echo "rc=\$?" > $OUT/rc.txt
EOS
bash "$ROOT/launch_argdump.sh"
echo "rc=$(cat $OUT/rc.txt 2>/dev/null)"
echo "=== ARGDUMP 自证 ==="
grep -c "ARGDUMP_PATCHED" "$OUT/run.log" 2>/dev/null || echo 0
grep -m2 -hE "ARGDUMP_(PATCHED|IMPORT_FAIL|NO_FUNC|WRITE_FAIL)" "$OUT/run.log" 2>/dev/null
echo "=== 落盘文件 ==="
ls -la /root/ops/argdump_out/ 2>/dev/null | head -8
for f in /root/ops/argdump_out/*.json; do
  [ -f "$f" ] || continue
  echo "--- $f ---"
  python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
print('fn=',d['fn'],'call=',d['call_index'])
for i,a in enumerate(d['args'][:6]):
    print('  arg%d'%i, {k:a[k] for k in ('py_type','shape','dtype','device','is_contiguous') if k in a})
for k,v in list(d['kwargs'].items())[:8]:
    print('  kw:%s'%k, {kk:v[kk] for kk in ('py_type','shape','dtype','value') if kk in v})
" "$f"
done
echo "ARGDUMP_DONE"
