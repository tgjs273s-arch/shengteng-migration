#!/bin/bash
# P2 候选数值复核：打补丁 → 跑一次候选回放（确定性模式 ON，与既有 a_detsc 基线同模式）→ 比对 → 还原
set -u
ROOT=$(ls -dt /root/ops/nzreplay_*/ 2>/dev/null | head -1); ROOT=${ROOT%/}
echo "ROOT=$ROOT"
CFG="$ROOT/cfg2_replay_from_ckpt.yaml"
test -f "$CFG" || { echo "ABORT 缺 cfg2"; exit 1; }

echo "=== 1) 打补丁 ==="
python3 /root/ops/_p2_patch.py apply || { echo "ABORT 打补丁失败"; exit 1; }
md5sum /root/MindSpeed-MM/mindspeed_mm/fsdp/ops/gdn/triton/chunk_o.py

echo "=== 2) 候选回放（DETSC on，与 a_detsc 同模式） ==="
OUT="$ROOT/cand_detsc"; mkdir -p "$OUT"
rm -f /tmp/detsc62_active_rank_*
cat > "$ROOT/launch_cand.sh" <<EOS
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:\$LD_LIBRARY_PATH
export NON_MEGATRON=true
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/root/triton_cache
export PYTHONPATH=/root/ops/detsc62:/root/MindSpeed-MM:\${PYTHONPATH:-}
export REPLAY_OUT=$OUT
cd /root/MindSpeed-MM && timeout 1200 torchrun \
  --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6150 \
  /root/qwen35-ascend-migrator/scripts/52_replay.py $CFG > $OUT/run.log 2>&1
echo "rc=\$?" > $OUT/rc.txt
EOS
bash "$ROOT/launch_cand.sh"
echo "cand rc=$(cat $OUT/rc.txt 2>/dev/null)"
echo "DETSC active 行数=$(grep -c 'DETSC active' $OUT/run.log 2>/dev/null || echo 0)"
grep -hE "REPLAY_PRECOND|REPLAY_C1|REPLAY_C3|REPLAY_C4|REPLAY_OPT_STATE" "$OUT/run.log" 2>/dev/null | sed 's/\[Rank [01] | Local Rank [01]\] //' | head -10

echo "=== 3) 比对：候选 vs 既有确定性基线 a_detsc ==="
cd /root/qwen35-ascend-migrator/scripts
python3 _replay_diff.py --a "$ROOT/a_detsc" --b "$OUT" > "$ROOT/diff_cand.txt" 2>&1
echo "diff rc=$?"
head -16 "$ROOT/diff_cand.txt"
grep -E "REPLAY_VERDICT|REPLAY_IDENTICAL|REPLAY_FAIL" "$ROOT/diff_cand.txt" || true

echo "=== 4) 还原框架 ==="
python3 /root/ops/_p2_patch.py restore
md5sum /root/MindSpeed-MM/mindspeed_mm/fsdp/ops/gdn/triton/chunk_o.py
echo "P2_CAND_DONE"
