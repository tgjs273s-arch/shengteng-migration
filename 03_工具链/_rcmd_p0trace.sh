#!/bin/bash
# P0：KeyError:'optimizer' 的确切抛出点 + 谁构造 state
set -u
R=/root/ops/nzreplay_20260922_150729
echo "=== a/run.log 里 rank0 的完整 traceback（L515-575） ==="
sed -n '515,575p' "$R/a/run.log"
echo
echo "=== save 主路径 L170-205（看 save_state 怎么组装） ==="
sed -n '170,205p' /root/MindSpeed-MM/mindspeed_mm/fsdp/checkpoint/dcp_checkpointer.py
echo
echo "=== 谁构造传给 checkpointer 的 state（含 optimizer 键） ==="
grep -rn "\"optimizer\"\|'optimizer'\|save_checkpointer\|load_checkpointer\|def save_checkpoint\|def load_checkpoint" \
  /root/MindSpeed-MM/mindspeed_mm/fsdp/ 2>/dev/null | grep -v "dcp_checkpointer.py" | head -20
echo "P0_TRACE_DONE"
