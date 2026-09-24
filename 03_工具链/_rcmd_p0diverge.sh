#!/bin/bash
# P0：load() 主路径 与 broadcast 路径的分歧（判断 load_rank0_and_broadcast 是否为可用路径）
set -u
DC=/root/MindSpeed-MM/mindspeed_mm/fsdp/checkpoint/dcp_checkpointer.py
BU=/root/MindSpeed-MM/mindspeed_mm/fsdp/checkpoint/broadcast_utils.py
echo "=== dcp_checkpointer.load() L236-272 ==="
sed -n '236,272p' "$DC"
echo
echo "=== broadcast_utils.py L120-175（OPTIMIZER 取用处） ==="
sed -n '120,175p' "$BU"
echo
echo "=== 该函数是否只在 rank0 读、以及是否处理 optimizer ==="
grep -nE "def rank0_load_and_broadcast_dcp_weights|OPTIMIZER|MODEL =|def " "$BU" | head -20
echo "P0_DIVERGE_DONE"
