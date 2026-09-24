#!/bin/bash
# P0：dcp 存取两侧的 state dict 键名为什么不对称
set -u
F=/root/MindSpeed-MM/mindspeed_mm/fsdp/checkpoint/dcp_checkpointer.py
echo "host=$(cat /proc/sys/kernel/hostname)"
echo "=== 类与键 ==="
grep -nE "^class |def __init__|def state_dict|def load_state_dict|_OPTIM|optimizer|extra_state|def save|def load" $F | head -40
echo
echo "=== L100-160（包装类 state_dict/load_state_dict） ==="
sed -n '100,160p' $F
echo
echo "=== L236-275（load 主路径） ==="
sed -n '236,275p' $F
echo
echo "=== 谁调用 _load_extra_state / 组合 state dict ==="
grep -rn "load_state_dict\|state_dict()\|extra_state" /root/MindSpeed-MM/mindspeed_mm/fsdp/train/train_engine.py 2>/dev/null | head -20
echo "P0_SRC_DONE"
