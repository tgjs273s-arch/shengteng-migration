#!/bin/bash
# 读 wy_fast.py 的 kernel 本体，判定 grid=24 与 100 个工作项的关系
set -u
F=/root/MindSpeed-MM/mindspeed_mm/fsdp/ops/gdn/triton/wy_fast.py
echo "=== L14-60（heuristics + 签名） ==="
sed -n '14,60p' "$F"
echo
echo "=== L60-100（body 开头：看有没有对 NT/B 的循环） ==="
sed -n '60,100p' "$F"
echo
echo "=== 关键结构扫描 ==="
grep -nE "for .* in range|i_t|i_b|pid|tl.program_id|num_warps|num_stages|cv_kernel|grid" "$F" | head -30
echo "WY_KERNEL_READ_DONE"
