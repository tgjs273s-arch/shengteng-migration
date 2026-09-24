#!/bin/bash
# ③ 侦察 2：检查点格式与优化器状态可行性
set -u
TR=/root/MindSpeed-MM/mindspeed_mm/fsdp/train/trainer.py
echo "=== trainer.py L425-480（checkpointer 组装 + hf 守卫） ==="
sed -n '425,480p' $TR
echo
echo "=== save_format / load_format 的取值来源 ==="
grep -rn "save_format\|load_format" /root/MindSpeed-MM/mindspeed_mm/fsdp/params/*.py 2>/dev/null | head -12
echo
echo "=== 已有检查点里到底有什么（以 AB_AVSB 的 00_A 为例） ==="
find /root/ops/AB_AVSB_20260921/00_A/checkpoint -maxdepth 2 | head -20
echo "--- 是否含优化器状态文件（optim / *.distcp / dcp） ---"
find /root/ops/AB_AVSB_20260921/00_A/checkpoint -iname "*optim*" -o -iname "*.distcp" 2>/dev/null | head -10
echo
echo "=== dp2_base 检查点 ==="
find /root/ops/dp2_base_20260921_113921/checkpoint -maxdepth 2 2>/dev/null | head -15
echo
echo "=== 基线权重目录形态 ==="
ls /root/Qwen3.5-0.8B-dcp 2>/dev/null | head -8
echo "RECON63B_DONE"
