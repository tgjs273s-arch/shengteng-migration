#!/bin/bash
# 收尾复核：确认远端框架仍处于"已还原"状态，并列出本轮产物
set -u
echo "=== 框架 md5（必须等于交付记录的原值） ==="
md5sum /root/MindSpeed-MM/mindspeed_mm/fsdp/optimizer/clip_grad_norm.py \
       /root/MindSpeed-MM/mindspeed_mm/fsdp/train/train_engine.py
echo "期望: 207f9cbad3da6adba9cd19a00b016b88  clip_grad_norm.py"
echo "期望: e0e8879ac9c94b9ece48bf18584d3166  train_engine.py"
echo "=== 注入残留全树扫描 ==="
echo "残留文件数=$(grep -rl '§4-2 INJECT' /root/MindSpeed-MM 2>/dev/null | wc -l)"
echo "=== 本轮产物 ==="
ls -d /root/ops/ab62_* /root/ops/prof62_* /root/ops/fwbackup_ab62 2>/dev/null
echo "=== 备份文件仍在（可追溯） ==="
ls -la /root/ops/fwbackup_ab62/ 2>/dev/null
echo "=== 磁盘 ==="
df -h /root | tail -1
echo "FINAL_CHECK_DONE"
