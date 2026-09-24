#!/bin/bash
# 远端调查②：mindspeed_mm/fsdp 训练入口的数据构建点（**只读**）
set -eu
cd /root/MindSpeed-MM
echo "--- trainer.py 数据相关行 ---"
grep -nE 'dataset|DataLoader|dataloader|collate|build_|get_batch|next\(' mindspeed_mm/fsdp/train/trainer.py | head -40 || true
echo "--- fsdp 目录结构 ---"
find mindspeed_mm/fsdp -maxdepth 2 -name '*.py' | head -30 || true
echo "--- data 相关模块（全包 maxdepth 4）---"
find mindspeed_mm -maxdepth 4 -name '*.py' | grep -iE 'data|dataset|loader|collate' | head -30 || true
echo "DONE2"
