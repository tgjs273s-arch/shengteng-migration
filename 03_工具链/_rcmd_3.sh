#!/bin/bash
# 远端调查③：取精确 API 面（只读）。目的：找到"每步 batch"最靠近的合法切入点
set -eu
cd /root/MindSpeed-MM
echo "=== trainer.py 数据构建点 ==="
grep -nE 'dataset|DataLoader|dataloader|collate|build_|get_batch|next\(' mindspeed_mm/fsdp/train/trainer.py | head -30 || true
echo "=== dataloader/dataloader.py 顶层符号 ==="
grep -nE '^class |^def |    def ' mindspeed_mm/data/dataloader/dataloader.py | head -40 || true
echo "=== datasets/__init__.py 顶层符号 ==="
grep -nE '^class |^def |^from |^import ' mindspeed_mm/data/datasets/__init__.py | head -30 || true
echo "=== data_collator.py 顶层符号 ==="
grep -nE '^class |^def |    def ' mindspeed_mm/data/dataloader/data_collator.py | head -25 || true
echo "DONE3"
