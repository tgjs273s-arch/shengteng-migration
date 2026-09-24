#!/bin/bash
# 确认远端 50_train.py 的入参（不猜）
set -u
cd /root/qwen35-ascend-migrator
echo "=== 50_train.py --help 片段 ==="
python3 scripts/50_train.py --help 2>&1 | head -40
echo "PROBE50_DONE"
