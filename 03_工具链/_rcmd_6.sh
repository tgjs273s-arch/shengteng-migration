#!/bin/bash
# 远端调查⑥：trainer.py 的 import 头（只读）——把包装入口的 import 钉死，不留猜的余地
set -eu
cd /root/MindSpeed-MM
echo "=== trainer.py 1-42 行 ==="
sed -n '1,42p' mindspeed_mm/fsdp/train/trainer.py
echo "=== ConfigManager / Arguments 的定义位置 ==="
grep -rn 'class ConfigManager' mindspeed_mm/ | head -5 || true
grep -rn 'class Arguments' mindspeed_mm/ | head -5 || true
echo "DONE6"
