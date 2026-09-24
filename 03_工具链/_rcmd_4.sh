#!/bin/bash
# 远端调查④：dataloader_provider 的注入路径 + 构建器签名（只读，决定"逐步 batch 指纹"能否落地）
set -eu
cd /root/MindSpeed-MM
echo "=== 谁在用 dataloader_provider ==="
grep -rn 'dataloader_provider' mindspeed_mm/ | head -20 || true
echo "=== trainer.py 339-400（get_dataloader 本体）==="
sed -n '339,400p' mindspeed_mm/fsdp/train/trainer.py
echo "=== prepare_base_dataloader 签名与主体头 ==="
sed -n '110,150p' mindspeed_mm/data/dataloader/dataloader.py
echo "DONE4"
