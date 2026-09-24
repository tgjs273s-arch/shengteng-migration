#!/bin/bash
# 远端调查⑤：trainer.py 的入口与 Trainer 构造（只读）——决定包装入口怎么写
set -eu
cd /root/MindSpeed-MM
echo "=== 入口/构造相关行 ==="
grep -nE '__main__|def main|parse_args|get_args|initialize|Trainer\(|dataloader_provider|model_provider' mindspeed_mm/fsdp/train/trainer.py | head -25 || true
echo "=== __main__ 区段（-A 40）==="
grep -n -A 40 '__main__' mindspeed_mm/fsdp/train/trainer.py | head -55 || true
echo "DONE5"
