#!/bin/bash
# §4-2 A/B 收尾：还原框架并逐字节核对（改过别人的代码就必须给出"已还原"的证据）
set -u
cd /root/ops
python3 _ab62_restore.py
echo "AB62_RESTORE_RC=$?"
echo "=== 独立复核：与交付记录的原始 md5 对比 ==="
md5sum /root/MindSpeed-MM/mindspeed_mm/fsdp/optimizer/clip_grad_norm.py \
       /root/MindSpeed-MM/mindspeed_mm/fsdp/train/train_engine.py
echo "期望: 207f9cbad3da6adba9cd19a00b016b88  clip_grad_norm.py"
echo "期望: e0e8879ac9c94b9ece48bf18584d3166  train_engine.py"
echo "=== 全局扫描：MindSpeed-MM 里是否还有注入残留 ==="
grep -rl "§4-2 INJECT" /root/MindSpeed-MM 2>/dev/null | head -5 || true
echo "残留文件数=$(grep -rl "§4-2 INJECT" /root/MindSpeed-MM 2>/dev/null | wc -l)"
echo "RESTORE_VERIFY_DONE"
