#!/bin/bash
# §4-2 端到端 A/B：重新后台启动 9 轮（注入已在位；先清 marker 以免旧残留当自证）
set -u
echo "=== 注入在位性检查 ==="
grep -c "§4-2 INJECT" /root/MindSpeed-MM/mindspeed_mm/fsdp/optimizer/clip_grad_norm.py 2>/dev/null || echo "clip_grad_norm: 0"
grep -c "§4-2 INJECT" /root/MindSpeed-MM/mindspeed_mm/fsdp/train/train_engine.py 2>/dev/null || echo "train_engine: 0"
echo "=== 清 marker ==="
rm -f /tmp/mm_cand_c1_active /tmp/mm_cand_c2_active
echo "c1=$(test -f /tmp/mm_cand_c1_active && echo yes || echo no) c2=$(test -f /tmp/mm_cand_c2_active && echo yes || echo no)"
echo "=== 后台启动 ==="
cd /root/ops
TS=$(date +%Y%m%d_%H%M%S)
LAUNCH=/root/ops/_ab62_${TS}.launch.log
echo "LAUNCH_LOG=$LAUNCH"
nohup python3 _ab62_run.py > "$LAUNCH" 2>&1 &
echo "LAUNCHED_AB62 pid=$!"
sleep 30
echo "=== 30 秒后 ==="
tail -n 12 "$LAUNCH" 2>/dev/null || echo "(还没有日志)"
echo "AB62_RELAUNCH_DONE"
