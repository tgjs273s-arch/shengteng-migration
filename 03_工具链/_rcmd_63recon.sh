#!/bin/bash
# ③ 非零学习率回放：只读侦察
set -u
CFG=/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml
echo "=== 1) 远端配置里的检查点与优化器相关键 ==="
grep -nE "save|load|optim|rng|interval|lr|warmup|decay|min_lr" "$CFG" | head -40
echo
echo "=== 2) 已有检查点？ ==="
ls -la /root/ops/ 2>/dev/null | grep -iE "ckpt|checkpoint|save" | head
find /root -maxdepth 4 -type d -name "*checkpoint*" 2>/dev/null | head
find /root -maxdepth 4 -type d -name "*save_path*" 2>/dev/null | head
echo
echo "=== 3) Trainer 的检查点/恢复入口 ==="
grep -rn "def .*\(save\|load\|resume\|checkpoint\)" /root/MindSpeed-MM/mindspeed_mm/fsdp/train/trainer.py 2>/dev/null | head -20
echo "--- 训练循环里怎么用它 ---"
grep -nE "save_checkpoint|load_checkpoint|no_save_optim|no_load_optim|save_interval|resume" /root/MindSpeed-MM/mindspeed_mm/fsdp/train/trainer.py 2>/dev/null | head -20
echo
echo "=== 4) 学习率调度：warmup / decay 配置来源 ==="
grep -rn "class .*Scheduler\|def get_lr\|warmup" /root/MindSpeed-MM/mindspeed_mm/fsdp/optimizer/lr_scheduler.py 2>/dev/null | head -15
echo
echo "=== 5) 52_replay.py 在远端是否存在（运行侧副本） ==="
ls -la /root/qwen35-ascend-migrator/scripts/52_replay.py 2>/dev/null || echo "(无)"
sha256sum /root/qwen35-ascend-migrator/scripts/52_replay.py 2>/dev/null
echo "RECON63_DONE"
