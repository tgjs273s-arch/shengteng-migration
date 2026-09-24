#!/bin/bash
# §4-2 代码级同步候选 —— 只读侦察（不改任何文件）
# 目的：在**立协议之前**把注入点事实核清（拒绝"凭记忆重建"）
echo "=== HOST IDENTITY ==="
hostname
cat /proc/sys/kernel/hostname 2>/dev/null
echo "boot_id: $(cat /proc/sys/kernel/random/boot_id 2>/dev/null)"
echo "=== NPU / 版本 ==="
npu-smi info 2>/dev/null | head -12
echo "--- torch_npu ---"
python3 -c "import torch,torch_npu;print('torch',torch.__version__);print('torch_npu',torch_npu.__version__)" 2>&1 | tail -4
echo "=== 目标文件 ==="
find /root/MindSpeed-MM -name train_engine.py 2>/dev/null
find /root/MindSpeed-MM -name "clip_grad_norm*.py" 2>/dev/null
echo "=== 调用点（train_engine.py） ==="
grep -rn "average_losses_across_data_parallel_group\|clip_grad_norm\|def train_step\|def train(" \
  /root/MindSpeed-MM --include=train_engine.py 2>/dev/null | head -30
echo "=== clip_grad_norm 定义（含 max_norm 语义） ==="
grep -rn "def clip_grad_norm\|max_norm" /root/MindSpeed-MM --include=clip_grad_norm.py 2>/dev/null | head -30
echo "=== 是否有 clip_grad<=0 的早退分支 ==="
grep -rn "max_norm > 0\|max_norm == 0\|max_norm <= 0\|if max_norm" /root/MindSpeed-MM --include=*.py 2>/dev/null | head -20
echo "=== 当前使用的配置（A 臂） ==="
ls -la /root/qwen35-ascend-migrator/out/plan/ 2>/dev/null
grep -n "clip_grad\|num_workers\|sampler" /root/qwen35-ascend-migrator/out/plan/train_config.yaml 2>/dev/null
echo "=== 磁盘上的 Skill 与 mock 数据 ==="
ls /root/qwen35-ascend-migrator/scripts/ 2>/dev/null | head -40
ls -la /root/data/ 2>/dev/null | head
echo "=== DONE ==="
