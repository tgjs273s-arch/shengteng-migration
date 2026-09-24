#!/bin/bash
# 连上后第一件事：确认这是**哪台机器**（目标 IP 变了：199.98.58.213 ≠ 之前机器 B 的 199.103.55.150）
# 只读；输出用于判断"之前的所有产物/结论是否还在同一台机器上"
set -eu
echo "=== 身份 ==="
echo "HOSTNAME=$(hostname)"
echo "DATE=$(date -Is)"
echo "UPTIME=$(uptime -p 2>/dev/null || uptime)"
echo "OS=$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")"
echo "KERNEL=$(uname -r)"
echo "=== NPU ==="
npu-smi info -l 2>&1 | head -18 || echo "npu-smi 不可用"
echo "=== 每 die 显存（若可用）==="
npu-smi info 2>&1 | head -24 || true
echo "=== CANN / 驱动 ==="
ls /usr/local/Ascend/ascend-toolkit/ 2>&1 | head -5 || true
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null | head -4 || true
echo "=== python 栈 ==="
python3 -c "import torch, torch_npu; print('torch=%s torch_npu=%s' % (torch.__version__, torch_npu.__version__))" 2>&1 | tail -3
python3 -c "import triton; print('triton=%s' % triton.__version__)" 2>&1 | tail -1
echo "=== 关键路径（之前的产物在哪台机器上？）==="
for p in /root/ops /root/qwen35-ascend-migrator /root/MindSpeed-MM /root/Qwen3.5-0.8B-hf /root/ops/A_recommended.yaml /root/ops/B_fallback.yaml; do
  if [ -e "$p" ]; then echo "EXISTS  $p"; else echo "MISSING $p"; fi
done
echo "=== /root/ops 目录（若存在）==="
ls -1 /root/ops 2>/dev/null | tail -18 || true
echo "=== 资源 ==="
df -h / | tail -1
free -g 2>/dev/null | head -2 || true
echo "PROBE_DONE"
