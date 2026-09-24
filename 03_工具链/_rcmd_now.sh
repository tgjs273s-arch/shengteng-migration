set -u
echo "=== 训练/实验进程 ==="
pgrep -af "torch""run|50_train|56_profile|51_train_fp|52_replay" | grep -v pgrep | head -5 || echo "（无）"
echo "=== NPU 上是否有进程 ==="
npu-smi info 2>/dev/null | grep -A3 "Process id" | head -6 || true
echo "=== 最近产物目录（按时间）==="
ls -1dt /root/ops/*/ 2>/dev/null | head -5
echo "=== 磁盘 ==="
df -h /root | tail -1
echo "CHK_DONE"
