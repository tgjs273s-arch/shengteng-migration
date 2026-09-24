#!/bin/bash
# 关机前核对：远端是否有在跑的任务 / 磁盘 / 需要保留的产物（**只读**）
set -eu
echo "--- 训练进程（字符串拆分写法，避免 pgrep 匹配到自己的命令行，坑 175）---"
if pgrep -af "trai""ner.py" >/dev/null 2>&1 || pgrep -af "torch""run" >/dev/null 2>&1; then
  echo "NPU_BUSY"; pgrep -af "trai""ner.py" | head -5; pgrep -af "torch""run" | head -5
else
  echo "NPU_FREE（无训练进程）"
fi
echo "--- 根分区 ---"
df -h /root | tail -1
echo "--- /root/ops 产物目录 ---"
ls -1 /root/ops | tail -22
echo "--- 特意保留的 checkpoint ---"
du -sh /root/ops/p2_keepckpt_* 2>/dev/null || echo "（无）"
echo "DONE"
