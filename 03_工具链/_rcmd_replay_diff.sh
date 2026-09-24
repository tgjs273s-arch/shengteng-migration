#!/bin/bash
# 远端工作⑱：等 b 结束，然后用 _replay_diff.py 做**跨运行比对**（协议判定树）
set -u
R=$(ls -1d /root/ops/replay_* | tail -1)
echo "ROOT=$R"
for i in $(seq 1 32); do
  [ -f "$R/all.done" ] && break
  sleep 3
done
echo "a rc=$(cat $R/a/rc.txt 2>/dev/null || echo 未结束)   b rc=$(cat $R/b/rc.txt 2>/dev/null || echo 未结束)"
echo "=== 跨运行比对（a vs b）==="
cd /root/qwen35-ascend-migrator/scripts
python3 _replay_diff.py --a "$R/a" --b "$R/b" 2>&1 | head -40
echo "REPLAY_DIFF_DONE"
