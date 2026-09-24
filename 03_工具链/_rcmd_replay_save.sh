#!/bin/bash
# 远端工作⑲：把回放比对输出落盘（供本地存证）
set -u
R=$(ls -1d /root/ops/replay_* | tail -1)
cd /root/qwen35-ascend-migrator/scripts
python3 _replay_diff.py --a "$R/a" --b "$R/b" > "$R/diff.txt" 2>&1
echo "DIFF_SAVED $R/diff.txt ($(wc -l < $R/diff.txt) 行)"
echo "=== 每 rank 的判定行 ==="
grep -E 'REPLAY_(VERDICT|IDENTICAL|FAIL)' "$R/diff.txt" || true
echo "=== 产物尺寸 ==="
ls -l "$R"/a/replay_dump.*.json "$R"/b/replay_dump.*.json "$R/diff.txt" | awk '{print $5, $9}'
echo "REPLAY_SAVE_DONE"
