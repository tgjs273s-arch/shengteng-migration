#!/bin/bash
# 远端工作⑯：读一次回放的结果（等两轮都结束，或先看 a）
set -u
R=$(ls -1d /root/ops/replay_* | tail -1)
echo "ROOT=$R"
for i in $(seq 1 30); do
  [ -f "$R/all.done" ] && break
  sleep 3
done
echo "=== a / b 的 rc ==="
cat "$R/a/rc.txt" 2>/dev/null || echo "a 未结束"
cat "$R/b/rc.txt" 2>/dev/null || echo "b 未结束"
echo "=== a 的 REPLAY_* 行 ==="
grep -E 'REPLAY_(FIXED|C1|C2|C3|C4|DUMP|DONE|FAIL)' "$R/a/run.log" 2>/dev/null | head -12 || true
echo "=== a 日志尾部（若失败看这里）==="
tail -n 6 "$R/a/run.log" 2>/dev/null | cut -c1-170
echo "=== 产物文件 ==="
ls -l "$R"/a/replay_dump.*.json "$R"/b/replay_dump.*.json 2>/dev/null || echo "（无 dump）"
echo "REPLAY_READ_DONE"
