#!/bin/bash
# 远端工作㉒：看 profile 进度（训练步号 + prof 目录内容）
set -u
L=$(ls -1t /root/ops/prof_*.launch.log | head -1)
OUT=${L%.launch.log}
echo "=== 训练进度（最后一步）==="
grep -oE 'iteration +[0-9]+/ *100' "$OUT/train.log" 2>/dev/null | tail -1 || echo "（无 iteration 行）"
echo "=== prof 目录内容 ==="
ls -1 "$OUT/prof" 2>/dev/null | head -10 || echo "（空）"
echo "=== 是否出现 step_trace_time.csv ==="
find "$OUT/prof" -name 'step_trace_time.csv' 2>/dev/null | head -3 || true
echo "=== 再等 90 秒 ==="
sleep 90
echo "=== 90 秒后：完成判定 ==="
[ -f "$OUT/result.json" ] && echo "RESULT_JSON 存在" || echo "尚无 result.json"
grep -oE 'iteration +[0-9]+/ *100' "$OUT/train.log" 2>/dev/null | tail -1 || true
grep -E 'P2PROF_' "$L" 2>/dev/null | tail -8 | cut -c1-200 || true
echo "PROF_POLL2_DONE"
