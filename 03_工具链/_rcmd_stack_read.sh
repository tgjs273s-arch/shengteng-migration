#!/bin/bash
# 远端工作㉘：读 with-stack 那一轮（把 Computing 拆成"真实计算 vs 重计算"的证据）
# 纪律：先看**文件大小**再决定碰不碰；大文件只取表头/聚合，不整读。
set -u
O=/root/ops/p2prof
for i in $(seq 1 30); do
  [ -f "$O/result.json" ] && break
  sleep 3
done
echo "=== 完成判定 ==="
[ -f "$O/result.json" ] && echo "RESULT_JSON 存在" || echo "仍无 result.json"
grep -oE 'iteration +[0-9]+/ *100' "$O/train.log" 2>/dev/null | tail -1 || echo "（无进度行）"
L=$(ls -1t /root/ops/profstack_*/launch.log 2>/dev/null | head -1)
grep -E 'P2PROF_' "$L" 2>/dev/null | tail -2 || true
echo "=== 桶 ==="
python3 - "$O/result.json" <<'PY' 2>/dev/null || echo "  （result.json 不可读）"
import json, io, sys
d = json.load(io.open(sys.argv[1], encoding="utf-8"))
print("  shares_pct:", {k: round(v, 2) for k, v in (d.get("shares_pct") or {}).items()})
print("  profiled_steps=%s wall=%s launches/step=%s" % (d.get("step_trace_rows"), d.get("wall_seconds"),
      d.get("kernel_launches_per_profiled_step")))
PY
echo "=== prof 产物（按大小，只列前 12）==="
find "$O/prof" -maxdepth 3 -type f -printf '%s %p\n' 2>/dev/null | sort -rn | head -12 | awk '{printf "  %10.1f MB  %s\n", $1/1048576, $2}'
echo "=== kernel_details.csv 表头（找调用栈列）==="
K=$(find "$O/prof" -name 'kernel_details.csv' | head -1)
if [ -n "$K" ]; then
  echo "  $K  ($(du -h "$K" | cut -f1))"
  head -1 "$K" | tr ',' '\n' | nl | head -20
else
  echo "  （无 kernel_details.csv）"
fi
echo "STACK_READ_DONE"
