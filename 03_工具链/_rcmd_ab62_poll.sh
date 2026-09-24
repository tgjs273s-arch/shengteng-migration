#!/bin/bash
# §4-2 A/B 轮询（只读；不用 pkill -f，见坑 207）
set -u
OUT=$(ls -dt /root/ops/ab62_*/ 2>/dev/null | head -1); OUT=${OUT%/}
echo "WATCH=$OUT"
L=$(ls -t /root/ops/_ab62_*.launch.log 2>/dev/null | head -1)
echo "LAUNCH=$L"
echo "=== 进度（已完成几轮） ==="
grep -c "^RUN " "$L" 2>/dev/null || echo 0
echo "=== 尾部 ==="
tail -n 8 "$L" 2>/dev/null
echo "=== 每轮窗口中位数（若有） ==="
grep "^RUN " "$L" 2>/dev/null | tail -12
echo "=== marker 自证 ==="
echo "c1=$(test -f /tmp/mm_cand_c1_active && echo yes || echo no) c2=$(test -f /tmp/mm_cand_c2_active && echo yes || echo no)"
echo "=== 完成标志 ==="
test -f "$OUT/runs_index.json" && echo "RUNS_INDEX_READY" || echo "not_yet"
grep -c "AB62_DONE" "$L" 2>/dev/null || true
echo "POLL_AB62_DONE"
