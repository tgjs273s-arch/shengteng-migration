set -u
OUT=/root/ops/prof_20260922_112113
echo "=== result.json ==="
cat "$OUT/result.json"
echo ""
echo "=== step_trace_time.csv 前 3 行 ==="
C=$(find "$OUT/prof" -name step_trace_time.csv | head -1)
head -3 "$C" | cut -c1-220
echo "=== 该窗口内的行（50-55）==="
awk -F, 'NR>1 && $1+0>=49 && $1+0<=56 {print}' "$C" | cut -c1-200 | head -10
echo "PROF_READ_DONE"
