set -u
OUT=/root/ops/prof_20260922_112113
C=$(find "$OUT/prof" -name step_trace_time.csv | head -1)
cp "$C" "$OUT/step_trace_time.csv"
grep -c . "$OUT/result.json" >/dev/null && echo "prof 产物就绪"
ls -l "$OUT/result.json" "$OUT/step_trace_time.csv" "$OUT/launch.log" 2>/dev/null | awk '{print $5, $9}'
echo "PROF_COPY_DONE"
