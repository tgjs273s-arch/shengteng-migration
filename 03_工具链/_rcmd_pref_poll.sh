set -u
B=$(ls -1dt /root/ops/pref_* | head -1)
echo "BASE=$B"
for i in $(seq 1 28); do [ -f "$B/all.done" ] && break; sleep 3; done
echo "=== 进度 ==="
grep -E 'RUN_(START|DONE)|PREF_ALL_DONE' "$B/all.out" 2>/dev/null | tail -8 || true
echo "=== 各轮 rc ==="
for d in "$B"/v*_r*; do [ -d "$d" ] && echo "  $(basename $d): $(cat $d/rc.txt 2>/dev/null || echo 未结束) logs=$(wc -l < $d/train.log 2>/dev/null || echo 0) 行"; done
echo "PREF_POLL_DONE"
