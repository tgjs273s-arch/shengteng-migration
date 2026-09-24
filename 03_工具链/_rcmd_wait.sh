set -u
N=$(ls -1d /root/ops/fp51neg_* | tail -1)
for i in $(seq 1 28); do [ -f "$N/rc.txt" ] && break; sleep 3; done
echo "NEG rc=$(cat $N/rc.txt 2>/dev/null || echo 未结束)"
echo "NEG yields=$(grep -c BATCHFP_MICRO $N/fp.log 2>/dev/null || echo 0)"
sync; sleep 2
echo "DONE_WAIT"
