set -u
CFG=/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml
TS=$(date +%Y%m%d_%H%M%S)
BASE=/root/ops/profmem3_$TS
OUT=$BASE/run
mkdir -p "$BASE"
echo "PROFMEM3_BASE=$BASE  PROF_OUT=$OUT"
cat > "$BASE/hbm_sample.sh" <<'EOS'
set -u
BASE="$1"
echo "t_epoch,chip0_hbm_mb,chip1_hbm_mb" > "$BASE/hbm.csv"
while [ ! -f "$BASE/stop_hbm" ]; do
  T=$(date +%s)
  L=$(npu-smi info 2>/dev/null | grep -oE '[0-9]+ / +65536' | awk '{print $1}' | tr '\n' ',')
  echo "$T,${L%,}" >> "$BASE/hbm.csv"
  sleep 5
done
EOS
nohup bash "$BASE/hbm_sample.sh" "$BASE" > "$BASE/hbm_sample.log" 2>&1 &
echo "HBM_SAMPLER pid=$!"
cd /root/qwen35-ascend-migrator
nohup python3 scripts/56_profile_run.py --skill /root/qwen35-ascend-migrator --config "$CFG" --steps 100 --prof-start 50 --prof-end 52 --with-memory --out "$OUT" --mind /root/MindSpeed-MM --timeout 1500 > "$BASE/launch.log" 2>&1 &
echo "LAUNCHED pid=$!"
sleep 30
grep -E 'PROF_|FATAL' "$BASE/launch.log" 2>/dev/null | head -4 || echo "（launch.log 暂无 PROF_ 行）"
echo "hbm 行数=$(wc -l < $BASE/hbm.csv 2>/dev/null || echo 0)"
echo "PROFMEM3_DONE"
