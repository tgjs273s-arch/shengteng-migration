set -u
CFG=/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml
TS=$(date +%Y%m%d_%H%M%S)
BASE=/root/ops/profstack_$TS
OUT=$BASE/run
mkdir -p "$BASE"
echo "PROFSTACK_BASE=$BASE  OUT=$OUT"
cd /root/qwen35-ascend-migrator
nohup python3 scripts/56_profile_run.py --skill /root/qwen35-ascend-migrator --config "$CFG" --steps 100 --prof-start 50 --prof-end 51 --with-stack > "$BASE/launch.log" 2>&1 &
echo "LAUNCHED_STACK pid=$!"
sleep 25
grep -E 'PROF_|FATAL' "$BASE/launch.log" 2>/dev/null | head -3 || echo "（暂无 PROF_ 行）"
echo "PROFSTACK_LAUNCH_DONE"
