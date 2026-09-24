#!/bin/bash
# 远端工作㉕：按工具契约重起（--out 必须不存在）—— 采样器写 BASE，profile 写 BASE/run
set -u
CFG=/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml
TS=$(date +%Y%m%d_%H%M%S)
BASE=/root/ops/profmem2_$TS
OUT=$BASE/run
mkdir -p "$BASE"
echo "PROFMEM2_BASE=$BASE  PROF_OUT=$OUT"

# 停掉上一轮遗留的采样器（它写在旧目录里，且训练没起来）
for f in $(ls -1d /root/ops/profmem_2*/ 2>/dev/null); do touch "$f/stop_hbm" 2>/dev/null || true; done
pkill -f 'hbm_sample.sh' 2>/dev/null || true
sleep 1

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
nohup python3 scripts/56_profile_run.py \
  --skill /root/qwen35-ascend-migrator --config "$CFG" \
  --steps 100 --prof-start 50 --prof-end 52 --with-memory \
  --out "$OUT" --mind /root/MindSpeed-MM --timeout 1500 \
  > "$BASE/launch.log" 2>&1 &
echo "LAUNCHED_PROFMEM2 pid=$!"
sleep 35
echo "=== launch.log（应看到 PROF_START 与理由行）==="
grep -E 'PROF_|FATAL' "$BASE/launch.log" 2>/dev/null | head -5 || true
echo "=== hbm.csv 行数 ==="
wc -l < "$BASE/hbm.csv" 2>/dev/null || echo 0
echo "PROFMEM2_LAUNCH_DONE"
