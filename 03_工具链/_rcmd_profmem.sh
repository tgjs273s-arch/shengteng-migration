#!/bin/bash
# 远端工作㉔：带显存采集的短 profile + 独立 HBM 采样
set -u
CFG=/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml
TS=$(date +%Y%m%d_%H%M%S)
OUT=/root/ops/profmem_$TS
mkdir -p "$OUT"
echo "PROFMEM_ROOT=$OUT"

cat > "$OUT/hbm_sample.sh" <<'EOS'
set -u
OUT="$1"
echo "t_epoch,chip0_hbm_mb,chip1_hbm_mb" > "$OUT/hbm.csv"
while [ ! -f "$OUT/stop_hbm" ]; do
  T=$(date +%s)
  L=$(npu-smi info 2>/dev/null | grep -oE '[0-9]+ / +65536' | awk '{print $1}' | tr '\n' ',')
  echo "$T,${L%,}" >> "$OUT/hbm.csv"
  sleep 5
done
EOS
nohup bash "$OUT/hbm_sample.sh" "$OUT" > "$OUT/hbm_sample.log" 2>&1 &
echo "HBM_SAMPLER pid=$!"

cd /root/qwen35-ascend-migrator
nohup python3 scripts/56_profile_run.py \
  --skill /root/qwen35-ascend-migrator --config "$CFG" \
  --steps 100 --prof-start 50 --prof-end 52 --with-memory \
  --out "$OUT" --mind /root/MindSpeed-MM --timeout 1500 \
  > "$OUT.launch.log" 2>&1 &
echo "LAUNCHED_PROFMEM pid=$!"
sleep 35
echo "=== launch.log ==="
tail -n 4 "$OUT.launch.log" 2>/dev/null || echo "（无）"
echo "=== hbm.csv 前 4 行 ==="
head -4 "$OUT/hbm.csv" 2>/dev/null || echo "（无）"
echo "PROFMEM_LAUNCH_DONE"
