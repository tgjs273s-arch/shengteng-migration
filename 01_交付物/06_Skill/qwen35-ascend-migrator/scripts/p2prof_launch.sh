#!/bin/bash
# p2prof_launch.sh — 官方几何 + profiler 五桶归因（性能余量量化第 2 步）
#
# 用法：bash scripts/p2prof_launch.sh /root/ops/<新目录>
#   · 只用官方几何（GBS=8），**不带** --allow-gbs-mismatch
#   · 通过配置项 tools.profile 开启 Mspeed-MM 自带 profiler（rank0，第 15–20 步，level1，with_cpu）
#   · 产出 step_trace_time.csv：Stage / Computing / Communication / Communication(Not Overlapped) / Free
#   · ⚠ profiling 会拖慢训练：其步时长**不得**用于性能对标，只用于桶的比例归因
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SKILL="${SKILL_DIR:-/root/qwen35-ascend-migrator}"   # ★ 上传到 /root/ops 后 HERE/.. 会变成 /root（坑 136），故显式给默认值
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export NON_MEGATRON=true
OUT="${1:-${P2_OUT:-/root/ops/p2prof}}"
python3 "$HERE/56_profile_run.py" \
    --skill "$SKILL" \
    --config "$SKILL/out/plan/train_config.yaml" \
    --steps "${P2_STEPS:-25}" \
    --prof-start "${P2_PS:-15}" \
    --prof-end "${P2_PE:-20}" \
    --out "$OUT"
