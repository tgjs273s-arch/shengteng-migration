#!/bin/bash
# p2fit_launch.sh — 三点分离诊断（性能余量量化第 1 步）
#
# 用法（真机）：bash scripts/p2fit_launch.sh /root/ops/<新目录>
#   · 三点：mbs4/gas1(GBS=8 官方) + mbs8/gas1 + mbs4/gas2，各 30 步，串行
#   · 只有第一点可参与官方对标；后两点带 --allow-gbs-mismatch → official_comparable=false
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SKILL="${SKILL_DIR:-/root/qwen35-ascend-migrator}"   # ★ 上传到 /root/ops 后 HERE/.. 会变成 /root（坑 136），故显式给默认值
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export NON_MEGATRON=true
OUT="${1:-${P2_OUT:-/root/ops/p2fit}}"
python3 "$HERE/96_cpu_quota_probe.py" || true
python3 "$HERE/54_three_point_fit.py" \
    --skill "$SKILL" \
    --config "$SKILL/out/plan/train_config.yaml" \
    --steps "${P2_STEPS:-30}" \
    --out "$OUT"
