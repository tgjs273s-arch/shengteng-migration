#!/bin/bash
# p2d_launch.sh — 第 4 点"预登记预测"验证（把三点拟合变成可证伪的检验）
#
# 用法：bash scripts/p2d_launch.sh /root/ops/<新目录>
#   step55 里**写死了**由三点解给出的预测值（mbs8/gas2 → 见脚本头部），跑完比对：
#   落在 ±10% 内 → P2D_OK；否则 P2D_FAIL（此时不得引用三点外推）
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SKILL="${SKILL_DIR:-/root/qwen35-ascend-migrator}"   # ★ 上传到 /root/ops 后 HERE/.. 会变成 /root（坑 136），故显式给默认值
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export NON_MEGATRON=true
OUT="${1:-${P2_OUT:-/root/ops/p2fit_D}}"
python3 "$HERE/55_validate_prediction.py" \
    --skill "$SKILL" \
    --config "$SKILL/out/plan/train_config.yaml" \
    --steps "${P2_STEPS:-30}" \
    --out "$OUT"
