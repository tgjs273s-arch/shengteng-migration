#!/bin/bash
# p61_launch.sh — 训练期设备 HBM + cgroup 节流时间序列（第 4 步数据）
#
# 用法：bash p61_launch.sh /root/ops/<新目录> [步骤数]
# 判据见 61_mem_comm_probe.py 头部：
#   · 只报峰值/趋势（0.5s 采样与 0.43s 步长会混叠，不当稳态值）
#   · 节流只报**训练窗口增量**，不用历史累计
#   · **显存占用升高本身不算收益**，只有"等待/步时长下降"才算
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SKILL="${SKILL_DIR:-/root/qwen35-ascend-migrator}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export NON_MEGATRON=true
OUT="${1:?需要一个新的绝对输出目录}"
python3 "$HERE/61_mem_comm_probe.py" \
    --skill "$SKILL" \
    --config "$SKILL/out/plan/train_config.yaml" \
    --steps "${P61_STEPS:-30}" \
    --out "$OUT"
