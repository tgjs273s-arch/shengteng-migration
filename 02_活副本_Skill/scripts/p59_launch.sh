#!/bin/bash
# p59_launch.sh — 单键 A/B/A 真实训练消融（性能用）
#
# 用法：bash p59_launch.sh <新输出目录> <点号键> <B值> [A值]
#   例：
#     # 优化器路径：现有 adam_fused=true 等价 foreach=False（逐参数）；B 改成 foreach
#     bash p59_launch.sh /root/ops/p59_opt_20260919 training.adam_fused false
#     # 数据预处理是否在关键路径
#     bash p59_launch.sh /root/ops/p59_prep_20260919 \
#          data.dataset_param.basic_parameters.preprocess_on_fly false
#
# 判据（见 59_config_ab.py 头部，跑前已定）：漂移<3%；B/A≤0.95 才算 WIN；
# step1 loss 必须仍为 1.924621；逐点 loss/grad_norm 差 <2%。
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SKILL="${SKILL_DIR:-/root/qwen35-ascend-migrator}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export NON_MEGATRON=true
OUT="${1:?需要一个新的绝对输出目录}"
KEY="${2:?需要点号配置键，例如 training.adam_fused}"
BVAL="${3:?需要 B 值}"
AVAL="${4:-}"
python3 "$HERE/59_config_ab.py" \
    --skill "$SKILL" \
    --config "$SKILL/out/plan/train_config.yaml" \
    --key "$KEY" --b "$BVAL" ${AVAL:+--a "$AVAL"} \
    --steps "${P59_STEPS:-30}" \
    --out "$OUT"
