#!/bin/bash
# p57_launch.sh — FSDP 分组结构 A/B/A 消融（减少"多而小"发射）
#
# 用法：bash scripts/p57_launch.sh /root/ops/<新目录>
#   · 单一变量：只改 parallel.fsdp_plan.apply_modules（去掉父级条目 → 压平为最细一层）
#   · 几何保持官方（GBS=8），30 步/组，A/B/A 三组串行，同 boot 内可比
#   · 判据写在 57_fsdp_group_ablation.py 头部：B ≤ 0.95×A 才算有效；基线漂移 >3% 则不下结论
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SKILL="${SKILL_DIR:-/root/qwen35-ascend-migrator}"   # ★ 上传到 /root/ops 后 HERE/.. 会变成 /root（坑 136），故显式给默认值
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export NON_MEGATRON=true
OUT="${1:-${P57_OUT:-/root/ops/p57}}"
python3 "$HERE/57_fsdp_group_ablation.py" \
    --skill "$SKILL" \
    --config "$SKILL/out/plan/train_config.yaml" \
    --steps "${P57_STEPS:-30}" \
    --order "${P57_ORDER:-A,B,A}" \
    --out "$OUT"
