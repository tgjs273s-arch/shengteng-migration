#!/bin/bash
# ============================================================
# npu-run-template.sh —— 昇腾 910B 环境运行模板
# 用法：复制为具体任务脚本，填入实际命令后在昇腾机器执行
# 铁律：运行前必须已在本机完成 git commit（可追溯）
# ============================================================
set -e   # 出错即停

# 1. 激活 CANN 环境（路径以实际安装为准）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 进入项目目录（改为昇腾机器上的实际路径）
cd /path/to/qwen3_5-migrate-poc

# 3. 确认当前版本可追溯（应与本机提交的 commit 一致）
git log -1 --oneline

# 4. 定义本次任务名与时间戳（用于日志命名）
TASK="T3_rmsnorm"
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

# 5. 执行任务并留全量日志（核心命令替换为你自己的）
python verify/compare.py --backend npu --stage "$TASK" \
    2>&1 | tee "logs/${TS}_${TASK}.log"

# 6. 核对：确认无 CPU-回退，且日志完整
echo "=== 运行完成，日志：logs/${TS}_${TASK}.log ==="
echo "请核对日志中的 backend 行（CPU-回退结果作废）"
