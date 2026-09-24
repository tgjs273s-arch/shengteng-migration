#!/bin/bash
# 远端工作①：一步回放的**前提查证**（Codex 要求：不能只看配置 dump，要看源码里它门控了什么）
# 另加：确认最近一次实验的产物与 triton 状态仍在（同一份磁盘、换了 IP）
set -eu
cd /root/MindSpeed-MM
echo "=== use_deter_comp 在框架里的使用点 ==="
grep -rn 'use_deter_comp' --include='*.py' . | head -20 || echo "（无命中 ⇒ 前提不成立，须记 UNVERIFIED）"
echo "=== 相关上下文（前后 2 行）==="
for f in $(grep -rl 'use_deter_comp' --include='*.py' . | head -5); do
  echo "--- $f ---"
  grep -n -B2 -A4 'use_deter_comp' "$f" | head -30
done
echo "=== 是否存在确定性相关的其它开关 ==="
grep -rn 'deterministic\|use_deter\|deter_comp' --include='*.py' . | grep -v 'use_deter_comp' | head -12 || true
echo "=== 最近实验产物是否还在（同盘换 IP 的证据）==="
ls -1d /root/ops/AB_AVSB_* /root/ops/phase1b_* /root/ops/p2_keepckpt_* 2>/dev/null | tail -8 || echo "（无历史目录）"
echo "=== triton 状态文件 ==="
python3 -c "import json;d=json.load(open('/root/ops/triton_state_after.json'));print({k:d[k] for k in list(d)[:6]})" 2>/dev/null || echo "（读不到 triton_state_after.json）"
echo "=== Skill 脚本是否含本轮新文件（应为 0）==="
ls -1 /root/qwen35-ascend-migrator/scripts/ | grep -c '_batchfp\|51_train_fp' || true
echo "=== 可用配置 ==="
ls -1 /root/ops/*.yaml | tail -8
echo "PROBE2_DONE"
