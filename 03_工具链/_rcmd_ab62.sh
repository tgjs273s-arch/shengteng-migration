#!/bin/bash
# §4-2 端到端 A/B：注入 + 后台启动 9 轮
set -u
echo "=== 清掉旧 marker（避免上一轮的残留被当成本轮的自证） ==="
rm -f /tmp/mm_cand_c1_active /tmp/mm_cand_c2_active
echo "markers_after_clean: c1=$(test -f /tmp/mm_cand_c1_active && echo yes || echo no) c2=$(test -f /tmp/mm_cand_c2_active && echo yes || echo no)"
echo "=== 步骤1：备份 + 注入 ==="
cd /root/ops
python3 _ab62_setup.py
SETUP_RC=$?
echo "SETUP_RC=$SETUP_RC"
if [ "$SETUP_RC" != "0" ]; then
  echo "AB62_LAUNCH_ABORT 注入失败，不启动运行"
  exit 1
fi
echo "=== 步骤2：后台启动 9 轮 ==="
TS=$(date +%Y%m%d_%H%M%S)
LAUNCH=/root/ops/_ab62_${TS}.launch.log
echo "LAUNCH_LOG=$LAUNCH"
nohup python3 _ab62_run.py > "$LAUNCH" 2>&1 &
echo "LAUNCHED_AB62 pid=$!"
sleep 20
echo "=== 20 秒后 ==="
tail -n 10 "$LAUNCH" 2>/dev/null || echo "(还没有日志)"
echo "AB62_LAUNCH_DONE"
