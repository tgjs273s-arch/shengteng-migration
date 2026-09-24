#!/bin/bash
# 确认 P2 补丁已还原（改过别人的代码就必须给出"已还原"的证据）
set -u
F=/root/MindSpeed-MM/mindspeed_mm/fsdp/ops/gdn/triton/chunk_o.py
echo "=== 当前文件 vs 备份 ==="
sha256sum "$F" /root/ops/fwbackup_p2/chunk_o.py.orig 2>/dev/null
echo "=== 补丁特征是否残留 ==="
echo "含 'device=g.device)': $(grep -c 'device=g.device)' "$F" 2>/dev/null)"
echo "含 '.to(g.device)'    : $(grep -c '\.to(g.device)' "$F" 2>/dev/null)"
echo "=== 还原日志尾部 ==="
tail -n 4 /root/ops/_p2_cand_restore.log 2>/dev/null || echo "(无独立日志，已在 _rcmd_p2cand 输出里看过)"
echo "P2_RESTORE_CHECK_DONE"
