#!/bin/bash
# 查 AI core 数（本卡唯一剩余前提）——只读，几乎不占卡
set -u
echo "=== 是否有进程在跑（先确认卡空闲） ==="
ps -eo pid,etime,cmd 2>/dev/null | grep -E "[t]orchrun|[5]0_train|[5]2_replay|[5]6_profile" | head -3 || echo "(无)"
echo "=== npu-smi 概览 ==="
npu-smi info 2>/dev/null | sed -n '1,12p'
echo "=== AI core 数（多路尝试） ==="
npu-smi info -t common -i 0 -c 0 2>/dev/null | grep -iE "aicore|ai core|core num|Chip" | head -8
npu-smi info -t aicore -i 0 -c 0 2>/dev/null | head -8
echo "=== torch_npu 侧 ==="
python3 -c "
import torch, torch_npu
p = torch.npu.get_device_properties(0)
for a in ('name','multi_processor_count','total_memory','aicore_num','cube_num'):
    if hasattr(p, a):
        print('  %-24s %s' % (a, getattr(p, a)))
print('  dir:', [x for x in dir(p) if not x.startswith('_')][:24])
" 2>&1 | tail -8
echo "CORE_COUNT_DONE"
