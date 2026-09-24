#!/bin/bash
# ③ 诊断 2：dcp 检查点的路径构造规则
set -u
echo "=== 搜索 iter_ 路径构造 ==="
grep -rn "iter_%\|iter_{-1}\|iter_0000\|iteration - 1\|iteration-1\|load_iteration" \
  /root/MindSpeed-MM/mindspeed_mm/fsdp/ 2>/dev/null | head -25
echo
echo "=== DistributedCheckpointer 的 load 实现 ==="
F=$(grep -rl "class DistributedCheckpointer" /root/MindSpeed-MM/mindspeed_mm 2>/dev/null | head -1)
echo "FILE=$F"
[ -n "$F" ] && grep -nE "def |iter_|load_path|path|iteration" "$F" 2>/dev/null | head -40
echo
echo "=== extra_state 里到底是什么（键名） ==="
python3 - <<'PY'
import torch, glob
fs = sorted(glob.glob("/root/ops/nzreplay_*/ckpt/iter_0000030/extra_state/extra_state_rank_0.pt"))
print("files:", fs)
if fs:
    d = torch.load(fs[0], map_location="cpu", weights_only=False)
    print("type:", type(d).__name__)
    if isinstance(d, dict):
        for k in sorted(d.keys()):
            v = d[k]
            print("  %-40s %s" % (k, type(v).__name__))
PY
echo "NZ_DIAG2_DONE"
