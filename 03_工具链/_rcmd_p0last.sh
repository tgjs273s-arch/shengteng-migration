#!/bin/bash
# P0 最后一块：save/load 侧的 state 组装条件 + 检查点实际存了哪些键
set -u
TE=/root/MindSpeed-MM/mindspeed_mm/fsdp/train/train_engine.py
R=/root/ops/nzreplay_20260922_150729
echo "=== train_engine.py 的 save()/load() 组装 ==="
grep -nE "def save|def load|state = |\"model\"|'model'|\"optimizer\"|'optimizer'|no_save_optim|no_load_optim" $TE | head -30
echo
echo "=== def save 附近原文 ==="
S=$(grep -n "    def save" $TE | head -1 | cut -d: -f1)
[ -n "$S" ] && sed -n "${S},$((S+45))p" $TE
echo
echo "=== 检查点 .metadata 里到底有哪些键 ==="
python3 - "$R/ckpt/iter_0000030/.metadata" <<'PY'
import sys
p = sys.argv[1]
raw = open(p, "rb").read()
print("size=%d" % len(raw))
import re
keys = sorted(set(re.findall(rb"[A-Za-z_][A-Za-z0-9_.]{2,60}", raw)))
for k in keys[:400]:
    s = k.decode("ascii", "replace")
    if any(x in s.lower() for x in ("model", "optim", "state", "param", "exp_avg")):
        print("  ", s)
PY
echo "=== distcp 文件名形态（能否看出 model/optim 分区） ==="
ls "$R/ckpt/iter_0000030" | sed 's/[0-9]\+/N/g' | sort | uniq -c | head
echo "P0_LAST_DONE"
