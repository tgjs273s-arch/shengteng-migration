#!/bin/bash
# 远端工作⑥：看第一轮指纹结果（训练 ~140 s，这里等 95 s 再读）
set -u
R=$(ls -1d /root/ops/fp51_* 2>/dev/null | tail -1)
echo "ROOT=$R"
sleep 95
echo "=== rc ==="
cat "$R/rc.txt" 2>/dev/null || echo "(未结束)"
echo "=== train.log 尾部 ==="
tail -n 8 "$R/train.log" 2>/dev/null || true
echo "=== 指纹文件 ==="
ls -l "$R"/fp.log* 2>/dev/null || echo "(无 fp.log)"
echo "=== BATCHFP 行（前 12 行）==="
head -n 12 "$R/fp.log" 2>/dev/null || true
echo "=== BATCHFP 行数与 rank 分布 ==="
for f in "$R"/fp.log*; do [ -e "$f" ] && echo "$(basename $f): $(grep -c BATCHFP "$f" 2>/dev/null || echo 0) 行"; done
echo "=== 是否出现不可摘要标记（必须为 0 才算指纹可信）==="
grep -c 'undigestible\|unhashable\|unstable_repr\|cycle:' "$R"/fp.log* 2>/dev/null || echo 0
echo "FP51_READ_DONE"
