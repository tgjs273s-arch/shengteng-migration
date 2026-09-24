#!/bin/bash
# 远端工作⑧：跨运行比对指纹（判据：**两轮 fp.log 必须逐字节相同**，否则数据侧不一致）
set -u
R1=$(ls -1d /root/ops/fp51_* | head -1)
R2=$(ls -1d /root/ops/fp51_* | tail -1)
echo "R1=$R1"
echo "R2=$R2"
sleep 90
echo "=== 两轮 rc ==="
echo "R1: $(cat $R1/rc.txt 2>/dev/null || echo 未结束)"
echo "R2: $(cat $R2/rc.txt 2>/dev/null || echo 未结束)"
echo "=== 两轮最后一步 ==="
grep -o 'iteration *100/ *100' "$R1/train.log" 2>/dev/null | tail -1 || echo "R1 未到 100"
grep -o 'iteration *100/ *100' "$R2/train.log" 2>/dev/null | tail -1 || echo "R2 未到 100"
echo "=== rank0 指纹 diff（逐字节）==="
if diff -q "$R1/fp.log" "$R2/fp.log" >/dev/null 2>&1; then
  echo "FP_IDENTICAL rank0=True（两轮指纹逐字节相同）"
else
  echo "FP_IDENTICAL rank0=False"
  diff "$R1/fp.log" "$R2/fp.log" | head -12
fi
echo "=== rank1 指纹 diff ==="
if diff -q "$R1/fp.log.rank1" "$R2/fp.log.rank1" >/dev/null 2>&1; then
  echo "FP_IDENTICAL rank1=True"
else
  echo "FP_IDENTICAL rank1=False"; diff "$R1/fp.log.rank1" "$R2/fp.log.rank1" | head -8
fi
echo "=== 两轮的 BATCHFP_MICRO 行（每步一行汇总）==="
echo "--- R1 ---"; grep BATCHFP_MICRO "$R1/fp.log" | sed 's/.*yield=/yield=/' | head -4
echo "--- R2 ---"; grep BATCHFP_MICRO "$R2/fp.log" | sed 's/.*yield=/yield=/' | head -4
echo "=== 两 rank 是否真的喂了**相同**的 batch（同一步 rank0 vs rank1）==="
echo "rank0 yield1 micro sha: $(grep 'BATCHFP_MICRO' $R1/fp.log | head -1 | sed 's/.*sha256=//')"
echo "rank1 yield1 micro sha: $(grep 'BATCHFP_MICRO' $R1/fp.log.rank1 | head -1 | sed 's/.*sha256=//')"
echo "FP51_DIFF_DONE"
