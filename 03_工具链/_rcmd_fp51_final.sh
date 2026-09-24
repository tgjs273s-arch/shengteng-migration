#!/bin/bash
# 远端工作⑫：全窗口指纹的**最终比对**
#   ① 基线内部：100 步是否都是同一个 batch？两个 rank 是否同 batch？
#   ② 负向对照：改了**一个真实文本字段**之后，指纹必须变（这是判据灵敏度的证明）
set -u
FULL=$(ls -1d /root/ops/fp51full_* | tail -1)
NEG=$(ls -1d /root/ops/fp51neg_* | tail -1)
for i in $(seq 1 32); do
  [ -f "$NEG/rc.txt" ] && break
  sleep 3
done
echo "基线 rc=$(cat $FULL/rc.txt 2>/dev/null || echo 未结束) yields=$(grep -c BATCHFP_MICRO $FULL/fp.log 2>/dev/null || echo 0)"
echo "负向 rc=$(cat $NEG/rc.txt 2>/dev/null || echo 未结束) yields=$(grep -c BATCHFP_MICRO $NEG/fp.log 2>/dev/null || echo 0)"
echo "=== ① 基线：不同 micro sha 的个数（=1 表示 100 步都在喂同一批数据）==="
grep BATCHFP_MICRO "$FULL/fp.log" | sed 's/.*sha256=//' | sort -u | wc -l
echo "=== ① 基线：rank1 的不同 micro sha 个数 ==="
grep BATCHFP_MICRO "$FULL/fp.log.rank1" | sed 's/.*sha256=//' | sort -u | wc -l
echo "=== ① 基线：rank0 与 rank1 的 micro sha 序列是否相同 ==="
diff <(grep BATCHFP_MICRO "$FULL/fp.log" | sed 's/.*sha256=//') \
     <(grep BATCHFP_MICRO "$FULL/fp.log.rank1" | sed 's/.*sha256=//') >/dev/null && echo "RANKS_SAME_BATCH=True" || echo "RANKS_SAME_BATCH=False"
echo "=== ② 负向对照：基线 vs 负向 的指纹字段行差异数 ==="
diff <(grep 'BATCHFP ' "$FULL/fp.log") <(grep 'BATCHFP ' "$NEG/fp.log") | grep -c '^[<>]' || true
echo "=== ② 差异样例（前 4 行，注意 field 名）==="
diff <(grep 'BATCHFP ' "$FULL/fp.log") <(grep 'BATCHFP ' "$NEG/fp.log") | head -4 | cut -c1-190
echo "=== ② 负向：不同 micro sha 个数（应与基线不同）==="
grep BATCHFP_MICRO "$NEG/fp.log" | sed 's/.*sha256=//' | sort -u | wc -l
echo "=== ② 逐字段：哪些 field 的 sha 变了（yield=1）==="
for f in input_ids labels attention_mask pixel_values position_ids image_grid_thw rope_deltas; do
  a=$(grep "yield=1 " "$FULL/fp.log" | grep "field=$f " | sed 's/.*sha256=//' | head -1)
  b=$(grep "yield=1 " "$NEG/fp.log" | grep "field=$f " | sed 's/.*sha256=//' | head -1)
  [ "$a" = "$b" ] && echo "  $f: 相同" || echo "  $f: **不同** ($a → $b)"
done
echo "FP51_FINAL_DONE"
