#!/bin/bash
# 远端工作㉛：叶子帧归因（**零卡时**：复用已存在的 operator_details.csv，只是修好解析）
# 实测栈格式：`<file>(<line>): <func>;\n<file>(<line>): <func>;...`（分隔符是"分号 + 字面反斜杠 n"）
# 且**最内层帧在最前** ⇒ 叶子帧 = 第一个片段里最后那个 ': ' 之后的函数名。
set -u
D=$(find /root/ops/p2prof/prof -maxdepth 5 -name operator_details.csv | head -1)
python3 - "$D" <<'PY'
import csv, io, sys, collections
p = sys.argv[1]
rd = csv.DictReader(io.open(p, encoding="utf-8", errors="replace"))
NAME, DUR, STK = "Name", "Device Self Duration(us)", "Call Stack"

def leaf(st):
    segs = [s.strip() for s in st.replace("\\n", ";").split(";") if s.strip()]
    if not segs:
        return "?"
    first = segs[0]
    return first.rsplit(":", 1)[-1].strip() if ":" in first else first[:40]

tot = 0.0
n = 0
by_leaf = collections.Counter()
wait_by_leaf = collections.Counter()
wait_tot = 0.0
mm_by_leaf = collections.Counter()
for r in rd:
    try:
        d = float(r.get(DUR) or 0)
    except ValueError:
        continue
    tot += d
    n += 1
    lf = leaf(r.get(STK) or "")
    by_leaf[lf] += d
    nm = (r.get(NAME) or "")
    if "wait_event" in nm:
        wait_by_leaf[lf] += d
        wait_tot += d
    elif "Matmul" in nm or "MatMul" in nm:
        mm_by_leaf[lf] += d
print("算子行数=%d 设备自耗时=%.0f us" % (n, tot))
print("--- 总体：叶子帧 Top 12 ---")
for k, v in by_leaf.most_common(12):
    print("  %-46s %10.0f us %6.2f%%" % (k[:46], v, v / tot * 100.0))
print("--- wait_event 合计 %.0f us（%.2f%%）；其叶子帧 Top 10 ---" % (wait_tot, wait_tot / tot * 100.0))
for k, v in wait_by_leaf.most_common(10):
    print("  %-46s %10.0f us %6.2f%%（占 wait 的 %.1f%%）" % (k[:46], v, v / tot * 100.0,
                                                              v / wait_tot * 100.0 if wait_tot else 0))
print("--- MatMul 的叶子帧 Top 5 ---")
for k, v in mm_by_leaf.most_common(5):
    print("  %-46s %10.0f us %6.2f%%" % (k[:46], v, v / tot * 100.0))
print("STACK_LEAF_DONE")
PY
