#!/bin/bash
# 远端工作㉚：用**设备侧**耗时重做栈归因 + 打印调用栈真实格式（修正上一版的解析与列选择）
set -u
D=$(find /root/ops/p2prof/prof -maxdepth 5 -name operator_details.csv | head -1)
echo "OPERATOR_DETAILS=$(du -h "$D" | cut -f1)"
python3 - "$D" <<'PY'
import csv, io, sys, collections
p = sys.argv[1]
rd = csv.DictReader(io.open(p, encoding="utf-8", errors="replace"))
cols = rd.fieldnames
print("列名:", cols)
NAME = "Name"
DUR = "Device Self Duration(us)"
STK = "Call Stack"
first = None
tot = 0.0
n = 0
by_marker = collections.Counter()
by_leaf = collections.Counter()
by_op_marker = collections.Counter()
MARK = (("recompute", ("recompute", "checkpoint", "CheckpointFunction", "_recomputation", "activation_offload")),
        ("backward", ("autograd", "backward")),
        ("optimizer", ("optimizer", "adamw", "AdamW", "apply_grad")),
        ("dataloader", ("dataloader", "DataLoader", "collate", "fetch")),
        ("communication", ("distributed", "all_reduce", "all_gather", "reduce_scatter", "hccl", "HCCL")))
for r in rd:
    if first is None:
        first = r.get(STK) or ""
    try:
        d = float(r.get(DUR) or 0)
    except ValueError:
        continue
    tot += d
    n += 1
    st = (r.get(STK) or "").replace("\n", ";")
    low = st.lower()
    label = "其它"
    for lab, keys in MARK:
        if any(k.lower() in low for k in keys):
            label = lab
            break
    by_marker[label] += d
    by_op_marker[(label, (r.get(NAME) or "?")[:40])] += d
    segs = [s.strip() for s in st.split(";") if s.strip()]
    leaf = "?"
    for s in segs:
        if " in " in s:
            leaf = s.split(" in ")[-1].strip()
    by_leaf[leaf[:60]] += d
print("算子行数=%d 设备自耗时合计=%.0f us（仅比例可用）" % (n, tot))
print("--- 调用栈真实格式（前 260 字符）---")
print("  " + (first or "（空）")[:260].replace("\n", "\\n"))
print("--- 按标记（设备自耗时占比）---")
for k, v in by_marker.most_common():
    print("  %-14s %10.0f us  %5.2f%%" % (k, v, v / tot * 100.0))
print("--- 叶子帧 Top 10 ---")
for k, v in by_leaf.most_common(10):
    print("  %-58s %10.0f us  %5.2f%%" % (k, v, v / tot * 100.0))
print("--- 每个标记族里最贵的算子 Top 3 ---")
seen = collections.Counter()
for (lab, op), v in by_op_marker.most_common():
    if seen[lab] >= 3:
        continue
    seen[lab] += 1
    print("  [%-12s] %-40s %10.0f us  %5.2f%%" % (lab, op, v, v / tot * 100.0))
print("STACK_AGG2_DONE")
PY
