#!/bin/bash
# 远端工作㉙：按**调用栈**聚合算子耗时 —— 拆出"真实计算 vs 重计算"
# ★ 本轮的桶比例不可与前两轮对比（with_stack 把 Free 抬到 65%）；只看**栈维度的归因**。
set -u
D=$(find /root/ops/p2prof/prof -maxdepth 2 -name operator_details.csv | head -1)
echo "OPERATOR_DETAILS=$D  ($(du -h "$D" | cut -f1))"
python3 - "$D" <<'PY'
import csv, io, sys, collections
p = sys.argv[1]
fh = io.open(p, encoding="utf-8", errors="replace")
rd = csv.DictReader(fh)
cols = rd.fieldnames or []
print("列数=%d" % len(cols))
print("列名:", cols)
stack_col = next((c for c in cols if "Stack" in c or "stack" in c), None)
dur_col = next((c for c in cols if c.startswith("Total Time") or "Duration" in c), None) or \
          next((c for c in cols if "Time" in c), None)
name_col = next((c for c in cols if c == "Name" or "Name" in c), None)
print("用列: name=%r duration=%r stack=%r" % (name_col, dur_col, stack_col))
if not stack_col:
    print("STACK_AGG_FAIL 该产物没有调用栈列（with_stack 未生效？）")
    raise SystemExit(0)
by_leaf = collections.Counter()
by_marker = collections.Counter()
tot = 0.0
n = 0
MARKERS = (("recompute", ("recompute", "checkpoint", "CheckpointFunction", "_recomputation",
                          "activation_offload")),
           ("backward", ("autograd", "backward", "torch/autograd")),
           ("forward_only", ()))
for r in rd:
    try:
        d = float(r.get(dur_col) or 0)
    except ValueError:
        continue
    tot += d
    n += 1
    st = (r.get(stack_col) or "")
    low = st.lower()
    # 叶子帧：栈里最后一个 "File \"...\", line N, in FUNC"
    leaf = "?"
    for part in st.split(";"):
        part = part.strip()
        if " in " in part:
            leaf = part.split(" in ")[-1].strip()
    by_leaf[leaf[:70]] += d
    hit = False
    for label, keys in MARKERS:
        if label == "forward_only":
            continue
        if any(k.lower() in low for k in keys):
            by_marker[label] += d
            hit = True
            break
    if not hit:
        by_marker["neither"] += d
print("算子行数=%d 总耗时=%.0f us（仅比例可用）" % (n, tot))
print("--- 按叶子帧（最后 in FUNC）耗时 Top 12 ---")
for k, v in by_leaf.most_common(12):
    print("  %-44s %10.0f us  %5.2f%%" % (k, v, v / tot * 100.0))
print("--- 按标记归类 ---")
for k, v in by_marker.most_common():
    print("  %-16s %10.0f us  %5.2f%%" % (k, v, v / tot * 100.0))
print("STACK_AGG_DONE")
PY
