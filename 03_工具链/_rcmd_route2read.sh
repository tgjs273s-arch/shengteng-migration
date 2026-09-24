#!/bin/bash
# 路线 2 收尾取证：stride/is_contiguous + wy_fast.py 的 wrapper 与 kernel
set -u
echo "=== 1) dump 里的 stride / is_contiguous ==="
python3 - <<'PY'
import json, glob
for f in sorted(glob.glob("/root/ops/argdump_out/*.json"))[:1]:
    d = json.load(open(f, encoding="utf-8"))
    print("file=%s call=%s" % (f.split("/")[-1], d.get("call_index")))
    for k, v in d.get("kwargs", {}).items():
        if "shape" in v:
            print("  %-12s shape=%-22s dtype=%-16s stride=%-22s contiguous=%s"
                  % (k, v.get("shape"), v.get("dtype"), v.get("stride"), v.get("is_contiguous")))
        else:
            print("  %-12s %s" % (k, v))
PY
echo
echo "=== 2) wy_fast.py 的 wrapper（L295-330，含 L307） ==="
sed -n '295,330p' /root/MindSpeed-MM/mindspeed_mm/fsdp/ops/gdn/triton/wy_fast.py
echo
echo "=== 3) 该文件里的 kernel 定义与 triton 配置 ==="
grep -nE "^def |^@triton|num_warps|num_stages|BLOCK|def prepare_wy_repr_bwd|\.contiguous\(\)|transpose\(|float32|to\(torch" \
  /root/MindSpeed-MM/mindspeed_mm/fsdp/ops/gdn/triton/wy_fast.py | head -40
echo "文件行数=$(wc -l < /root/MindSpeed-MM/mindspeed_mm/fsdp/ops/gdn/triton/wy_fast.py)"
echo "ROUTE2_READ_DONE"
