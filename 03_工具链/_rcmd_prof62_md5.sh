#!/bin/bash
# §4-2 事后取证：证明框架**未被修改**（本协议不做代码改动），并落盘归因数字
set -u
echo "=== 框架 md5（应与开工前一致） ==="
md5sum /root/MindSpeed-MM/mindspeed_mm/fsdp/optimizer/clip_grad_norm.py \
       /root/MindSpeed-MM/mindspeed_mm/fsdp/train/train_engine.py
echo "开工前记录: clip_grad_norm.py=207f9cbad3da6adba9cd19a00b016b88"
echo "开工前记录: train_engine.py  =e0e8879ac9c94b9ece48bf18584d3166"
echo "=== 本次 profile 结果关键字段 ==="
OUT=$(ls -dt /root/ops/prof62_*/ 2>/dev/null | head -1); OUT=${OUT%/}
python3 - "$OUT/result.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for k in ("schema","valid","steps_parsed","wall_seconds"):
    print("  %-14s = %s" % (k, d.get(k)))
print("  shares_pct    =", d.get("shares_pct"))
print("  identity      =", d.get("identity", {}).get("boot"), d.get("identity", {}).get("hostname"))
print("  geom          =", d.get("geometry"))
PY
echo "MD5CHECK_DONE"
