#!/bin/bash
# 远端工作⑩：等全窗口基线 + 证明派生配置**只差一个键**
set -u
R=$(ls -1d /root/ops/fp51full_* | tail -1)
echo "FULL_ROOT=$R"
echo "=== 派生配置 vs 基线配置的差异键 ==="
python3 - <<'PY'
import yaml, io
a = yaml.safe_load(io.open("/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml", encoding="utf-8"))
b = yaml.safe_load(io.open("/root/ops/fp_neg/cfg_perturbed.yaml", encoding="utf-8"))
def flat(o, p=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flat(v, "%s.%s" % (p, k) if p else str(k)))
    else:
        out[p] = o
    return out
fa, fb = flat(a), flat(b)
keys = sorted(set(fa) | set(fb))
diff = [(k, fa.get(k), fb.get(k)) for k in keys if fa.get(k) != fb.get(k)]
print("NEG_CFG 差异键数 = %d" % len(diff))
for k, x, y in diff[:10]:
    print("   %-62s 基线=%r 派生=%r" % (k, x, y))
PY
sleep 95
echo "=== 基线 rc / 进度 ==="
cat "$R/rc.txt" 2>/dev/null || echo "(未结束)"
grep -c 'BATCHFP ' "$R/fp.log" 2>/dev/null || echo 0
echo "记录到的 yield 数（rank0）= $(grep -c 'BATCHFP_MICRO' $R/fp.log 2>/dev/null || echo 0)"
grep -o 'iteration *1[0-9][0-9]/ *100' "$R/train.log" 2>/dev/null | tail -1 || tail -1 "$R/train.log" | cut -c1-140
echo "FP51_FULL_READ_DONE"
