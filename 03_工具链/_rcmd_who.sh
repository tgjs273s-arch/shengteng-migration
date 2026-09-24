set -u
echo "=== p2ab 批次目录 ==="
ls -1 /root/ops/p2ab_20260922_162640/ 2>/dev/null | head
echo "=== 该批次的驱动/说明 ==="
ls -l /root/ops/p2ab_20260922_162640/*.sh /root/ops/p2ab_20260922_162640/*.md /root/ops/p2ab_20260922_162640/*.out 2>/dev/null | head -5
echo "=== 00_N 的配置差异（相对基线 A）==="
python3 - <<'PY'
import yaml, io, os
a = yaml.safe_load(io.open("/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml", encoding="utf-8"))
p = "/root/ops/p2ab_20260922_162640/00_N/config.yaml"
if os.path.exists(p):
    b = yaml.safe_load(io.open(p, encoding="utf-8"))
    def flat(o, pre=""):
        out = {}
        if isinstance(o, dict):
            for k, v in o.items(): out.update(flat(v, "%s.%s" % (pre, k) if pre else str(k)))
        else: out[pre] = o
        return out
    fa, fb = flat(a), flat(b)
    diff = [(k, fa.get(k), fb.get(k)) for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
    print("  差异键数 =", len(diff))
    for k, x, y in diff[:8]: print("   %-54s A=%r 00_N=%r" % (k, x, y))
else:
    print("  缺 config.yaml")
PY
echo "=== 进度 ==="
grep -oE 'iteration +[0-9]+/ *[0-9]+' /root/ops/p2ab_20260922_162640/00_N/train.log 2>/dev/null | tail -1 || echo "（无进度行）"
echo "=== 其它新目录（谁起的？）==="
for d in fwbackup_p2 nzreplay_20260922_151513 p2shapes_20260922_155829 detsc62; do
  echo "--- $d"; ls -1t /root/ops/$d 2>/dev/null | head -3
done
echo "WHOCHK_DONE"
