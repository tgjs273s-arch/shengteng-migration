#!/bin/bash
# P2 A/B 轮询（只读）
set -u
R=$(ls -dt /root/ops/p2ab_*/ 2>/dev/null | head -1); R=${R%/}
echo "WATCH=$R"
L=$(ls -t /root/ops/_p2ab_*.launch.log 2>/dev/null | head -1)
echo "=== 进度 ==="
grep -E "^RUN |^RESTORE|^###" "$L" 2>/dev/null | tail -14
echo "=== all.done ==="
cat "$R/all.done" 2>/dev/null || echo "not_yet"
echo "=== status.json ==="
python3 - "$R/status.json" <<'PY'
import json, sys, os
p = sys.argv[1]
if not os.path.isfile(p): print("(无)"); raise SystemExit
d = json.load(open(p, encoding="utf-8"))
print("  verification_passed=%s exit=%s failures=%s" % (d.get("verification_passed"), d.get("exit_code"), d.get("failures")))
print("  restored_ok=%s orig=%s restored=%s" % (d.get("restored_ok"), str(d.get("orig_sha256"))[:12], str(d.get("restored_sha256"))[:12]))
for r in d.get("runs", []):
    print("  %-8s arm=%-3s rc=%s wall=%-7s iter=%s sha=%s" % (r.get("tag"), r.get("arm"), r.get("rc"), r.get("wall_seconds"), r.get("iteration_lines"), str(r.get("arm_file_sha256"))[:10]))
PY
echo "=== 框架当前 sha（须等于备份） ==="
sha256sum /root/MindSpeed-MM/mindspeed_mm/fsdp/ops/gdn/triton/chunk_o.py /root/ops/fwbackup_p2/chunk_o.py.orig 2>/dev/null | awk '{print substr($1,1,16), $2}'
echo "P2AB_POLL_DONE"
