#!/bin/bash
# A/A 确定性对照：状态 + 判定行 + diff 摘要
set -u
ROOT=$(ls -dt /root/ops/nzreplay_*/ 2>/dev/null | head -1); ROOT=${ROOT%/}
echo "ROOT=$ROOT"
L=$(ls -t /root/ops/_nzreplay_detsc_*.launch.log 2>/dev/null | head -1)
echo "LAUNCH=$L"
echo "=== 进度 ==="; grep -E "^###" "$L" 2>/dev/null | tail -10
echo "=== all.done ==="; cat "$ROOT/all.done" 2>/dev/null || echo "(无)"
echo "=== DETSC 自证 marker ==="
ls -la /tmp/detsc62_active_rank_* 2>/dev/null || echo "(无 marker)"
echo "=== DETSC active 行数 ==="
for t in a_detsc b_detsc; do
  echo -n "  $t: "; grep -c "DETSC active" "$ROOT/$t/run.log" 2>/dev/null || echo 0
done
echo "=== status.json ==="
python3 - "$ROOT/status.json" <<'PY'
import json, sys, os
p = sys.argv[1]
if not os.path.isfile(p): print("(无)"); raise SystemExit
d = json.load(open(p, encoding="utf-8"))
print("  detsc_mode=%s verification_passed=%s exit=%s failures=%s"
      % (d.get("detsc_mode"), d.get("verification_passed"), d.get("exit_code"), d.get("failures")))
print("  restored_iteration=%s" % d.get("restored_iteration"))
for k, v in (d.get("phases") or {}).items(): print("  %-14s %s" % (k, v))
PY
echo "=== detsc diff（若已生成） ==="
f="$ROOT/diff_on.txt"
[ -f "$f" ] && { head -16 "$f"; echo "  ..."; grep -E "REPLAY_VERDICT" "$f"; } || echo "(还没有 diff_on.txt)"
echo "NZ_DETSC_RESULT_DONE"
