#!/bin/bash
# P1 结果：状态 / 前提 / 固定状态 / C1-C4 / diff
set -u
ROOT=$(ls -dt /root/ops/nzreplay_*/ 2>/dev/null | head -1); ROOT=${ROOT%/}
echo "WATCH=$ROOT"
L=$(ls -t /root/ops/_nzreplay_reuse_*.launch.log /root/ops/_nzreplay_*.launch.log 2>/dev/null | head -1)
echo "LAUNCH=$L"
echo "=== 进度 ==="; grep -E "^###" "$L" 2>/dev/null | tail -12
echo "=== all.done ==="; cat "$ROOT/all.done" 2>/dev/null || echo "(无)"
echo "=== status.json 摘要 ==="
python3 - "$ROOT/status.json" <<'PY'
import json, sys, os
p = sys.argv[1]
if not os.path.isfile(p):
    print("(还没有 status.json)"); raise SystemExit
d = json.load(open(p, encoding="utf-8"))
print("  flow_finished=%s verification_passed=%s exit_code=%s"
      % (d.get("flow_finished"), d.get("verification_passed"), d.get("exit_code")))
print("  failures=%s" % d.get("failures"))
print("  resume_root=%s tracker=%s" % (d.get("resume_root"), d.get("tracker")))
print("  restored_iteration=%s" % d.get("restored_iteration"))
for k, v in (d.get("phases") or {}).items():
    print("  phase %-12s %s" % (k, v))
PY
echo "=== 回放前提（每 rank） ==="
for t in a b; do
  echo "--- $t ---"
  grep -hE "REPLAY_PRECOND|REPLAY_FIXED|REPLAY_C1|REPLAY_C3|REPLAY_C4|REPLAY_OPT_STATE|REPLAY_C2" \
    "$ROOT/$t/run.log" 2>/dev/null | sed 's/\[Rank [01] | Local Rank [01]\] //' | head -14
done
echo "=== 实际 lr / 步号（从 dump 读，按 group） ==="
for t in a b; do
  python3 - "$ROOT/$t/replay_dump.rank0.json" "$t" <<'PY'
import json, sys, os
p, tag = sys.argv[1], sys.argv[2]
if not os.path.isfile(p):
    print("  %s: (无 dump)" % tag); raise SystemExit
d = json.load(open(p, encoding="utf-8"))
pre = d.get("preconditions") or {}
fx = d.get("fixed") or {}
print("  %s: iteration=%s lr_param_groups=%s lr_sched_last=%s lr_consistent=%s"
      % (tag, fx.get("iteration"), pre.get("lr_param_groups"), pre.get("lr_sched_last"),
         pre.get("lr_consistent")))
print("      opt_tensors=%s params_changed=%s opt_state_changed=%s status=%s verdicts=%s"
      % (pre.get("opt_state_tensors"), pre.get("params_changed"), pre.get("opt_state_changed"),
         pre.get("status"), pre.get("verdicts")))
print("      C1 loss=%s logits=%s" % ((d.get("C1_forward") or {}).get("loss", {}).get("sha256"),
                                     (d.get("C1_forward") or {}).get("logits", {}).get("sha256")))
print("      C3 sha=%s n=%s" % ((d.get("C3_post_comm_grad") or {}).get("sha_all"),
                               (d.get("C3_post_comm_grad") or {}).get("n_tensors")))
print("      C4 sha=%s n=%s" % ((d.get("C4_after_one_update") or {}).get("sha_all"),
                               (d.get("C4_after_one_update") or {}).get("n_tensors")))
PY
done
echo "=== diff.txt ==="; tail -25 "$ROOT/diff.txt" 2>/dev/null || echo "(还没有)"
echo "NZ_RESULT_DONE"
