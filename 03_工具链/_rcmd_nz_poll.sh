#!/bin/bash
# ③ 非零学习率回放：轮询（只读）
set -u
ROOT=$(ls -dt /root/ops/nzreplay_*/ 2>/dev/null | head -1); ROOT=${ROOT%/}
echo "WATCH=$ROOT"
L=$(ls -t /root/ops/_nzreplay_*.launch.log 2>/dev/null | head -1)
echo "LAUNCH=$L"
echo "=== 进度行 ==="
grep -E "^###|NZREPLAY" "$L" 2>/dev/null | tail -12
echo "=== 检查点 ==="
ls -1d "$ROOT"/ckpt/iter_* 2>/dev/null || echo "(还没有)"
echo "=== 检查点内容（是否含优化器状态） ==="
CK=$(ls -1d "$ROOT"/ckpt/iter_* 2>/dev/null | tail -1)
[ -n "${CK:-}" ] && ls "$CK" 2>/dev/null | head -12
[ -n "${CK:-}" ] && echo "optim 相关文件数=$(find "$CK" -iname '*optim*' 2>/dev/null | wc -l)"
echo "=== 回放判定行 ==="
for t in a b; do
  echo "--- $t ---"
  grep -E "REPLAY_(PRECOND|FIXED|C1|C3|C4|OPT_STATE|C2)" "$ROOT/$t/run.log" 2>/dev/null | head -8
  [ -f "$ROOT/$t/rc.txt" ] && cat "$ROOT/$t/rc.txt"
done
echo "=== diff.txt ==="
[ -f "$ROOT/diff.txt" ] && tail -20 "$ROOT/diff.txt" || echo "(还没有)"
echo "=== done? ==="
[ -f "$ROOT/all.done" ] && cat "$ROOT/all.done" || echo "not_yet"
echo "NZ_POLL_DONE"
