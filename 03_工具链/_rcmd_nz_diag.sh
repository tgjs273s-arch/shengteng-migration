#!/bin/bash
# ③ 诊断：回放为什么 rc=1
set -u
ROOT=$(ls -dt /root/ops/nzreplay_*/ 2>/dev/null | head -1); ROOT=${ROOT%/}
echo "ROOT=$ROOT"
echo "=== 检查点完整清单（前 30 + 目录） ==="
find "$ROOT/ckpt/iter_0000030" -maxdepth 1 2>/dev/null | head -30
echo "总文件数=$(find "$ROOT/ckpt/iter_0000030" -type f | wc -l)"
echo "=== 其中名字含 metadata / optim / state 的 ==="
find "$ROOT/ckpt/iter_0000030" -type f \( -iname "*metadata*" -o -iname "*optim*" -o -iname "*state*" \) 2>/dev/null | head -10
echo
echo "=== a/run.log 尾部（真正的错误） ==="
tail -40 "$ROOT/a/run.log" 2>/dev/null
echo
echo "=== a/run.log 里的 REPLAY_ / Error / Traceback ==="
grep -nE "REPLAY_|Error|Traceback|raise |Exception" "$ROOT/a/run.log" 2>/dev/null | tail -25
echo "NZ_DIAG_DONE"
