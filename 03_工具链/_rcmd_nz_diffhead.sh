#!/bin/bash
# P1 判定树原文（rank0 段）+ 判定逻辑
set -u
ROOT=$(ls -dt /root/ops/nzreplay_*/ 2>/dev/null | head -1); ROOT=${ROOT%/}
echo "ROOT=$ROOT"
echo "=== diff.txt 前 30 行（rank0 段） ==="
head -30 "$ROOT/diff.txt"
echo
echo "=== diff.txt 总行数与判定行 ==="
wc -l < "$ROOT/diff.txt"
grep -nE "REPLAY_VERDICT|REPLAY_IDENTICAL|REPLAY_FAIL|C1 |C3 |C4 " "$ROOT/diff.txt" | head -20
echo
echo "=== _replay_diff.py 的判定树逻辑（看 C1 用什么字段） ==="
grep -nE "first_diff|C1|C3|C4|loss|logits|VERDICT" /root/qwen35-ascend-migrator/scripts/_replay_diff.py | head -30
echo "NZ_DIFFHEAD_DONE"
