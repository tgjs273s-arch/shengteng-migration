#!/bin/bash
# §4-2 侦察：轮询带调用栈的 profile（只读；不用 pkill -f，见坑 207）
# ★ 修正：`ls -dt /root/ops/prof62_*` 会把 `prof62_*.launch.log` **文件**也选进来（它更新），
#   于是 WATCH 指向了一个文件 ⇒ 必须用**尾斜杠**只选目录。
set -u
OUT=$(ls -dt /root/ops/prof62_*/ 2>/dev/null | head -1)
OUT=${OUT%/}
echo "WATCH=$OUT"
if [ -z "$OUT" ]; then echo "NO_PROF62_DIR"; exit 0; fi
echo "=== launch.log 尾部 ==="
tail -n 12 "$OUT.launch.log" 2>/dev/null || echo "(无 launch.log)"
echo "=== result.json ==="
if [ -f "$OUT/result.json" ]; then
  echo "RESULT_READY size=$(stat -c%s "$OUT/result.json")"
else
  echo "no result.json yet"
fi
echo "=== prof 产物 ==="
ls "$OUT/prof" 2>/dev/null | head -20 || echo "(无 prof 目录)"
echo "=== 进程（只读） ==="
ps -eo pid,etime,cmd 2>/dev/null | grep "[5]6_profile_run" | head -3 || echo "(无 56_profile_run)"
echo "POLL62_DONE"
