set -u
L=$(ls -1t /root/ops/profstack_*/launch.log 2>/dev/null | head -1)
echo "LAUNCH_LOG=$L"
grep -c . "$L" >/dev/null 2>&1 && tail -2 "$L" | cut -c1-160
[ -d /root/ops/p2prof ] && echo "p2prof 目录已建" || echo "（p2prof 尚未建）"
grep -oE 'iteration +[0-9]+/ *100' /root/ops/p2prof/train.log 2>/dev/null | tail -1 || echo "（暂无训练进度行）"
echo "STACK_CHECK_DONE"
