set -u
ls -1d /root/ops/profmem2_* 2>/dev/null || echo "（无 profmem2 目录）"
ls -1 /root/ops/profmem2_*/ 2>/dev/null | head -8 || true
echo "--- 进程 ---"
pgrep -af "56_profile_run|50_train|trai""ner.py" | head -5 || echo "（无训练进程）"
echo "--- hbm ---"
tail -3 /root/ops/profmem2_*/hbm.csv 2>/dev/null || echo "（无 hbm.csv）"
echo "CHECK_DONE"
