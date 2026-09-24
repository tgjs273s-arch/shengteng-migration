set -u
echo "=== training_args 里与日志/打印有关的键 ==="
grep -nE 'log_interval|log_|print|verbose|report' /root/MindSpeed-MM/mindspeed_mm/fsdp/params/training_args.py | head -12
echo "=== training_log 的调用点与条件 ==="
grep -nE 'training_log\(|log_interval|iteration %' /root/MindSpeed-MM/mindspeed_mm/fsdp/train/train_engine.py | head -12
echo "=== training_log 本体 336-360 ==="
sed -n '336,360p' /root/MindSpeed-MM/mindspeed_mm/fsdp/train/train_engine.py | cut -c1-150
echo "LOGCONF_DONE"
