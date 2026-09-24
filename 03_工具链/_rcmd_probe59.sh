#!/bin/bash
# 探明远端 Skill 里 59_config_ab.py 的可用名字（避免再猜）
set -u
cd /root/qwen35-ascend-migrator/scripts
echo "=== 远端 59 md5 / 本地交付副本 md5 ==="
md5sum 59_config_ab.py
echo "=== 远端 59 顶层名字（只列我们依赖的） ==="
python3 - <<'PY'
import importlib, sys
sys.path.insert(0, "/root/qwen35-ascend-migrator/scripts")
m = importlib.import_module("59_config_ab")
for name in ("PAT", "win_ms_of", "compute_valid", "paired_stats", "T975", "STEP1_ANCHOR"):
    print("  %-16s %s" % (name, "OK" if hasattr(m, name) else "**缺失**"))
print("  module file =", getattr(m, "__file__", "?"))
PY
echo "=== 远端 Skill 是否有 62_reportability.py ==="
ls -la 62_reportability.py 2>/dev/null || echo "  (远端没有 62 —— 正常，判定在本地做)"
echo "PROBE59_DONE"
