#!/bin/bash
# 远端工作②：把"确定性开关到底门控了什么"查到底 —— 读 set_seed 真身 + 启动器是否支持换入口
set -eu
echo "=== set_seed 定义位置 ==="
grep -rn 'def set_seed' /root/MindSpeed-MM/ 2>/dev/null | head -5 || true
python3 - <<'PY' 2>&1 | head -20
import importlib.util, inspect, sys, os
sys.path.insert(0, "/root/MindSpeed-MM")
try:
    from mindspeed.fsdp.utils.random import set_seed
    print("SET_SEED_FILE =", inspect.getsourcefile(set_seed))
    print(inspect.getsource(set_seed))
except Exception as e:
    print("IMPORT_FAIL %s: %s" % (type(e).__name__, e))
PY
echo "=== 框架里还有哪些"确定性/可复现"相关环境变量被设置 ==="
grep -rn 'HCCL_DETERMINISTIC\|ASCEND_LAUNCH_BLOCKING\|use_deterministic_algorithms\|CUBLAS_WORKSPACE\|deterministic' /root/MindSpeed-MM/mindspeed* 2>/dev/null | head -12 || true
echo "=== 50_train.py 是否有 dry-run / 只打印命令的开关 ==="
grep -n 'dry\|print_only\|--print' /root/qwen35-ascend-migrator/scripts/50_train.py | head -12 || true
echo "=== 现有 env.json 的关键字段 ==="
python3 -c "import json;d=json.load(open('/root/qwen35-ascend-migrator/out/probe/env.json'));print(list(d)[:12])" 2>&1 | head -3
echo "PROBE3_DONE"
