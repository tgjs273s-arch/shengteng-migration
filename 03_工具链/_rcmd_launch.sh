#!/bin/bash
# 远端工作④：确定性开关的最后一环 + 取 50_train.py 的真实启动命令（用于换入口跑 51）
set -u
cd /root/qwen35-ascend-migrator
echo "=== set_deterministic_algorithms 真身 ==="
python3 - <<'PY' 2>&1 | head -30
import inspect, sys
sys.path.insert(0, "/root/MindSpeed-MM")
try:
    from mindspeed.fsdp.utils.random import set_deterministic_algorithms as f
    print("FILE =", inspect.getsourcefile(f))
    print(inspect.getsource(f))
except Exception as e:
    print("IMPORT_FAIL %s: %s" % (type(e).__name__, e))
PY
echo "=== 50_train.py --help（取参数名）==="
python3 scripts/50_train.py --help 2>&1 | head -28 || true
echo "=== dry-run：真实启动命令 ==="
python3 scripts/50_train.py --config /root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml --env out/probe/env.json --dry-run 2>&1 | tail -14 || true
echo "=== 51_train_fp.py --help（确认它自己的参数）==="
python3 scripts/51_train_fp.py --help 2>&1 | head -24 || true
echo "PREREQ4_DONE"
