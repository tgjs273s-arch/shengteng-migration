#!/bin/bash
# 远端工作㉜：查 FSDP 计划里**真实存在**的键（pregather / reshard_after_forward 等），不猜
set -u
CFG=/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml
echo "=== 当前配置的 parallel 段 ==="
python3 - "$CFG" <<'PY'
import yaml, io, json, sys
d = yaml.safe_load(io.open(sys.argv[1], encoding="utf-8"))
print(json.dumps(d.get("parallel", {}), ensure_ascii=False, indent=1)[:900])
PY
echo "=== fsdp_plan 相关参数定义（params/*.py）==="
grep -rnE 'pregather|reshard_after_forward|fsdp_plan' /root/MindSpeed-MM/mindspeed_mm/fsdp/params/*.py | head -12
echo "=== 使用点（训练引擎/并行应用）==="
grep -rnE 'pregather|reshard_after_forward' /root/MindSpeed-MM/mindspeed_mm/fsdp/train/train_engine.py \
  /root/MindSpeed-MM/mindspeed_mm/fsdp/distributed/*.py 2>/dev/null | head -14
echo "=== 是否有现成函数 pregather_fsdp_params ==="
grep -rn 'def pregather_fsdp_params' /root/MindSpeed-MM/mindspeed_mm/ | head -3
echo "FSDPKEYS_DONE"
