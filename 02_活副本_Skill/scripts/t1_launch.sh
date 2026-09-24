#!/bin/bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export NON_MEGATRON=true
python3 /root/ops/96_cpu_quota_probe.py
python3 /root/ops/97_cpu_thread_sweep.py --skill /root/qwen35-ascend-migrator --config /root/qwen35-ascend-migrator/out/plan/train_config.yaml --out "${T1_OUT:-/root/ops/t1_results_20260916}" --start-index "${T1_START_INDEX:-0}"
