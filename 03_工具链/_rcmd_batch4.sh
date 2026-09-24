#!/bin/bash
# 远端工作㉞：四臂 × 2 轮排队跑（同一批次内完成，保证可比；每轮 100 步、无 profiling）
#   臂：a=基线(预取1, log1) / b=预取4 / c=log_interval=10 / d=预取4+log10
#   纪律：派生脚本**先断言键存在**；`log_interval` 当前不在 yaml 里但**是框架真实字段**
#         （training_args.py:299），故允许显式添加并打印说明。
set -eu
SRC=/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml
BASE=/root/ops/batch4_$(date +%Y%m%d_%H%M%S)
mkdir -p "$BASE"
echo "BATCH4_BASE=$BASE"

# 先记录 elapsed_time 的口径（决定 log_interval>1 时 iter_ms 的语义）
echo "=== elapsed_time 口径（train_engine.py 231-262 相关行）==="
sed -n '231,262p' /root/MindSpeed-MM/mindspeed_mm/fsdp/train/train_engine.py | grep -nE 'elapsed|log_interval|start|time\.' | head -10

derive () {  # $1=臂名 $2=prefetch $3=log_interval $4=输出
  python3 - "$SRC" "$2" "$3" "$4" <<'PY'
import yaml, io, sys
src, pf, li, dst = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
d = yaml.safe_load(io.open(src, encoding="utf-8"))
plan = d["parallel"]["fsdp_plan"]
need = ("num_to_forward_prefetch", "num_to_backward_prefetch")
miss = [k for k in need if k not in plan]
if miss:
    print("BATCH4_DERIVE_MISSING %s（拒绝自造键）" % miss); sys.exit(9)
plan["num_to_forward_prefetch"] = plan["num_to_backward_prefetch"] = pf
tr = d["training"]
added = "log_interval" not in tr
tr["log_interval"] = li
io.open(dst, "w", encoding="utf-8").write(yaml.safe_dump(d, allow_unicode=True, sort_keys=False))
print("BATCH4_CFG dst=%s prefetch=%d log_interval=%d%s" % (dst, pf, li,
      "（log_interval 原缺，已显式添加；框架真实字段 training_args.py:299）" if added else ""))
PY
}

derive a 1 1  "$BASE/cfg_a.yaml"
derive b 4 1  "$BASE/cfg_b.yaml"
derive c 1 10 "$BASE/cfg_c.yaml"
derive d 4 10 "$BASE/cfg_d.yaml"

cat > "$BASE/all.sh" <<EOF
set -u
cd /root/qwen35-ascend-migrator
for arm in a b c d; do
  for r in 1 2; do
    D="$BASE/\${arm}_r\${r}"; mkdir -p "\$D"
    echo "RUN_START arm=\$arm r=\$r \$(date +%H:%M:%S)"
    python3 scripts/50_train.py --config "$BASE/cfg_\${arm}.yaml" --env out/probe/env.json \\
      --log "\$D/train.log" --workdir /root/MindSpeed-MM --foreground > "\$D/driver.log" 2>&1
    echo "rc=\$?" > "\$D/rc.txt"
    echo "RUN_DONE arm=\$arm r=\$r rc=\$(cat \$D/rc.txt)"
  done
done
echo "BATCH4_ALL_DONE" > "$BASE/all.done"
EOF
nohup bash "$BASE/all.sh" > "$BASE/all.out" 2>&1 &
echo "LAUNCHED_BATCH4 pid=$!"
sleep 15
tail -n 3 "$BASE/all.out" 2>/dev/null || true
echo "BATCH4_LAUNCH_DONE"
