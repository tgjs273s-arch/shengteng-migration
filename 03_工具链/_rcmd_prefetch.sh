#!/bin/bash
# 远端工作㉝：预取深度对照实验（profile 支持：等待集中在 reduce_scatter/all_gather；显存余量大）
#   档位：prefetch=1（基线，即现有 A 配置）/ 2 / 4（forward 与 backward 同时改 —— **声明为 2 键捆**）
#   每档 2 轮（可重复性）；100 步协议不变；无 profiling（绝对时间可用）
set -eu
BASE=/root/ops/pref_$(date +%Y%m%d_%H%M%S)
SRC=/root/ops/qwen3_5_0_8B_mock_dp2_v3.yaml
mkdir -p "$BASE"
echo "PREF_BASE=$BASE"

derive () {  # $1=档位  $2=输出 yaml
  python3 - "$SRC" "$1" "$2" <<'PY'
import yaml, io, sys, json
src, val, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = yaml.safe_load(io.open(src, encoding="utf-8"))
plan = d["parallel"]["fsdp_plan"]
need = ("num_to_forward_prefetch", "num_to_backward_prefetch")
missing = [k for k in need if k not in plan]
if missing:
    print("PREF_DERIVE_MISSING %s（拒绝自造键）" % missing); sys.exit(9)
plan["num_to_forward_prefetch"] = val
plan["num_to_backward_prefetch"] = val
io.open(dst, "w", encoding="utf-8").write(yaml.safe_dump(d, allow_unicode=True, sort_keys=False))
print("PREF_CFG val=%d dst=%s fwd=%s bwd=%s" % (val, dst, plan["num_to_forward_prefetch"],
                                                plan["num_to_backward_prefetch"]))
PY
}

cat > "$BASE/all.sh" <<EOF
set -u
cd /root/qwen35-ascend-migrator
for v in 1 2 4; do
  for r in 1 2; do
    D="$BASE/v\${v}_r\${r}"
    mkdir -p "\$D"
    echo "RUN_START v=\$v r=\$r dir=\$D \$(date +%H:%M:%S)"
    python3 scripts/50_train.py --config "$BASE/cfg_v\${v}.yaml" --env out/probe/env.json \\
      --log "\$D/train.log" --workdir /root/MindSpeed-MM --foreground > "\$D/driver.log" 2>&1
    echo "rc=\$?" > "\$D/rc.txt"
    echo "RUN_DONE v=\$v r=\$r rc=\$(cat \$D/rc.txt)"
  done
done
echo "PREF_ALL_DONE" > "$BASE/all.done"
EOF
for v in 1 2 4; do derive "$v" "$BASE/cfg_v$v.yaml"; done
nohup bash "$BASE/all.sh" > "$BASE/all.out" 2>&1 &
echo "LAUNCHED_ALL pid=$!"
sleep 20
tail -n 3 "$BASE/all.out" 2>/dev/null || true
echo "PREF_LAUNCH_DONE"
