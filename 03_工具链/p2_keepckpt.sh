#!/bin/bash
# p2_keepckpt.sh —— 补正对照：2 次 use_deter_comp=true（100 步），**保留 checkpoint**
#
# 为什么重跑：`compare_ckpt.py` 这条方法需要"确定性路径下权重应逐位相同"作为正对照，
# 而我把上一批 P2 的 checkpoint `rm -rf` 掉了（坑 187）。本脚本除"保留 checkpoint"外，
# 与 phase1b_p2.sh 完全一致（同一派生断言、同一 100 步协议、同一生效取证）。
set -u
SKILL=/root/qwen35-ascend-migrator
BASE=/root/ops/A_recommended.yaml
KEY=training.use_deter_comp
ROOT=/root/ops/p2_keepckpt_$(date +%Y%m%d_%H%M%S)
mkdir -p "$ROOT"
echo "P2K_ROOT=$ROOT  BASE=$BASE  KEY=$KEY  $(date +%H:%M:%S)"

W=0
while pgrep -f "trai""ner.py" >/dev/null 2>&1 || pgrep -f "torch""run" >/dev/null 2>&1; do
  echo "P2K_WAIT NPU 占用中（已等 ${W}s）"; sleep 20; W=$((W+20))
  [ "$W" -gt 1800 ] && { echo "P2K_WAIT_TIMEOUT"; exit 4; }
done
echo "P2K_NPU_FREE waited=${W}s"

for i in 1 2; do
  d="$ROOT/run$i"; mkdir -p "$d"
  python3 - "$BASE" "$d/cfg.yaml" "$KEY" "$d/save" <<'PY'
import sys, yaml
src, dst, key, save = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
d = yaml.safe_load(open(src, encoding="utf-8"))
cur, parts = d, key.split(".")
for p in parts[:-1]:
    if p not in cur:
        print("P2K_DERIVE_MISSING 父级 %s（拒绝自造）" % p); sys.exit(9)
    cur = cur[p]
leaf = parts[-1]
if leaf not in cur:
    print("P2K_DERIVE_MISSING %s（拒绝自造键名）" % key); sys.exit(9)
old = cur[leaf]; cur[leaf] = True
d["training"]["train_iters"] = 100
d["training"]["save"] = save                       # ★ 每轮独立目录，且**不清理**
yaml.safe_dump(d, open(dst, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
b = yaml.safe_load(open(dst, encoding="utf-8"))
assert b["training"]["use_deter_comp"] is True, "回读断言失败"
gbs = (int(b["training"]["micro_batch_size"]) * int(b["training"]["gradient_accumulation_steps"])
       * int(b["parallel"]["data_parallel_size"]))
assert gbs == 8, "GBS 红线：%d" % gbs
print("P2K_DERIVE_OK %s %r→True train_iters=%d save=%s GBS=%d"
      % (dst, old, b["training"]["train_iters"], save, gbs))
PY
  [ $? -ne 0 ] && { echo "P2K_ABORT 派生失败 run$i"; exit 3; }
  echo "P2K_RUN run$i START $(date +%H:%M:%S)"
  python3 "$SKILL/scripts/50_train.py" --config "$d/cfg.yaml" --env "$SKILL/out/probe/env.json" \
      --log "$d/train.log" --workdir /root/MindSpeed-MM --world-size 2 > "$d/driver.log" 2>&1
  echo "P2K_RUN run$i rc=$? END $(date +%H:%M:%S)"
  echo "P2K_EFFECTIVE run$i: $(grep -m1 use_deter_comp "$d/train.log" 2>/dev/null || echo NOT_FOUND_IN_LOG)"
  echo "P2K_CKPT run$i: $(find "$d/save" -name '*.safetensors' -printf '%p %s\n' 2>/dev/null | head -n 1)"
done
echo "P2K_DONE $ROOT $(date +%H:%M:%S)"
echo "P2K_DISK $(df -h /root | tail -n 1)"
