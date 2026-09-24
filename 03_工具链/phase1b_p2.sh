#!/bin/bash
# phase1b_p2.sh —— P2 臂：training.use_deter_comp=true（**仅诊断，不改出货配置**）
#
# 与上一版的差别（都是复核指出的缺陷）：
#   1) 键路径用**正确的** `training.use_deter_comp`（上一版写成 tools.* ⇒ 被 SKIP）；
#   2) **跑原 100 步协议**，不用改 train_iters 的 8 步（cosine+warmup 下截断不等价）；
#      要"前 8 步"就从这份 100 步日志里读，而不是把总步数改小；
#   3) 从**运行日志里的生效配置**（config dump 会打印 use_deter_comp）取证"键是否真被消费"——
#      "键存在 ≠ 功能生效"；
#   4) 派生后**回读断言** use_deter_comp is True 且 GBS=8。
set -u
SKILL=/root/qwen35-ascend-migrator
BASE=/root/ops/A_recommended.yaml
KEY=training.use_deter_comp
ROOT=/root/ops/phase1b_p2_$(date +%Y%m%d_%H%M%S)
mkdir -p "$ROOT"
echo "P2_ROOT=$ROOT  BASE=$BASE  KEY=$KEY  $(date +%H:%M:%S)"

W=0
while pgrep -f "trai""ner.py" >/dev/null 2>&1 || pgrep -f "torch""run" >/dev/null 2>&1; do
  echo "P2_WAIT NPU 占用中（已等 ${W}s）"; sleep 20; W=$((W+20))
  [ "$W" -gt 1800 ] && { echo "P2_WAIT_TIMEOUT"; exit 4; }
done
echo "P2_NPU_FREE waited=${W}s"

python3 - "$BASE" "$ROOT/cfg.yaml" "$KEY" <<'PY'
import sys, yaml
src, dst, key = sys.argv[1], sys.argv[2], sys.argv[3]
d = yaml.safe_load(open(src, encoding="utf-8"))
cur = d
parts = key.split(".")
for p in parts[:-1]:
    if p not in cur:
        print("P2_DERIVE_MISSING 父级 %s 不存在（拒绝自造）" % p); sys.exit(9)
    cur = cur[p]
leaf = parts[-1]
if leaf not in cur:
    print("P2_DERIVE_MISSING %s 不存在（拒绝自造键名）" % key); sys.exit(9)
old = cur[leaf]
cur[leaf] = True
d["training"]["train_iters"] = 100                 # 原协议，不做截断
d["training"]["save"] = dst.replace(".yaml", "_save")
yaml.safe_dump(d, open(dst, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
b = yaml.safe_load(open(dst, encoding="utf-8"))
assert b["training"]["use_deter_comp"] is True, "回读断言失败：use_deter_comp 不是 True"
gbs = (int(b["training"]["micro_batch_size"]) * int(b["training"]["gradient_accumulation_steps"])
       * int(b["parallel"]["data_parallel_size"]))
assert gbs == 8, "GBS 红线被破坏：%d" % gbs
print("P2_DERIVE_OK %s  %r→True  train_iters=%d  GBS=%d" % (dst, old, b["training"]["train_iters"], gbs))
PY
[ $? -ne 0 ] && { echo "P2_ABORT 派生失败"; exit 3; }

for i in 1 2; do
  d="$ROOT/run$i"; mkdir -p "$d"
  echo "P2_RUN run$i START $(date +%H:%M:%S)"
  python3 "$SKILL/scripts/50_train.py" --config "$ROOT/cfg.yaml" --env "$SKILL/out/probe/env.json" \
      --log "$d/train.log" --workdir /root/MindSpeed-MM --world-size 2 > "$d/driver.log" 2>&1
  echo "P2_RUN run$i rc=$? END $(date +%H:%M:%S)"
  echo "P2_ITERS run$i n=$(grep -c iteration "$d/train.log")"
  # ★ 生效取证：train.log 里会 dump 生效配置（含 use_deter_comp 的实际值）
  echo "P2_EFFECTIVE run$i: $(grep -m1 use_deter_comp "$d/train.log" 2>/dev/null || echo NOT_FOUND_IN_LOG)"
  cp "$ROOT/cfg.yaml" "$d/derived_config.yaml"
  rm -rf "$ROOT/cfg_save"
done
echo "P2_DONE $ROOT $(date +%H:%M:%S)"
