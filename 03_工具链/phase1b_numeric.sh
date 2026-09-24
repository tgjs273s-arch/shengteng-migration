#!/bin/bash
# phase1b_numeric.sh —— 阶段 1b 数值追因：四臂短实验（8 步，固定一切，只动声明的那一项）
#
# 判据见 protocols/numeric_diag_phase1b.json（**先冻结、后执行**）。本脚本只负责"跑"，
# 不在这里做任何结论；每臂两次运行，产物按  run dir/train.log  落盘，供本地转换为
# results.json 后由 numeric_diag.py 裁决。
#
# ★ 两条来自实测的纪律：
#   1) `50_train.py --steps N` **不覆盖**配置里的 `train_iters`（实测：本想跑 6 步却跑了 100 步）
#      ⇒ 本脚本**改配置**里的 `training.train_iters: 8`，不依赖命令行参数。
#   2) **不猜键路径**：P1/P2 要改的键若在配置里不存在，就如实记 `SKIPPED(键不存在)`，
#      绝不自造一个键名去"让它看起来跑过了"（坑 163/179）。
set -u

SKILL=/root/qwen35-ascend-migrator
BASE=/root/ops/A_recommended.yaml
STAMP=$(date +%Y%m%d_%H%M%S)
ROOT=/root/ops/phase1b_$STAMP
STEPS=8
mkdir -p "$ROOT"
echo "P1B_ROOT=$ROOT  BASE=$BASE  STEPS=$STEPS  $(date +%H:%M:%S)"
[ -f "$BASE" ] || { echo "P1B_ABORT 缺基础配置 $BASE"; exit 3; }

# ---- 等待 NPU 空闲（不打断别人的运行）----
W=0
while pgrep -f "trai""ner.py" >/dev/null 2>&1 || pgrep -f "torch""run" >/dev/null 2>&1; do
  echo "P1B_WAIT NPU 占用中（已等 ${W}s）"; sleep 20; W=$((W+20))
  [ "$W" -gt 1800 ] && { echo "P1B_WAIT_TIMEOUT"; exit 4; }
done
echo "P1B_NPU_FREE waited=${W}s"

# ---- 派生配置：只允许改**已存在**的键；回读断言 GBS=8 ----
derive () {   # $1=arm  $2=键  $3=值(真/假/整数)  $4=输出文件
  python3 - "$BASE" "$4" "$2" "$3" "$STEPS" <<'PY'
import sys, yaml
src, dst, key, val, steps = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
d = yaml.safe_load(open(src, encoding="utf-8"))
cur = d
parts = key.split(".")
for p in parts[:-1]:
    if p not in cur:
        print("P1B_DERIVE_MISSING %s（父级 %s 不存在）" % (key, p)); sys.exit(9)
    cur = cur[p]
if parts[-1] not in cur:
    print("P1B_DERIVE_MISSING %s（键不存在，拒绝自造）" % key); sys.exit(9)
old = cur[parts[-1]]
low = val.lower()
cur[parts[-1]] = True if low == "true" else False if low == "false" else int(val) if val.isdigit() else val
d.setdefault("training", {})["train_iters"] = steps          # ★ 改配置，不靠 --steps
d["training"]["save"] = dst.replace(".yaml", "_save")         # 各臂各写自己的目录
yaml.safe_dump(d, open(dst, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
b = yaml.safe_load(open(dst, encoding="utf-8"))
gbs = (int(b["training"]["micro_batch_size"]) * int(b["training"]["gradient_accumulation_steps"])
       * int(b["parallel"]["data_parallel_size"]))
assert gbs == 8, "GBS 红线被破坏：%d" % gbs
assert b["training"]["train_iters"] == steps, "train_iters 未被写成 %d" % steps
print("P1B_DERIVE_OK arm_dst=%s key=%s %r→%r train_iters=%d GBS=%d"
      % (dst, key, old, cur[parts[-1]], b["training"]["train_iters"], gbs))
PY
}

run_twice () {   # $1=arm目录  $2=配置
  for i in 1 2; do
    d="$1/run$i"; mkdir -p "$d"
    echo "P1B_RUN arm=$(basename $1) run$i START $(date +%H:%M:%S)"
    python3 "$SKILL/scripts/50_train.py" --config "$2" --env "$SKILL/out/probe/env.json" \
        --log "$d/train.log" --workdir /root/MindSpeed-MM --world-size 2 > "$d/driver.log" 2>&1
    echo "P1B_RUN arm=$(basename $1) run$i rc=$? END $(date +%H:%M:%S)"
    grep -c "iteration" "$d/train.log" 2>/dev/null | sed 's/^/P1B_ITERS arm='"$(basename $1)"' run'"$i"' n=/'
    # 清 checkpoint（不是本实验的证据，且很占空间）
    rm -rf "$(dirname "$2")/$(basename "$2" .yaml)_save" 2>/dev/null
  done
}

# ---- P0：同配置重复（判据可用性的基线）----
mkdir -p "$ROOT/P0"
derive P0 parallel.data_parallel_size 2 "$ROOT/P0/cfg.yaml" && run_twice "$ROOT/P0" "$ROOT/P0/cfg.yaml"

# ---- P1：num_workers=1（检验多 worker 取样/交错）----
mkdir -p "$ROOT/P1"
if derive P1 data.dataloader_param.num_workers 1 "$ROOT/P1/cfg.yaml"; then
  run_twice "$ROOT/P1" "$ROOT/P1/cfg.yaml"
else
  echo "P1B_SKIPPED P1 num_workers 键不存在"
fi

# ---- P2：确定性算子（**仅诊断**；键不存在则如实跳过）----
mkdir -p "$ROOT/P2"
if derive P2 tools.use_deter_comp true "$ROOT/P2/cfg.yaml"; then
  run_twice "$ROOT/P2" "$ROOT/P2/cfg.yaml"
else
  echo "P1B_SKIPPED P2 use_deter_comp 键不存在（拒绝自造键名）"
fi

# ---- P3：前向→梯度→更新 分段（无现成开关则如实记 UNVERIFIED）----
mkdir -p "$ROOT/P3"
if grep -rqs "forward_only\|no_update\|skip_optimizer" "$SKILL/scripts/50_train.py" "$SKILL/scripts/59_config_ab.py" 2>/dev/null; then
  echo "P1B_P3_AVAILABLE 检测到分段开关 ⇒ 需人工确认用法后再跑（本脚本不猜）"
else
  echo "P1B_P3_UNVERIFIED 现有栈未见前向-only/不更新 的现成开关 ⇒ 按协议记为 UNVERIFIED（不猜实现）"
fi

echo "P1B_DONE root=$ROOT $(date +%H:%M:%S)"
echo "P1B_TREE"; find "$ROOT" -maxdepth 3 -name 'train.log' -printf '%p %s\n' 2>/dev/null
