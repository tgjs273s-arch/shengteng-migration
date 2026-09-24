#!/bin/bash
# ============================================================================
# run_from_zero.sh — 从零环境一键验证 Skill 端到端能力
#
# 用途：在一台**全新的、什么都没有的**昇腾机器上，验证 qwen35-ascend-migrator
#       能否独立完成 P0→P7 全流程（环境探测 → 迁移识别 → 方案 → 算子 → 资产
#       → 训练 → 性能 → 判定）。
#
# 设计（★ 关键）：**失败不中断**。目标是把所有缺口一次性暴露出来，
#       而不是在第一个错误处停下。每个阶段独立裁决 PASS/FAIL/SKIP。
#
# 用法：
#   bash scripts/run_from_zero.sh                 # quick 模式（默认，跳过 COCO 19GB 下载）
#   bash scripts/run_from_zero.sh --full          # 全量模式（下载全量 COCO，耗时长）
#   bash scripts/run_from_zero.sh --from P4       # 从指定阶段续跑
#   bash scripts/run_from_zero.sh --data-dir /data --msmm-dir /root/MindSpeed-MM
#
# 产物：
#   out/logs/<stage>.log          每阶段完整日志
#   out/from_zero_report.md       总报告（含裁决表 + 缺口清单）
#   退出码 = 失败阶段数（0 = 全通过）
# ============================================================================
set -u

SKILL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SKILL_ROOT" || exit 2

MODE=quick
DATA_DIR=/root/data
MSMM_DIR=/root/MindSpeed-MM
FROM=PRE
STEPS=100
NODL=0            # ★ --no-download：只验证机制不下载（可与后台资产下载并行）
RESUME=0          # ★ --resume：仅当产物存在**且新于其输入**时才跳过（防止消费陈旧产物，坑 35）

while [ $# -gt 0 ]; do
  case "$1" in
    --quick)     MODE=quick ;;
    --full)      MODE=full ;;
    --from)      FROM="$2"; shift ;;
    --data-dir)  DATA_DIR="$2"; shift ;;
    --msmm-dir)  MSMM_DIR="$2"; shift ;;
    --steps)     STEPS="$2"; shift ;;
    --resume)    RESUME=1 ;;
    --force)     RESUME=0 ;;
    --no-download) NODL=1 ;;     # ★ 只验证流程机制、不触发下载（便于与后台资产下载并行）
    *) echo "未知参数: $1"; exit 2 ;;
  esac
  shift
done

mkdir -p out/logs out/probe out/analyze out/plan out/verify out/assets out/train out/bench out/judge
TSV=out/logs/_verdicts.tsv
: > "$TSV"
REPORT=out/from_zero_report.md
START_TS=$(date '+%Y-%m-%d %H:%M:%S')
STAGE_ORDER="PRE P0 P1 P2 P3 P4 P5 P6 P7"
REACHED=0
STAGE_ART=""      # 由各阶段在调用 run_stage 前设置：本阶段产物路径
STAGE_IN=""       # 本阶段的输入路径（用于陈旧性判断）

say() { echo; echo "################ $* ################"; }

# ★ 续跑闸门：产物必须存在 **且比它的每个输入都新**，否则视为过期 → 重跑。
#   这是坑 35 的结构性防线 —— 只看"文件在不在"会让自检消费陈旧产物并假阳性。
resume_ok() {
  [ "$RESUME" = "1" ] || return 1
  local art="$1"; shift
  [ -s "$art" ] || return 1
  local i
  for i in "$@"; do
    [ -e "$i" ] || continue
    [ "$art" -nt "$i" ] || return 1
  done
  return 0
}

# 判断某阶段是否应当执行（--from 支持）
should_run() {
  local cur="$1" started=0
  for s in $STAGE_ORDER; do
    [ "$s" = "$FROM" ] && started=1
    [ "$s" = "$cur" ] && { [ "$started" = "1" ] && return 0 || return 1; }
  done
  return 1
}

# run_stage <id> <名称> <超时秒> <成功判据描述> <判据命令> -- <实际命令...>
run_stage() {
  local id="$1" name="$2" tmo="$3" crit="$4" judge="$5"; shift 5
  [ "$1" = "--" ] && shift
  local log="out/logs/${id}.log"

  # ★ 续跑闸门（仅在 --resume 时生效）：产物存在且新于输入才跳过，并**显式记录为 RESUMED**
  #   而不是 PASS —— 让报告能区分"本次真的跑了"和"复用了已有产物"。
  if [ -n "$STAGE_ART" ] && resume_ok "$STAGE_ART" ${STAGE_IN:-}; then
    echo "[RESUMED] $id 产物已存在且新于输入：$STAGE_ART（--resume 生效，未重新执行）"
    printf '%s\t%s\tRESUMED\t0\t%s\n' "$id" "$name" "$STAGE_ART" >> "$TSV"
    STAGE_ART=""; STAGE_IN=""
    return 0
  fi
  STAGE_ART=""; STAGE_IN=""

  say "$id · $name"
  echo "判据: $crit"
  echo "命令: $*"
  local t0 t1 rc
  t0=$(date +%s)
  timeout "$tmo" bash -c "$*" > "$log" 2>&1
  rc=$?
  t1=$(date +%s)
  local dur=$((t1 - t0))

  if [ $rc -eq 124 ]; then
    echo "[TIMEOUT] ${dur}s（上限 ${tmo}s）—— 详见 $log"; tail -15 "$log"
    printf '%s\t%s\tTIMEOUT\t%s\t%s\n' "$id" "$name" "$dur" "$log" >> "$TSV"; return 1
  fi
  # 判据独立求值（不用命令退出码，因为脚本可能非零退出但产物有效）
  if eval "$judge"; then
    echo "[PASS] ${dur}s —— $log"; tail -5 "$log"
    printf '%s\t%s\tPASS\t%s\t%s\n' "$id" "$name" "$dur" "$log" >> "$TSV"; return 0
  else
    echo "[FAIL] rc=$rc ${dur}s —— 判据不满足: $crit"
    echo "--- 日志尾部 ---"; tail -25 "$log"
    printf '%s\t%s\tFAIL\t%s\t%s\n' "$id" "$name" "$dur" "$log" >> "$TSV"; return 1
  fi
}

skip_stage() {
  printf '%s\t%s\tSKIP\t0\t-\n' "$1" "$2" >> "$TSV"
  echo "[SKIP] $1 $2"
}

echo "=================================================================="
echo " qwen35-ascend-migrator · 从零环境端到端验证"
echo " 时间: $START_TS  模式: $MODE  起始阶段: $FROM  步数: $STEPS"
echo " Skill 根: $SKILL_ROOT"
echo " 数据目录: $DATA_DIR   MSMM: $MSMM_DIR"
echo "=================================================================="

# ---------------------------------------------------------------- PRE 预检能力矩阵
# ★ 鲁棒性第一道闸门：**开工前**就导出"哪些能跑 / 哪些会降级 / 哪些被阻塞"，
#   并把降级写入 out/degradations.json。避免"跑到一半才发现环境不支持"。
if should_run PRE; then
  REACHED=1
  STAGE_ART="out/preflight/capability.json"
  run_stage PRE "预检能力矩阵" 900 \
    "out/preflight/capability.json 存在（降级已登记）" \
    "[ -s out/preflight/capability.json ]" \
    -- "python3 scripts/05_preflight.py --json --want $MODE"
fi

# ---------------------------------------------------------------- P0 环境探测
if should_run P0; then
  REACHED=1
  STAGE_ART="out/probe/env.json"
  run_stage P0 "环境探测" 900 \
    "out/probe/env.json 存在且可解析" \
    "[ -s out/probe/env.json ] && python3 -c \"import json;d=json.load(open('out/probe/env.json'));assert 'path' in d\"" \
    -- "python3 scripts/00_probe_env.py --out out/probe/env.json --data-dir $DATA_DIR"
  # 打印探测结论（决定后续路径）
  if [ -f out/probe/env.json ]; then
    python3 - <<'PY'
import json
d = json.load(open('out/probe/env.json'))
print("  >>> 执行路径 path =", d.get('path'))
for k in ('chip_name','die_count','cann','torch','torch_npu','triton','transformers','python'):
    if k in d: print("      %-14s %s" % (k, d[k]))
if isinstance(d.get('notes'), list):
    for n in d['notes'][:8]: print("      note:", n)
PY
  fi
fi

# ---------------------------------------------------------------- P1 迁移点识别
if should_run P1; then
  REACHED=1
  STAGE_ART="out/analyze/migrate_points.json"
  STAGE_IN="refs/modeling_qwen3_5__transformers_v5.2.0.py"
  SRC="refs/modeling_qwen3_5__transformers_v5.2.0.py"
  run_stage P1 "迁移点识别(AST)" 600 \
    "out/analyze/migrate_points.json 存在" \
    "[ -s out/analyze/migrate_points.json ]" \
    -- "python3 scripts/10_analyze_points.py $SRC --out out/analyze"
  [ -f out/analyze/migrate_points_report.md ] && head -30 out/analyze/migrate_points_report.md
fi

# ---------------------------------------------------------------- P2 方案与配置
if should_run P2; then
  REACHED=1
  STAGE_ART="out/plan/train_config.yaml"
  STAGE_IN="out/probe/env.json"
  run_stage P2 "迁移方案与配置生成" 600 \
    "out/plan/train_config.yaml 存在" \
    "[ -s out/plan/train_config.yaml ]" \
    -- "python3 scripts/20_plan_migration.py --env out/probe/env.json --points out/analyze/migrate_points.json --out out/plan --steps $STEPS"
  # ★ 红线自检：GBS 必须 = 8
  if [ -s out/plan/train_config.yaml ]; then
    python3 - <<'PY'
import re
t = open('out/plan/train_config.yaml', encoding='utf-8').read()
def g(k):
    m = re.search(r'^\s*%s\s*:\s*(\S+)' % k, t, re.M)
    return m.group(1) if m else None
def gi(k):
    v = g(k)
    try: return int(str(v).strip('"\''))
    except Exception: return None
mbs, gas = gi('micro_batch_size'), gi('global_batch_size')
ws = gi('world_size')
print("  >>> 配置自检: mbs=%s global_batch_size=%s world_size=%s" % (mbs, gas, ws))
if mbs and ws:
    gbs = mbs * ws
    print("      GBS = mbs %d x dp %d = %d  %s" % (mbs, ws, gbs, "OK(==8)" if gbs == 8 else "!! 与官方 GBS=8 不符, 需人工确认"))
PY
  fi
fi

# ---------------------------------------------------------------- P3 算子验证
if should_run P3; then
  REACHED=1
  STAGE_ART="out/verify/ops_matrix.json"
  run_stage P3 "算子与契约验证" 900 \
    "out/verify/ops_matrix.json 存在（且日志无 argparse 接口错）" \
    "[ -s out/verify/ops_matrix.json ] && ! grep -q \"unrecognized arguments\\|error: \" out/logs/P3.log" \
    -- "python3 scripts/30_verify_ops.py --env out/probe/env.json --out out/verify"
fi

# ---------------------------------------------------------------- P4 资产准备
if should_run P4; then
  REACHED=1
  STAGE_ART="out/assets/assets.json"
  [ "${NODL:-0}" = "1" ] && P4DL="--no-download" || P4DL=""
  if [ "$MODE" = "full" ]; then
    P4ARGS="--data-dir $DATA_DIR --msmm-dir $MSMM_DIR"
    P4TMO=14400      # 全量 COCO 约 19GB，给 4 小时
    echo "(full 模式：将下载全量 COCO train2017，耗时较长)"
  else
    P4ARGS="--data-dir $DATA_DIR --msmm-dir $MSMM_DIR --skip-coco"
    P4TMO=3600
    echo "(quick 模式：跳过 COCO 19GB 下载 → 数据可比性将降级，属预期)"
  fi
  run_stage P4 "权重与数据准备" $P4TMO \
    "资产实质齐备（日志出现 ASSETS_OK；仅文件存在不算通过）" \
    "grep -q \"ASSETS_OK\" out/logs/P4.log" \
    -- "python3 scripts/40_prepare_assets.py --env out/probe/env.json $P4ARGS $P4DL --out out/assets"
  [ -f out/assets/assets.json ] && python3 -c "
import json;d=json.load(open('out/assets/assets.json'))
print('  >>> 数据可比性等级:', d.get('data_comparability'))
print('  >>> 警告:', d.get('warnings'))
for k,v in d.get('assets',{}).items(): print('      %-16s %s' % (k, v.get('status')))
"
fi

# ---------------------------------------------------------------- P5 训练执行
if should_run P5; then
  REACHED=1
  STAGE_ART="out/train/train.log"
  STAGE_IN="out/plan/train_config.yaml"
  run_stage P5 "训练执行($STEPS 步)" 10800 \
    "out/train/train.log 含 iteration 记录" \
    "[ -s out/train/train.log ] && grep -q 'iteration' out/train/train.log" \
    -- "python3 scripts/50_train.py --config out/plan/train_config.yaml --env out/probe/env.json --log out/train/train.log --workdir $MSMM_DIR --steps $STEPS --timeout 10000"
  # 打印 step1 loss（与官方 1.924621 比对的第一个锚点）
  if [ -s out/train/train.log ]; then
    echo "  >>> 首步 loss（官方基线 step1 = 1.924621）："
    grep -m1 -iE 'iteration\s+1[^0-9]' out/train/train.log || head -3 out/train/train.log
  fi
fi

# ---------------------------------------------------------------- P6 性能基准
if should_run P6; then
  REACHED=1
  # ★ 坑 117：产物名以**产出脚本**为准。`60_bench.py:81` 写的是
  #   `round_{round}_{tag}.json`，本驱动传 `--round 1 --tag baseline`
  #   → 真实产物是 `round_1_baseline.json`。旧判据写死 `round_1.json`
  #   → **P6 永远 FAIL**（而 rc=0、基准其实已经产出）。
  #   纪律：判据里的产物名必须来自产出脚本，不能凭印象写。
  BENCH_ART="out/bench/round_1_baseline.json"
  SERIES_ART="out/bench/loss_series.csv"
  WINDOW_ART="out/bench/window_50_100.json"
  # ★ 坑 119：本阶段名叫"性能基准(50-100 口径)"，但旧判据**只查一个 json 存在** ——
  #   而 `60_bench.py` 明确声明自己只做**算子级微基准**
  #   （`scope: operator-micro(v0); end-to-end 由 MindSpeed-MM 侧执行`）。
  #   真正的 50–100 窗口指标此前**没有任何脚本被驱动调用**（`95_extract_series.py`
  #   有该能力但从未接线）→ 判据弱于其问题：文件在=PASS，性能有没有测出来没人管。
  #   修法：算子微基准 + 窗口指标**两件产物都产出**，判据断言窗口指标的可证伪字段。
  STAGE_ART="$WINDOW_ART"
  STAGE_IN="out/train/train.log"
  run_stage P6 "性能基准(算子微基准 + 50-100 窗口口径)" 600 \
    "微基准 $BENCH_ART 存在，且 $WINDOW_ART 的 steps_used≥1 / median_ms / samples_per_s 均非空" \
    "[ -s $BENCH_ART ] && python3 -c \"import json;d=json.load(open('$WINDOW_ART'));assert d.get('steps_used',0)>=1 and d.get('median_ms') and d.get('samples_per_s')\"" \
    -- "python3 scripts/60_bench.py --round 1 --tag baseline --out out/bench --note 'from_zero $MODE'
python3 scripts/95_extract_series.py --log out/train/train.log --out $SERIES_ART --window 50,100 --summary-json $WINDOW_ART"
  [ -f "$WINDOW_ART" ] && python3 -c "
import json;d=json.load(open('$WINDOW_ART'))
print('  >>> bench(window %s): steps_used=%s median_ms=%s mean_ms=%s samples_per_s=%s'
      % (d.get('window'), d.get('steps_used'), d.get('median_ms'), d.get('mean_ms'), d.get('samples_per_s')))
"
  [ -f "$BENCH_ART" ] && python3 -c "
import json;d=json.load(open('$BENCH_ART'))
print('  >>> bench(micro): rows=%s backend=%s' % (len(d.get('micro_rows') or []), d.get('backend')))
"
fi

# ---------------------------------------------------------------- P7 精度判定
if should_run P7; then
  REACHED=1
  STAGE_ART="out/judge/verdict.json"
  STAGE_IN="out/train/train.log"
  TAG="fromzero_$(date +%Y%m%d_%H%M%S)"
  # ★ 坑 116：`judge_comparable.py` 的 `verdict.json` **按设计只含证据字段**
  #   （verdict_id / reachability.level / open_gates …），**不含结论名**；
  #   结论名只出现在它 stdout 的 `JUDGE_OK verdict=…` 行，`70_judge.py` 也是
  #   从那里解析并落到 `judge_summary.json`。旧判据直接断言
  #   `verdict.json.get('verdict')` → **P7 永远 FAIL**（而裁决其实完全合格）。
  #   修法：证据完整性看 `verdict.json`（必须有 verdict_id），
  #        结论完整性看 `judge_summary.json`（必须有 verdict + level + verdict_id）。
  run_stage P7 "精度判定(P7)" 900 \
    "verdict.json 有 verdict_id（证据非空壳）且 judge_summary.json 有 verdict/level/verdict_id（结论非空壳）" \
    "python3 -c \"import json;v=json.load(open('out/judge/verdict.json'));s=json.load(open('out/judge/judge_summary.json'));assert v.get('verdict_id') and s.get('verdict') and s.get('level') and s.get('verdict_id')\"" \
    -- "python3 scripts/70_judge.py --log out/train/train.log --config out/plan/train_config.yaml --data-json $DATA_DIR/output_llava_coco_data.json --baseline officialB --baseline-log examples/train/official_baseline.log --out out/judge --registry sk04_judge/evidence/registry.json --tag $TAG"
  [ -f out/judge/judge_summary.json ] && python3 -c "
import json
s=json.load(open('out/judge/judge_summary.json'))
v=json.load(open('out/judge/verdict.json'))
print('  >>> verdict:', s.get('verdict'))
print('  >>> level  :', s.get('level'))
print('  >>> color  :', s.get('color'))
print('  >>> id     :', s.get('verdict_id') or v.get('verdict_id'))
og=v.get('open_gates') or s.get('open_gates') or []
print('  >>> open gates:', og)
"
fi

# ---------------------------------------------------------------- 汇总报告
END_TS=$(date '+%Y-%m-%d %H:%M:%S')
FAILED=$(awk -F'\t' '$3=="FAIL"' "$TSV" | wc -l)
TIMEOUTS=$(awk -F'\t' '$3=="TIMEOUT"' "$TSV" | wc -l)
PASSED=$(awk -F'\t' '$3=="PASS"' "$TSV" | wc -l)
RESUMED=$(awk -F'\t' '$3=="RESUMED"' "$TSV" | wc -l)

# ★ 降级账本一致性检查（鲁棒性要求：**不许悄悄降级**）
#   规则：任一阶段日志出现 DEGRADED 标记，则 out/degradations.json 必须非空。
#   日志说降级了、账本却为空 → 说明有代码路径绕过了 degrade() → 计为失败。
DEG_JSON=out/degradations.json
DEG_N=0
DEG_CUR=0
DEG_STALE=0
DEG_STALE_NAMES=""
if [ -s "$DEG_JSON" ]; then
  # ★ 坑 118：账本会**跨运行累积**（它由 PRE 阶段写出；`--from P6` 这类续跑不会重写它），
  #   旧实现只数条数 → 报告里出现"本轮降级 4 项"，而那 4 条其实是几小时前写的、
  #   其中有的**在当前环境已不成立**（例如"torch_npu 不可用"而训练刚跑通）。
  #   **陈旧账本比空账本更有害：它把历史状态当当期事实。**
  #   这里按时间戳把"本轮"与"陈旧·非本轮"分开报，陈旧项**不计入当期降级**。
  DEG_N=$(python3 -c "import json;print(len(json.load(open('$DEG_JSON'))))" 2>/dev/null || echo 0)
  # ★ 时间戳格式必须归一后再比：账本写 ISO（`2026-09-16T13:18:06`），
  #   START_TS 来自 `date` 用空格（`2026-09-16 20:01:52`）。直接比字符串会因
  #   `'T'(0x54) > ' '(0x20)` 得出"旧的更新" —— 第一版修复正是这样把 4 条陈旧
  #   条目全算成了"本轮"（驱动自己打印的 `本轮 4 / 陈旧 0` 与 ts=13:18 自相矛盾，
  #   靠这条输出才发现）。→ 归一：`T`→空格，截断到秒。
  DEG_CUR=$(python3 -c "
import json,sys
def norm(s):
    return str(s).replace('T',' ')[:19]
try:
    d=json.load(open('$DEG_JSON'))
except Exception:
    print(0); sys.exit()
print(len([x for x in d if norm(x.get('ts','')) >= norm('$START_TS')]))
" 2>/dev/null || echo 0)
  DEG_STALE=$((DEG_N - DEG_CUR))
  DEG_STALE_NAMES=$(python3 -c "
import json,sys
def norm(s):
    return str(s).replace('T',' ')[:19]
try:
    d=json.load(open('$DEG_JSON'))
except Exception:
    sys.exit()
print(','.join('%s(%s)' % (x.get('name'), x.get('ts')) for x in d if norm(x.get('ts','')) < norm('$START_TS')))
" 2>/dev/null || echo "")
fi
LOG_DEG=$(grep -l 'DEGRADED' out/logs/*.log 2>/dev/null | wc -l)
export START_TS_REF="$START_TS"   # 供报告段的 heredoc 判断"本轮 vs 陈旧"
LEDGER_BAD=0
if [ "${LOG_DEG:-0}" -gt 0 ] && [ "${DEG_N:-0}" -eq 0 ]; then
  LEDGER_BAD=1
  echo "[FAIL] 降级账本不一致：${LOG_DEG} 个阶段日志提到 DEGRADED，但 ${DEG_JSON} 为空"
  FAILED=$((FAILED + 1))
fi

{
  echo "# 从零环境端到端验证报告"
  echo
  echo "- 开始: $START_TS"
  echo "- 结束: $END_TS"
  echo "- 模式: **$MODE**   起始阶段: $FROM   训练步数: $STEPS"
  echo "- Skill 根: \`$SKILL_ROOT\`"
  echo "- 数据目录: \`$DATA_DIR\`   MindSpeed-MM: \`$MSMM_DIR\`"
  echo
  echo "## 裁决表"
  echo
  echo "| 阶段 | 名称 | 结果 | 耗时(s) | 日志 |"
  echo "|---|---|---|---|---|"
  while IFS=$'\t' read -r id name st dur log; do
    echo "| $id | $name | **$st** | $dur | [$log]($log) |"
  done < "$TSV"
  echo
  echo "**通过 $PASSED · 失败 $FAILED · 超时 $TIMEOUTS · 复用已有产物 $RESUMED**"
  echo
  echo "## ★ 降级登记（不许悄悄降级）"
  echo
  if [ "${DEG_N:-0}" -gt 0 ]; then
    echo "账本共 **$DEG_N** 条：**本轮 $DEG_CUR 条** / **陈旧·非本轮 $DEG_STALE 条**（详见 \`out/degradations.json\`）："
    echo
    python3 - <<'PYD' 2>/dev/null || echo "(degradations.json 解析失败)"
import json, os
p = 'out/degradations.json'
def _norm(s):
    return str(s).replace('T', ' ')[:19]
start = _norm(os.environ.get('START_TS_REF', ''))
try:
    for d in json.load(open(p, encoding='utf-8')):
        ts = str(d.get('ts', ''))
        mark = '' if (not start or _norm(ts) >= start) else ' ⚠**陈旧（上一轮遗留，非本轮事实）**'
        print("- **%s** [%s] ts=%s — %s%s" % (d.get('name'), d.get('severity'), ts, d.get('reason'), mark))
except Exception as e:
    print("(解析失败: %s)" % e)
PYD
    if [ "${DEG_STALE:-0}" -gt 0 ]; then
      echo
      echo "> ⚠ **账本含 $DEG_STALE 条陈旧条目**（ts 早于本轮开始 $START_TS）：$DEG_STALE_NAMES"
      echo "> 它们由**上一次** PRE 阶段写入，本轮续跑（\`--from\`）不会重写账本 →"
      echo "> **复核时不得当作当期降级事实**（坑 118：陈旧账本比空账本更有害）。"
      echo "> 如需当期口径，请跑 \`--from PRE\` 重新预检。"
    fi
  else
    echo "无降级登记（若阶段日志中出现 DEGRADED 标记则视为账本不一致，已计为失败）。"
  fi
  echo
  echo "## P0 环境档案"
  echo
  echo '```json'
  [ -f out/probe/env.json ] && head -60 out/probe/env.json || echo "(未产出)"
  echo '```'
  echo
  echo "## 失败/超时阶段的关键日志（最后 40 行）"
  echo
  awk -F'\t' '$3=="FAIL"||$3=="TIMEOUT"{print $1"\t"$5}' "$TSV" | while IFS=$'\t' read -r id log; do
    echo "### $id — \`$log\`"
    echo
    echo '```'
    tail -40 "$log" 2>/dev/null || echo "(无日志)"
    echo '```'
    echo
  done
  echo "## 数据可比性声明（诚实红线）"
  echo
  if [ "$MODE" = "quick" ]; then
    echo "> ⚠️ **本次为 quick 模式（\`--skip-coco\`）：COCO 图片未全量下载，数据可比性降级。**"
    echo "> 结论只能表述为「流程连通性已验证」，**不可**宣称精度已对齐官方。"
    echo "> 需改用 \`--full\` 重跑 P4 及之后阶段。"
  else
    echo "> full 模式：以 \`out/assets/assets.json\` 的 \`data_comparability\` 字段为准，不得越级宣称。"
  fi
} > "$REPORT"

echo "=================================================================="
echo " 结果: PASS=$PASSED FAIL=$FAILED TIMEOUT=$TIMEOUTS RESUMED=$RESUMED DEGRADATIONS=$DEG_N(本轮 $DEG_CUR / 陈旧 $DEG_STALE)"
echo " 报告: $REPORT"
echo " 裁决: $TSV"
echo " 降级账本: $DEG_JSON"
echo "=================================================================="
sed 's/^/  /' "$TSV"
[ "$REACHED" = "1" ] || { echo "!! 未执行任何阶段（--from=$FROM 无效？）"; exit 2; }
exit $((FAILED + TIMEOUTS))
