#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""62_reportability.py —— **可报告性 / 性能达标闸门**（v2：区间下界为硬门）

v2 相对 v1 的三处**契约变更**（起因：外部复核用合成坏例证明 v1 有缺口）
--------------------------------------------------------------------
1. ★ **"至少省时 5%"必须由区间下界判定，而不是点估计**。
   v1 只比较 |点估计| ≥ 阈值，区间跨零仅**告警** ⇒ 复核用合成坏例（点估计 10%、
   95% 区间跨零）实测仍得到 `REPORTABLE` + 退出码 0。**该缺口已修**：
   `performance_pass = (ci95_savings_pct[0] >= threshold_pct)`。
   噪声倍数规则（`2 × 实测底噪`）**保留为工程筛选条件**，但**不再**替代置信区间。
2. ★ **不再用 A/B 字母决定含义**。改名为**角色制**：`baseline` / `candidate`，
   并显式存储 `baseline_run_id` / `candidate_run_id` / `savings_pct` / `ci95_savings_pct`。
   `savings_pct` 的定义**固定为**：
       savings_pct = (baseline − candidate) / 分母 × 100      **正号 = 候选更快**
   分母由 `--denominator {baseline,candidate}` 显式声明并写进产物
   （默认 `baseline`，即标准"省时比例"，也正是本项目主口径 `(B−A)/B`）。
3. ★ **三类结果拆开**：`run_validity`（运行有效）/ `accuracy_validity`（精度有效）/
   `performance_pass`（性能达标），外加派生的 `adoption_ready`。
   **性能达标不等于可以采纳** —— 精度或运行关未闭合时，性能结论照样成立，
   但 `adoption_ready=False`（退出码 3）。

fail-closed（判不动就**不许**当通过）
------------------------------------
缺区间 / 区间非有限 / 区间长度≠2 / 下界>上界 / 运行无效 / 配对身份冲突
（`baseline_run_id == candidate_run_id`）/ 分母为 0 / 非有限值 ⇒ `INVALID_INPUT`。
★ 特别地：**没有区间的旧产物无法通过本闸门**（不能让"没测区间"当"达标"）。

退出码
------
  0 = 性能达标 **且** 可采纳（运行/精度关都 OK）
  3 = 性能达标，但**运行/精度关未闭合** ⇒ 不可采纳（性能结论仍成立）
  4 = 判得动，但**未达**门槛（区间下界 < 阈值）
  5 = 判不动（INVALID_INPUT）
  2 = 用法错误；自检失败 = 1

用法
----
    python3 scripts/62_reportability.py --noise out/null_summary.json \
        --summary out/summary.json --denominator baseline \
        --accuracy-validity unverified --out out/reportability.json
    python3 scripts/62_reportability.py --selftest        # 判据自检（0-GPU）
"""
import argparse
import importlib
import json
import math
import os
import sys

DEFAULT_MIN_EFFECT_PCT = 5.0     # 效应下限（纪律：实测底噪 ≈2%，<5% 不可报告）
DEFAULT_NOISE_MULTIPLIER = 2.0   # 底噪倍数（冻结规则见 protocols/prefetch_depth_20260922.json）
DEFAULT_MIN_PAIRS = 3            # 纪律：n=2 只做方向判断
NOISE_SCHEMA = "p59null.v1"
AB_SCHEMAS = ("p59ab.v2", "t63_two_config_ab.v1")
DENOM_BASELINE = "baseline"      # (baseline - candidate)/baseline —— 标准"省时比例"
DENOM_CANDIDATE = "candidate"    # (baseline - candidate)/candidate
SCOPE_MOCK = "同机同批次（非官方可比）：不得声称优于官方基线"
SCOPE_OFFICIAL = "同机同批次（官方可比几何）"
OUT_SCHEMA = "p62report.v2"

EXIT_PASS = 0
EXIT_ADOPTION_BLOCKED = 3
EXIT_NOT_MET = 4
EXIT_INVALID = 5
EXIT_USAGE = 2

_p59 = None
# 记录是否走了 `paired.diffs` 退回路径（单线程 CLI，按调用清空；只作产物标注用）
_FALLBACK_USED = []


def _p59_mod():
    """复用 `59_config_ab.py` 的 `T975`（纪律 #8：同一语义只留一处实现）。"""
    global _p59
    if _p59 is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        _p59 = importlib.import_module("59_config_ab")
    return _p59


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------- 解析层
def parse_noise(noise):
    """返回 (floor_pct, n_runs, reason)。reason 非 None 表示底噪判不动。"""
    if noise is None:
        return None, None, "NOISE_MISSING"
    if not isinstance(noise, dict):
        return None, None, "NOISE_NOT_OBJECT"
    if noise.get("schema") != NOISE_SCHEMA:
        return None, None, "NOISE_SCHEMA_UNKNOWN"
    n = noise.get("runs")
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        return None, None, "NOISE_RUNS_LT2"
    fl = noise.get("spread_pct")
    if not _finite(fl):
        return None, None, "NOISE_NONFINITE"
    fl = float(fl)
    if fl <= 0.0:
        return None, n, "NOISE_ZERO"
    return fl, n, None


def _run_ms(rec):
    """取该轮的"窗口中位数"。

    ★ **字段名两个都要认**：`59` 写 `window_median_ms`，而 `63` 写
      `window_11_end_median_ms`。只认前者会让 `63` 产物**静默退回**到自带的
      `paired.diffs`，而那里面 `A_ms/B_ms` 是**配置标签**、不是角色
      ⇒ 调用方声明的角色会被**悄悄绕过**（本函数的注释就是被这件事打出来的）。
    复用 `59` 的 `win_ms_of`（纪律 #8：同一语义只留一处实现）。
    """
    try:
        v = _p59_mod().win_ms_of(rec)
        return float(v) if _finite(v) else None
    except KeyError:
        return None
    except Exception:
        for k in ("window_11_end_median_ms", "window_median_ms"):
            if _finite(rec.get(k)):
                return float(rec[k])
        return None


def parse_summary(ab, baseline_variant="A", candidate_variant="B"):
    """从 59/63 的 A/B 产物取逐对数据。

    ★ **角色必须显式声明，不能按字母猜**：项目里的 `variant` A/B 指的是**两个配置**
      （例如 A=推荐配置、B=保底配置），**不等于**"基线/候选"。要评估"配置 A 相对 B 省时"，
      候选就是 A、基线是 B ⇒ 必须传 `baseline_variant="B", candidate_variant="A"`。
      调用方未显式声明时，本函数按默认 A/B 假定角色，并由上层打出 `roles_assumed` 告警
      —— **不允许把"假定"当成"声明"**。

    返回 (diffs, pairs, official_comparable, schema, ids, reason)
    `ids` = (baseline_run_id, candidate_run_id)（能导出就导出，导不出留占位）。
    """
    if ab is None:
        return None, None, None, None, (None, None), "SUMMARY_MISSING"
    del _FALLBACK_USED[:]
    if not isinstance(ab, dict):
        return None, None, None, None, (None, None), "SUMMARY_NOT_OBJECT"
    sch = ab.get("schema")
    schema = sch if sch else "<unlabeled>"
    if sch is not None and sch not in AB_SCHEMAS:
        return None, None, None, schema, (None, None), "SUMMARY_SCHEMA_UNKNOWN"

    official = bool(ab.get("official_comparable"))
    diffs = None
    runs = ab.get("runs")
    if isinstance(runs, list) and runs:
        base_runs = [r for r in runs if isinstance(r, dict) and r.get("variant") == baseline_variant]
        cand_runs = [r for r in runs if isinstance(r, dict) and r.get("variant") == candidate_variant]
        if base_runs and cand_runs:
            diffs = []
            n = min(len(base_runs), len(cand_runs))
            for i in range(n):
                a = _run_ms(base_runs[i])
                b = _run_ms(cand_runs[i])
                if a is not None and b is not None:
                    diffs.append({"pair": i + 1, "A_ms": a, "B_ms": b})
            if not diffs:
                diffs = None
    if diffs is None:
        pr = ab.get("paired")
        if isinstance(pr, dict) and isinstance(pr.get("diffs"), list):
            ok = [d for d in pr["diffs"]
                  if isinstance(d, dict) and _finite(d.get("A_ms")) and _finite(d.get("B_ms"))]
            # ★ 退回路径的 `A_ms/B_ms` 是**产物自己的配置标签**。若调用方声明的角色与
            #   默认 A/B 相反，必须**按角色换位**，否则声明的角色会被静默忽略
            #   （这正是"不按字母决定含义"要防的事）。
            if ok and (baseline_variant, candidate_variant) == ("B", "A"):
                ok = [{"pair": d.get("pair"), "A_ms": float(d["B_ms"]), "B_ms": float(d["A_ms"])}
                      for d in ok]
            diffs = ok or None
            if diffs:
                _FALLBACK_USED.append("paired.diffs（角色已按声明换位）"
                                      if (baseline_variant, candidate_variant) == ("B", "A")
                                      else "paired.diffs（默认角色）")
    if diffs is None:
        return None, None, official, schema, (None, None), "SUMMARY_NO_PAIRS"

    b_id = ab.get("baseline_run_id")
    c_id = ab.get("candidate_run_id")
    if not b_id or not c_id:
        # 尽量从 runs 的 tag 导出（升级路径：将来产物应自带这两个字段）
        if isinstance(runs, list) and runs:
            bt = [r.get("tag") for r in runs
                  if isinstance(r, dict) and r.get("variant") == baseline_variant]
            ct = [r.get("tag") for r in runs
                  if isinstance(r, dict) and r.get("variant") == candidate_variant]
            b_id = b_id or ("/".join([t for t in bt if t]) or None)
            c_id = c_id or ("/".join([t for t in ct if t]) or None)
    return diffs, len(diffs), official, schema, (b_id, c_id), None


def per_pair_savings(diffs, denominator):
    """逐对算"省时"：**正号 = 候选更快**。(baseline − candidate) / 分母 × 100"""
    vals = []
    for d in diffs:
        base, cand = float(d["A_ms"]), float(d["B_ms"])
        den = base if denominator == DENOM_BASELINE else cand
        if den == 0.0:
            return None, "SAVINGS_ZERO_DENOMINATOR"
        vals.append((base - cand) / den * 100.0)
    return vals, None


def summarize(vals):
    """均值 + 样本 sd + 95% 区间（t 临界值复用 59 的 T975）。"""
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return mean, None, None
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    sd = var ** 0.5
    tcrit = _p59_mod().T975.get(n - 1, 1.96)
    half = tcrit * sd / (n ** 0.5)
    return mean, sd, [mean - half, mean + half]


# --------------------------------------------------------------------- 判据层
def evaluate(savings_pct, ci, pairs, threshold, run_validity, accuracy_validity,
             min_pairs=DEFAULT_MIN_PAIRS, sentinel_metric=False,
             baseline_run_id=None, candidate_run_id=None, official_comparable=False):
    """核心判据（纯函数）：三类结果**独立**给出，另派生 `adoption_ready`。

    硬门（v2 的要点）：`performance_pass = (ci95_savings_pct[0] >= threshold)`。
    """
    # ---- fail-closed 输入检查 ----
    if not _finite(savings_pct):
        return {"verdict": "INVALID_INPUT", "reason": "SAVINGS_NONFINITE",
                "note": "省时点估计不是有限数（NaN/inf）⇒ 不得当通过"}
    if ci is None:
        return {"verdict": "INVALID_INPUT", "reason": "CI_MISSING",
                "note": "★ 缺 95% 区间 ⇒ **判不动**。v2 起『至少省时 X%』必须由**区间下界**判定，"
                        "没有区间就无法判定达标；不得用点估计顶替（v1 的缺口正在此）"}
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return {"verdict": "INVALID_INPUT", "reason": "CI_MALFORMED",
                "note": "95% 区间必须是长度 2 的数组"}
    if not (_finite(ci[0]) and _finite(ci[1])):
        return {"verdict": "INVALID_INPUT", "reason": "CI_NONFINITE",
                "note": "区间端点不是有限数"}
    if float(ci[0]) > float(ci[1]):
        return {"verdict": "INVALID_INPUT", "reason": "CI_INVERTED",
                "note": "区间下界 > 上界 ⇒ 产物自相矛盾"}
    if not isinstance(pairs, int) or isinstance(pairs, bool) or pairs < 0:
        return {"verdict": "INVALID_INPUT", "reason": "PAIRS_MISSING",
                "note": "缺配对数 ⇒ 无法判断样本量"}
    if not _finite(threshold):
        return {"verdict": "INVALID_INPUT", "reason": "THRESHOLD_NONFINITE",
                "note": "阈值不是有限数"}
    if run_validity == "INVALID":
        return {"verdict": "INVALID_INPUT", "reason": "RUN_INVALID",
                "note": "运行无效 ⇒ 计时不可信，性能判据一并判不动（fail-closed）"}
    if run_validity not in ("OK", "UNVERIFIED"):
        return {"verdict": "INVALID_INPUT", "reason": "RUN_VALIDITY_UNKNOWN_VALUE",
                "note": "run_validity 只接受 OK / INVALID / UNVERIFIED"}
    if accuracy_validity not in ("OK", "FAIL", "UNVERIFIED"):
        return {"verdict": "INVALID_INPUT", "reason": "ACCURACY_VALIDITY_UNKNOWN_VALUE",
                "note": "accuracy_validity 只接受 OK / FAIL / UNVERIFIED"}
    if baseline_run_id and candidate_run_id and str(baseline_run_id) == str(candidate_run_id):
        return {"verdict": "INVALID_INPUT", "reason": "PAIRING_IDENTITY_CONFLICT",
                "note": "基线与候选的 run id 相同 ⇒ 这是同一份数据，配对身份冲突，判不动"}

    savings_pct = float(savings_pct)
    threshold = float(threshold)
    ci_lo, ci_hi = float(ci[0]), float(ci[1])
    scope = SCOPE_OFFICIAL if official_comparable else SCOPE_MOCK

    # ---- 三类结果（独立）----
    performance_pass = (ci_lo >= threshold) and (pairs >= min_pairs)
    adoption_ready = (performance_pass and run_validity == "OK"
                      and accuracy_validity == "OK" and not sentinel_metric)

    res = {
        "verdict": None, "reason": None,
        "run_validity": run_validity,
        "accuracy_validity": accuracy_validity,
        "performance_pass": bool(performance_pass),
        "adoption_ready": bool(adoption_ready),
        "savings_pct": round(savings_pct, 4),
        "ci95_savings_pct": [round(ci_lo, 4), round(ci_hi, 4)],
        "threshold_pct": round(threshold, 4),
        "pairs": pairs, "min_pairs": min_pairs,
        "baseline_run_id": baseline_run_id, "candidate_run_id": candidate_run_id,
        "sentinel_metric": bool(sentinel_metric),
        "claim_scope": scope, "official_comparable": bool(official_comparable),
        "ci_lower_ge_threshold": bool(ci_lo >= threshold),
        "ci_excludes_zero": bool(ci_lo > 0.0),
        "direction": ("candidate_faster" if savings_pct > 0 else
                      ("candidate_slower" if savings_pct < 0 else "no_difference")),
    }

    if not performance_pass:
        if pairs < min_pairs and ci_lo >= threshold:
            res["reason"] = "PAIRS_LT_MIN"
            res["note"] = ("区间下界达标，但配对数 n=%d < %d ⇒ 证据不足，判不达标"
                           "（纪律：n=2 只做方向判断）" % (pairs, min_pairs))
        else:
            res["reason"] = "CI_LOWER_BELOW_THRESHOLD"
            res["note"] = ("★ 95%% 区间下界 %.3f%% < 阈值 %.3f%% ⇒ **不达「至少省时 %.1f%%」**。"
                           "点估计 %.3f%%（区间 [%.3f%%, %.3f%%]）**不能**用来宣称达标："
                           "区间下界才是硬门。区间跨零 = 在本样本量下不显著。"
                           % (ci_lo, threshold, threshold, savings_pct, ci_lo, ci_hi))
        res["verdict"] = "PERFORMANCE_NOT_MET"
        return res

    blockers = []
    if run_validity != "OK":
        blockers.append("run_validity=%s" % run_validity)
    if accuracy_validity != "OK":
        blockers.append("accuracy_validity=%s" % accuracy_validity)
    if sentinel_metric:
        blockers.append("sentinel_metric=True（指标被哨兵值替代，验收观测已失效）")
    if blockers:
        res["verdict"] = "PERFORMANCE_MET_ADOPTION_BLOCKED"
        res["reason"] = "ADOPTION_GATE_OPEN"
        res["adoption_blockers"] = blockers
        res["note"] = ("性能达标（区间下界 %.3f%% >= %.3f%%），**但不可采纳**：%s。"
                       "性能结论本身成立；采纳需另开运行/精度关。" % (ci_lo, threshold, "；".join(blockers)))
        return res

    res["verdict"] = "PERFORMANCE_MET_ADOPTION_READY"
    res["reason"] = "OK"
    res["note"] = ("区间下界 %.3f%% >= 阈值 %.3f%%、n=%d、运行与精度关均 OK ⇒ **达标且可采纳**。"
                   "口径限定语必须同现：%s" % (ci_lo, threshold, pairs, scope))
    return res


# --------------------------------------------------------------------- 运行层
def _load_json(path):
    with open(path, encoding="utf-8-sig") as fh:
        return json.load(fh)


def run(args):
    try:
        noise = _load_json(args.noise)
    except Exception as exc:
        print("P62_INVALID reason=NOISE_UNREADABLE detail=%s: %s" % (type(exc).__name__, exc))
        return EXIT_INVALID
    floor_pct, n_runs, nreason = parse_noise(noise)
    if nreason:
        print("P62_INVALID reason=%s noise=%s" % (nreason, args.noise))
        print("P62_NOTE 底噪判不动 ⇒ 不得把任何效应当「通过」。处置：先跑 "
              "`59_config_ab.py --null-test 3` 生成 null_summary.json。")
        return EXIT_INVALID

    threshold_from_noise = args.noise_multiplier * floor_pct
    threshold = max(threshold_from_noise, args.min_effect)

    # ★ 角色：默认 A/B 只作**假定**，并在产物里标记 roles_assumed（见 parse_summary 的说明）
    bv = args.baseline_variant or "A"
    cv = args.candidate_variant or "B"
    roles_assumed = False

    if args.savings is not None:
        try:
            savings = float(args.savings)
        except ValueError:
            print("P62_USAGE --savings 不是数：%r" % args.savings)
            return EXIT_USAGE
        ci = None
        if args.ci_low is not None and args.ci_high is not None:
            ci = [float(args.ci_low), float(args.ci_high)]
        pairs = int(args.pairs)
        official = bool(args.official_comparable)
        schema, ids = "cli", (args.baseline_run_id, args.candidate_run_id)
        src = "cli"
    else:
        try:
            ab = _load_json(args.summary) if args.summary else None
        except Exception as exc:
            print("P62_INVALID reason=SUMMARY_UNREADABLE detail=%s: %s" % (type(exc).__name__, exc))
            return EXIT_INVALID
        roles_assumed = not bool(args.baseline_variant and args.candidate_variant)
        diffs, pairs, official, schema, ids, sreason = parse_summary(
            ab, baseline_variant=bv, candidate_variant=cv)
        src = args.summary
        if sreason:
            print("P62_INVALID reason=%s summary=%s" % (sreason, args.summary))
            print("P62_NOTE 效应判不动 ⇒ 不得当通过。处置：给 `--summary <summary.json>`，"
                  "或用 `--savings <百分点> --ci-low <下界> --ci-high <上界> --pairs <n>`。")
            return EXIT_INVALID
        vals, vreason = per_pair_savings(diffs, args.denominator)
        if vreason:
            print("P62_INVALID reason=%s denominator=%s" % (vreason, args.denominator))
            return EXIT_INVALID
        savings, _sd, ci = summarize(vals)
        if schema == "<unlabeled>":
            print("P62_WARN schema_unverified 该产物没有 schema 标签（逐对契约已校验）")
        if roles_assumed:
            print("P62_WARN roles_assumed 未显式声明角色 ⇒ 按默认 baseline_variant=%s / "
                  "candidate_variant=%s 假定。★ variant 的 A/B 指的是**两个配置**，"
                  "**不等于**基线/候选；要评估某个配置相对另一个的省时，必须显式指定"
                  "（例如评估「配置 A 相对 B 省时」 ⇒ --baseline-variant B --candidate-variant A）"
                  % (bv, cv))

    b_id = args.baseline_run_id or ids[0]
    c_id = args.candidate_run_id or ids[1]
    res = evaluate(savings, ci, pairs, threshold,
                   run_validity=args.run_validity,
                   accuracy_validity=args.accuracy_validity,
                   min_pairs=args.min_pairs,
                   sentinel_metric=args.sentinel_metric,
                   baseline_run_id=b_id, candidate_run_id=c_id,
                   official_comparable=official)
    res.update({
        "schema": OUT_SCHEMA, "denominator": args.denominator,
        "savings_definition": ("(baseline - candidate) / %s × 100 ；**正号 = 候选更快**"
                               % ("baseline" if args.denominator == DENOM_BASELINE else "candidate")),
        "summary_source": src, "summary_schema": schema, "noise_source": args.noise,
        "baseline_variant": bv, "candidate_variant": cv, "roles_assumed": bool(roles_assumed),
        "pairs_source": (_FALLBACK_USED[0] if _FALLBACK_USED else "runs（按声明角色分臂）"),
        "noise_runs": n_runs, "noise_floor_pct": round(floor_pct, 4),
        "noise_multiplier": args.noise_multiplier,
        "noise_screen_pct": round(threshold_from_noise, 4),
        "noise_screen_role": ("仅作**工程筛选**（冻结规则：小于底噪 2 倍的效应先按 UNCERTAIN）；"
                              "**不替代**区间下界硬门"),
        "min_effect_pct": args.min_effect,
        "threshold_governed_by": ("measured_noise_x_mult" if threshold_from_noise >= args.min_effect
                                  else "min_effect_floor"),
    })

    print("P62_INPUT baseline_run_id=%s candidate_run_id=%s" % (b_id, c_id))
    print("P62_INPUT noise_floor_pct=%.3f (runs=%s) savings_pct=%.4f pairs=%s denominator=%s"
          % (floor_pct, n_runs, savings, pairs, args.denominator))
    print("P62_DIRECTION %s（%s）" % (res["direction"], res["savings_definition"]))
    print("P62_THRESHOLD threshold_pct=%.3f governed_by=%s（噪声筛选项 %.3f = %.1f×%.3f）"
          % (threshold, res["threshold_governed_by"], threshold_from_noise,
             args.noise_multiplier, floor_pct))
    print("P62_CI95_SAVINGS %s  lower_ge_threshold=%s excludes_zero=%s"
          % (res["ci95_savings_pct"], res["ci_lower_ge_threshold"], res["ci_excludes_zero"]))
    print("P62_THREE_RESULTS run_validity=%s accuracy_validity=%s performance_pass=%s adoption_ready=%s"
          % (res["run_validity"], res["accuracy_validity"], res["performance_pass"], res["adoption_ready"]))
    if res.get("adoption_blockers"):
        print("P62_ADOPTION_BLOCKERS %s" % "；".join(res["adoption_blockers"]))
    print("P62_VERDICT %s reason=%s" % (res["verdict"], res["reason"]))
    print("P62_SCOPE %s" % res["claim_scope"])
    print("P62_NOTE %s" % res["note"])

    if args.out:
        d = os.path.dirname(os.path.abspath(args.out))
        if d and not os.path.isdir(d):
            os.makedirs(d)
        with open(args.out, "w", encoding="utf-8", newline="") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)
        print("P62_OUT %s" % args.out)

    if res["verdict"] == "PERFORMANCE_MET_ADOPTION_READY":
        print("P62_PASS verdict=%s savings_pct=%.4f ci_lower=%.4f threshold=%.4f"
              % (res["verdict"], res["savings_pct"], res["ci95_savings_pct"][0], res["threshold_pct"]))
        return EXIT_PASS
    if res["verdict"] == "PERFORMANCE_MET_ADOPTION_BLOCKED":
        print("P62_PASS_ADOPTION_BLOCKED savings_pct=%.4f ci_lower=%.4f threshold=%.4f"
              % (res["savings_pct"], res["ci95_savings_pct"][0], res["threshold_pct"]))
        return EXIT_ADOPTION_BLOCKED
    if res["verdict"] == "PERFORMANCE_NOT_MET":
        print("P62_NOT_MET reason=%s savings_pct=%.4f ci_lower=%.4f threshold=%.4f"
              % (res["reason"], res["savings_pct"], res["ci95_savings_pct"][0], res["threshold_pct"]))
        return EXIT_NOT_MET
    print("P62_INVALID reason=%s" % res["reason"])
    return EXIT_INVALID


# --------------------------------------------------------------------- 自检
def _noise(spread=2.0, runs=3, schema=NOISE_SCHEMA):
    return {"schema": schema, "runs": runs, "spread_pct": spread,
            "medians_ms": [100.0, 101.0, 102.0]}


def _e2e_tmpdir():
    """给端到端自检找一个**可写**的临时目录。

    ★ 为什么不能只用 `tempfile.mkdtemp()`：它内部是 `os.mkdir(p, 0o700)`，
      而在某些宿主（例如带文件沙箱的 Windows）**0o700 建出的目录后续不可写**
      （见 `docs/PITFALLS_坑表.md` 坑 214）⇒ 自检会以一个与本判据无关的
      `PermissionError` 失败。这里先试 `mkdtemp`（正常宿主的行为），
      写入探针失败就退回"系统临时根目录 + 唯一名"（默认 mode，可写），
      使自检**在两类宿主下都能跑**，而不是要求宿主去适配判据。
    """
    import shutil
    import tempfile
    try:
        d = tempfile.mkdtemp(prefix="p62e2e_")
        probe = os.path.join(d, ".probe")
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("1")
        os.remove(probe)
        return d
    except OSError:
        pass
    d = os.path.join(tempfile.gettempdir(), "p62_e2e_%d" % os.getpid())
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    return d


def selftest():
    """★ 每条坏例同时断言 **最终 JSON 字段 + 退出码**（不只是"输出里出现过某个词"）。

    判别力自问（纪律 #4）：**若机制真的坏了，这个数会变吗？**
    """
    import shutil
    import subprocess

    cases = []

    def ev(name, savings, ci, pairs, want_verdict, want_reason,
           threshold=5.0, run_validity="OK", accuracy_validity="OK",
           sentinel=False, min_pairs=DEFAULT_MIN_PAIRS,
           b_id="B1", c_id="C1"):
        r = evaluate(savings, ci, pairs, threshold, run_validity, accuracy_validity,
                     min_pairs=min_pairs, sentinel_metric=sentinel,
                     baseline_run_id=b_id, candidate_run_id=c_id)
        ok = (r["verdict"] == want_verdict and r["reason"] == want_reason)
        cases.append((name, ok, r["verdict"], r["reason"], want_verdict, want_reason))

    # ---- ★ 复核发现的缺口：点估计达标但区间跨零（v1 曾判 REPORTABLE）----
    ev("★坏例(复核现场)：点估计 10%、区间 [-14.84, 34.84] 跨零 ⇒ 不达标",
       10.0, [-14.84, 34.84], 3, "PERFORMANCE_NOT_MET", "CI_LOWER_BELOW_THRESHOLD")
    ev("坏例：区间下界 3% < 5%（点估计 10%）⇒ 不达标",
       10.0, [3.0, 17.0], 3, "PERFORMANCE_NOT_MET", "CI_LOWER_BELOW_THRESHOLD")
    ev("坏例：区间下界 4.99%（差一点）⇒ 不达标",
       10.0, [4.99, 20.0], 3, "PERFORMANCE_NOT_MET", "CI_LOWER_BELOW_THRESHOLD")
    ev("正例：区间下界恰为 5.0% ⇒ 达标（>= 即通过）",
       10.0, [5.0, 20.0], 3, "PERFORMANCE_MET_ADOPTION_READY", "OK")
    ev("正例：区间下界 19.606% ⇒ 达标（交付主结论的量级）",
       19.606, [18.323, 20.890], 3, "PERFORMANCE_MET_ADOPTION_READY", "OK")

    # ---- 缺区间：v2 起判不动（不得用点估计顶替）----
    ev("★坏例：缺区间 ⇒ 判不动（不能用点估计顶替）",
       10.0, None, 3, "INVALID_INPUT", "CI_MISSING")
    ev("坏例：区间长度 1 ⇒ 判不动", 10.0, [5.0], 3, "INVALID_INPUT", "CI_MALFORMED")
    ev("坏例：区间端点 NaN ⇒ 判不动", 10.0, [float("nan"), 20.0], 3,
       "INVALID_INPUT", "CI_NONFINITE")
    ev("坏例：区间下界 > 上界 ⇒ 判不动", 10.0, [20.0, 5.0], 3,
       "INVALID_INPUT", "CI_INVERTED")
    ev("坏例：省时点估计 NaN ⇒ 判不动", float("nan"), [5.0, 20.0], 3,
       "INVALID_INPUT", "SAVINGS_NONFINITE")
    ev("坏例：省时点估计 inf ⇒ 判不动", float("inf"), [5.0, 20.0], 3,
       "INVALID_INPUT", "SAVINGS_NONFINITE")

    # ---- 运行/精度/样本量 ----
    ev("坏例：运行无效 ⇒ 判不动（计时不可信）",
       10.0, [5.0, 20.0], 3, "INVALID_INPUT", "RUN_INVALID", run_validity="INVALID")
    ev("坏例：配对身份冲突（同一 run id）⇒ 判不动",
       10.0, [5.0, 20.0], 3, "INVALID_INPUT", "PAIRING_IDENTITY_CONFLICT",
       b_id="SAME", c_id="SAME")
    ev("坏例：区间下界达标但 n=2 < 3 ⇒ 证据不足，不达标",
       19.606, [10.0, 30.0], 2, "PERFORMANCE_NOT_MET", "PAIRS_LT_MIN")
    ev("正例：--min-pairs 2 时 n=2 可达标（判据可配置）",
       19.606, [10.0, 30.0], 2, "PERFORMANCE_MET_ADOPTION_READY", "OK", min_pairs=2)

    # ---- 三类结果拆开：性能达标 ≠ 可采纳 ----
    ev("★正例：性能达标但精度未验证 ⇒ 达标·不可采纳",
       10.0, [6.0, 20.0], 3, "PERFORMANCE_MET_ADOPTION_BLOCKED", "ADOPTION_GATE_OPEN",
       accuracy_validity="UNVERIFIED")
    ev("★正例：性能达标但精度 FAIL ⇒ 达标·不可采纳",
       10.0, [6.0, 20.0], 3, "PERFORMANCE_MET_ADOPTION_BLOCKED", "ADOPTION_GATE_OPEN",
       accuracy_validity="FAIL")
    ev("★正例：性能达标但指标被哨兵值替代 ⇒ 达标·不可采纳",
       10.0, [6.0, 20.0], 3, "PERFORMANCE_MET_ADOPTION_BLOCKED", "ADOPTION_GATE_OPEN",
       sentinel=True)
    ev("坏例：run_validity 非法取值 ⇒ 判不动",
       10.0, [6.0, 20.0], 3, "INVALID_INPUT", "RUN_VALIDITY_UNKNOWN_VALUE",
       run_validity="MAYBE")

    # ---- 解析层：角色制 + 身份 ----
    d, p, o, sch, ids, rs = parse_summary(None)
    cases.append(("解析：缺 summary ⇒ SUMMARY_MISSING", rs == "SUMMARY_MISSING",
                  sch, rs or "-", None, "SUMMARY_MISSING"))
    d, p, o, sch, ids, rs = parse_summary({"schema": "p59ab.v1", "paired": {"diffs": []}})
    cases.append(("解析：schema 不认识 ⇒ SUMMARY_SCHEMA_UNKNOWN",
                  rs == "SUMMARY_SCHEMA_UNKNOWN", sch, rs or "-", None, "SUMMARY_SCHEMA_UNKNOWN"))
    d, p, o, sch, ids, rs = parse_summary({"schema": "p59ab.v2"})
    cases.append(("解析：无逐对数据 ⇒ SUMMARY_NO_PAIRS", rs == "SUMMARY_NO_PAIRS",
                  sch, rs or "-", None, "SUMMARY_NO_PAIRS"))
    d, p, o, sch, ids, rs = parse_summary(
        {"schema": "p59ab.v2", "baseline_run_id": "B9", "candidate_run_id": "C9",
         "paired": {"diffs": [{"A_ms": 100.0, "B_ms": 90.0}]}})
    cases.append(("解析：显式 run id 被原样取用",
                  rs is None and ids == ("B9", "C9") and p == 1, ids, rs or "-", ("B9", "C9"), "-"))

    # ---- 方向/分母：正号必须恒等于"候选更快" ----
    v_b, rb = per_pair_savings([{"A_ms": 100.0, "B_ms": 90.0}], DENOM_BASELINE)
    v_c, rc = per_pair_savings([{"A_ms": 100.0, "B_ms": 90.0}], DENOM_CANDIDATE)
    cases.append(("方向：基线作分母时 100→90 ⇒ +10%（正=候选更快）",
                  rb is None and abs(v_b[0] - 10.0) < 1e-9, v_b, rb or "-", [10.0], "-"))
    cases.append(("方向：候选作分母时 100→90 ⇒ +11.11%（正=候选更快，数值不同但同向）",
                  rc is None and abs(v_c[0] - 11.111111) < 1e-4, v_c, rc or "-", [11.111111], "-"))
    v0, r0 = per_pair_savings([{"A_ms": 0.0, "B_ms": 90.0}], DENOM_BASELINE)
    cases.append(("坏例：分母 baseline=0 ⇒ 判不动（零分母不得当通过）",
                  r0 == "SAVINGS_ZERO_DENOMINATOR" and v0 is None, v0, r0 or "-",
                  None, "SAVINGS_ZERO_DENOMINATOR"))

    # ---- ★ 角色制：`variant` 的 A/B 指的**是两个配置**，不等于"基线/候选" ----
    #   下例用交付主结论那一对的真实窗口中位数：A=418.6 ms（推荐配置）、B=521.45 ms（保底配置）
    syn = {"schema": "p59ab.v2",
           "runs": [{"tag": "r_A1", "variant": "A", "window_median_ms": 418.6},
                    {"tag": "r_B1", "variant": "B", "window_median_ms": 521.45}]}
    d1, _p1, _o1, _s1, _i1, r1 = parse_summary(syn, baseline_variant="A", candidate_variant="B")
    d2, _p2, _o2, _s2, _i2, r2 = parse_summary(syn, baseline_variant="B", candidate_variant="A")
    v1, _ = per_pair_savings(d1, DENOM_BASELINE)
    v2, _ = per_pair_savings(d2, DENOM_BASELINE)
    cases.append(("★角色：把 A 当基线（错配）⇒ 省时为负 -24.57%",
                  r1 is None and abs(v1[0] + 24.5702) < 0.01, round(v1[0], 4), r1 or "-", -24.5702, "-"))
    cases.append(("★角色：把 B 当基线（= 交付口径「配置 A 相对 B 省时」）⇒ 省时 +19.72%",
                  r2 is None and abs(v2[0] - 19.7163) < 0.01, round(v2[0], 4), r2 or "-", 19.7163, "-"))

    # ---- ★ 端到端：真的起子进程跑一遍，断言最终 JSON 与退出码（不只断言打印）----
    tmp = _e2e_tmpdir()
    npath = os.path.join(tmp, "noise.json")
    with open(npath, "w", encoding="utf-8") as fh:
        json.dump(_noise(1.53), fh, ensure_ascii=False)
    e2e = []
    for label, bad, want_rc, want_verdict in [
        ("端到端坏例：点估计 10%、区间跨零", True, EXIT_NOT_MET, "PERFORMANCE_NOT_MET"),
        ("端到端正例：区间下界 6%", False, EXIT_ADOPTION_BLOCKED,
         "PERFORMANCE_MET_ADOPTION_BLOCKED"),
    ]:
        args = [sys.executable, os.path.abspath(__file__), "--noise", npath,
                "--pairs", "3", "--accuracy-validity", "unverified"]
        if bad:
            args += ["--savings", "10", "--ci-low", "-14.84", "--ci-high", "34.84"]
        else:
            args += ["--savings", "10", "--ci-low", "6", "--ci-high", "20"]
        outp = os.path.join(tmp, "out_%d.json" % (1 if bad else 2))
        args += ["--out", outp]
        pr = subprocess.run(args, capture_output=True, text=True)
        got = None
        if os.path.exists(outp):
            try:
                got = json.load(open(outp, encoding="utf-8"))
            except Exception:
                got = None
        ok = (pr.returncode == want_rc and got is not None
              and got.get("verdict") == want_verdict
              and got.get("schema") == OUT_SCHEMA)
        e2e.append((label, ok, pr.returncode, want_rc,
                    (got or {}).get("verdict"), want_verdict))

    failed = [c for c in cases if not c[1]] + [c for c in e2e if not c[1]]
    shutil.rmtree(tmp, ignore_errors=True)
    for name, ok, got_v, got_r, want_v, want_r in cases:
        print("  %s %s" % ("PASS" if ok else "FAIL", name))
        if not ok:
            print("        got verdict=%s reason=%s | want verdict=%s reason=%s"
                  % (got_v, got_r, want_v, want_r))
    for name, ok, got_rc, want_rc, got_v, want_v in e2e:
        print("  %s %s" % ("PASS" if ok else "FAIL", name))
        if not ok:
            print("        got rc=%s verdict=%s | want rc=%s verdict=%s"
                  % (got_rc, got_v, want_rc, want_v))
    total, nfail = len(cases) + len(e2e), len(failed)
    if nfail:
        print("P62_SELFTEST_FAIL cases=%d failed=%d" % (total, nfail))
        return 1
    print("P62_SELFTEST_OK cases=%d failed=0" % total)
    return 0


def main():
    ap = argparse.ArgumentParser(description="性能达标闸门 v2：区间下界硬门 + 三类结果拆开")
    ap.add_argument("--noise", default=None, help="59 --null-test 产出的 null_summary.json")
    ap.add_argument("--summary", default=None, help="59/63 产出的 A/B summary.json")
    ap.add_argument("--savings", default=None, help="手工给省时点估计（正 = 候选更快）")
    ap.add_argument("--ci-low", default=None, dest="ci_low", help="手工给 95%% 区间下界")
    ap.add_argument("--ci-high", default=None, dest="ci_high", help="手工给 95%% 区间上界")
    ap.add_argument("--pairs", type=int, default=0, help="配对数")
    ap.add_argument("--denominator", default=DENOM_BASELINE,
                    choices=[DENOM_BASELINE, DENOM_CANDIDATE],
                    help="省时比例的分母（默认 baseline = 标准省时比例，也是本项目主口径）")
    ap.add_argument("--baseline-run-id", default=None, dest="baseline_run_id")
    ap.add_argument("--candidate-run-id", default=None, dest="candidate_run_id")
    ap.add_argument("--baseline-variant", default=None, dest="baseline_variant",
                    help="★ 显式声明哪个 variant 是**基线**（默认 A，但默认会被标记 roles_assumed）")
    ap.add_argument("--candidate-variant", default=None, dest="candidate_variant",
                    help="★ 显式声明哪个 variant 是**候选**（默认 B）")
    ap.add_argument("--run-validity", default="OK", dest="run_validity",
                    type=lambda s: s.upper(), choices=["OK", "INVALID", "UNVERIFIED"])
    ap.add_argument("--accuracy-validity", default="UNVERIFIED", dest="accuracy_validity",
                    type=lambda s: s.upper(), choices=["OK", "FAIL", "UNVERIFIED"])
    ap.add_argument("--sentinel-metric", action="store_true", default=False, dest="sentinel_metric",
                    help="指标被哨兵值替代（验收观测已失效）⇒ 阻止采纳")
    ap.add_argument("--official-comparable", action="store_true", default=False,
                    dest="official_comparable")
    ap.add_argument("--min-effect", type=float, default=DEFAULT_MIN_EFFECT_PCT, dest="min_effect",
                    help="达标门槛（%%，默认 " + str(DEFAULT_MIN_EFFECT_PCT)
                         + "；纪律「<5%% 的改善不可报告」）")
    ap.add_argument("--noise-multiplier", type=float, default=DEFAULT_NOISE_MULTIPLIER,
                    dest="noise_multiplier",
                    help="底噪倍数（默认 " + str(DEFAULT_NOISE_MULTIPLIER)
                         + "；**仅作工程筛选**，不替代区间下界）")
    ap.add_argument("--min-pairs", type=int, default=DEFAULT_MIN_PAIRS, dest="min_pairs",
                    help="可支撑数字结论的最小配对数（默认 " + str(DEFAULT_MIN_PAIRS) + "）")
    ap.add_argument("--out", default=None, help="判定产物 JSON")
    ap.add_argument("--selftest", action="store_true", default=False, help="判据自检（0-GPU）")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.noise:
        print("P62_USAGE 必须给 --noise <null_summary.json>（先量底噪）")
        return EXIT_USAGE
    if args.savings is None and not args.summary:
        print("P62_USAGE 必须给 --summary <summary.json> 或 --savings <百分点>")
        return EXIT_USAGE
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
