#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
judge_comparable.py — R2-SK04 · 可比性判定机（纯函数，来源 R2-C6-03）

功能
----
对"runA 指纹快照（fingerprint_cfg.py 输出）+ 可选实测指纹（fingerprint_observed.py 输出）+
可选数据双文件判定（data_id.py --compare 输出）"与目标基准（officialA/officialB）做
确定性裁决，输出：
    {配置偏离表, 数据同源判定, N_eff, 可达模式 pointwise|window|none, 基准 officiality,
     verdict_id, 证据门, 颜色/exit_hint, 人类可读 reasons}

纯函数契约（C6-03："验收结论 = 判定机对两份不可变 manifest 的确定性函数输出"）：
  * 无随机、无墙钟、无网络；输出只依赖输入文件【内容】；
  * verdict_id = sha256(基线 yaml 字节 + run 指纹 JSON 字节 [+ observed JSON 字节]
    [+ data_cmp JSON 字节]) —— 同一批不可变证据必得同一 verdict_id；
  * 报告/验收只准引用 verdict_id，禁止人工粘贴数字宣称 PASS（C6-03 纪律）。

裁决逻辑（与 fingerprint_cfg 同源：判定函数 import 自 fingerprint_cfg，不重复实现）：
  * 可达模式（spec 口径）：pointwise_feasible = 7 维全等 + N_eff 同；
    window_feasible = GBS_eff=8；否则 none。
  * 证据门（gates）把"逐点 PASS 可宣称性"与"仅窗口口径"分开：
    - data_identity：run 与 baseline 数据字节身份未核对（官方文件不可得）→ 未闭合；
    - sample_order：shuffle 未关 或 数据字节未核 → 每步 batch 组成未证实；
    - kernel_evidence：声明 triton 但日志 roll back to CPU → 数值路径与声明不符；
    - loss_band：run 实测 step1 带 vs 基线带（A/B）不一致 → 量级不可比（compare.py
      provenance 守卫同口径，LOSS_BAND_SPLIT=5.0）；
    - lr_seq：observed 对账显示与参考 CSV 不一致时挂出。
  * 颜色/裁决（最终 verdict 五态；--gate 时按色退出）：
    POINTWISE_OK(绿/exit0)  : level=pointwise 且全部证据门闭合
    WINDOW_OK(黄/exit3)     : level=window（或 pointwise 候选但有未闭合门）→ 只许窗口均值口径
    NOT_COMPARABLE(红/exit4): level=none，或 loss_band 门确认冲突（不同带不喂 S-M3）
    NEEDS_EVIDENCE(黄/exit3): 候选可行但关键证据缺失（无 observed/数据身份），不许宣称 PASS
    INVALID_INPUT(2)        : 参数/IO/一致性错误
  默认退出码 0 = 裁决成功产出（红/黄也属成功产出，exit_hint 在 JSON/摘要里）；--gate 才按色退出。

纯 CPU；依赖 fingerprint_cfg.py（同目录）+ PyYAML（基线 yaml）+ 标准库。不依赖 torch/A2。

用法
----
  python3 scripts/judge_comparable.py \
      --run <fingerprint_cfg.json> \
      [--observed <fingerprint_observed.json>] \
      [--data-cmp <data_id --compare 输出 json>] \
      [--baseline officialB] [--out verdict.json] [--gate]
  # 一行摘要：JUDGE_OK verdict=WINDOW_OK level=window_feasible officiality=proposed
  #           deviations=9 data_same=unverified neff=8/8 verdict_id=…
  # 退出码：默认 0 = 裁决成功产出（红/黄也属成功产出，exit_hint 只在 JSON/摘要里）；
  #        --gate 时按裁决退出：0=绿(POINTWISE_OK)/3=黄(WINDOW_OK|NEEDS_EVIDENCE)/4=红
  #        (NOT_COMPARABLE)；2=输入/一致性错误。
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fingerprint_cfg as fpc  # 判定核心与 fingerprint_cfg 同源（防口径漂移）


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_or_none(p):
    if not p:
        return None
    path = Path(p)
    return path if path.is_file() else None


def canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def band_name(b):
    return {"A": "A(从零/随机初始化尺度 loss≈12)", "B": "B(预训练续训尺度 loss≈1.4)"}.get(b, "?")


def load_inputs(args):
    run_path = file_or_none(args.run)
    if run_path is None:
        raise FileNotFoundError("--run fingerprint_cfg.json 不存在: %s" % args.run)
    run = json.loads(run_path.read_text(encoding="utf-8-sig"))
    if run.get("schema") != "sk04_fingerprint_cfg.v1":
        raise ValueError("--run 不是 sk04_fingerprint_cfg.v1 输出")
    observed = obs_path = data_cmp = data_cmp_path = None
    if args.observed:
        obs_path = file_or_none(args.observed)
        if obs_path is None:
            raise FileNotFoundError("--observed 不存在: %s" % args.observed)
        observed = json.loads(obs_path.read_text(encoding="utf-8-sig"))
    if args.data_cmp:
        data_cmp_path = file_or_none(args.data_cmp)
        if data_cmp_path is None:
            raise FileNotFoundError("--data-cmp 不存在: %s" % args.data_cmp)
        data_cmp = json.loads(data_cmp_path.read_text(encoding="utf-8-sig"))
    return run, observed, data_cmp, run_path, obs_path, data_cmp_path


def re_derive(run, baseline_ref):
    """用指纹快照里记录的 run 侧值 + 基线 yaml，重新做一遍偏离表与可达性
    （判定不信任快照里的 equal/level 字段，全部重算，保证纯函数自洽）。"""
    base_doc, base_path = fpc.load_baseline(baseline_ref)
    if base_doc.get("baseline_id") != run["baseline"]["id"]:
        raise ValueError("--baseline(%s) 与 --run 快照里记录的 baseline(%s) 不一致"
                         % (base_doc.get("baseline_id"), run["baseline"]["id"]))
    base_fp = base_doc["fingerprint"]
    base_derived = base_doc["derived"]
    rows = []
    for spec in fpc.DIM_SPECS:
        run_rec = next((r for r in run["dims"] if r["dim"] == spec["id"]), None)
        if run_rec is None:
            raise ValueError("快照缺维度 %s" % spec["id"])
        run_val = run_rec["run"]["value"]
        run_present = run_rec["run"]["present"]
        base_entry = base_fp.get(spec["id"], {})
        base_val = base_entry.get("value")
        fake_run_rec = {"value": run_val, "present": run_present}
        equal, status, detail = fpc.cmp_values(spec["kind"], fake_run_rec, base_val)
        rows.append({"dim": spec["id"], "label": spec["label"],
                     "run": run_rec["run"],
                     "baseline": {"value": base_val,
                                  "evidence": [{"file": e.get("file"), "line": e.get("line")}
                                               for e in base_entry.get("evidence", [])]},
                     "equal": equal, "status": status, "detail": detail})
    # 派生重算（N_eff / GBS）
    neff_run = run["derived"]["N_eff"]["run"]
    neff_base = base_derived.get("n_eff", {}).get("value")
    gbs_run = run["derived"]["GBS_eff"]["run"]
    gbs_base = base_derived.get("gbs_eff")
    derived = {"N_eff": {"run": neff_run, "baseline": neff_base,
                         "equal": neff_run is not None and neff_run == neff_base},
               "GBS_eff": {"run": gbs_run, "baseline": gbs_base,
                           "equal": gbs_run is not None and gbs_run == gbs_base}}
    return base_doc, rows, derived


def data_same_source(run, data_cmp):
    """数据同源判定：优先 data_id 双文件实测；否则用快照 data_identity + 声明字节。"""
    if data_cmp is not None:
        st = data_cmp.get("status")
        if st in ("same_bytes", "same_content_same_order", "same_content_reordered"):
            return {"same_source": True, "status": st,
                    "detail": "data_id --compare 实测：%s" % data_cmp.get("three_state")}
        return {"same_source": False, "status": st,
                "detail": "data_id --compare：%s（%s）" % (data_cmp.get("three_state"), data_cmp.get("detail", ""))}
    di = run.get("data_identity", {})
    base_file = None
    for d in run.get("dims", []):
        if d["dim"] == "dataset_file":
            base_file = d["baseline"]["value"]
    if isinstance(base_file, dict):
        base_file = base_file.get("file")
    rb = run.get("dataset_basename_run")
    bb = Path(str(base_file)).name if base_file else None
    if di.get("status") == "same_bytes":
        return {"same_source": True, "status": "same_bytes",
                "detail": "run 数据集与基线声明字节一致"}
    if rb and bb and rb == bb:
        return {"same_source": "unverified", "status": "declared_same_file_byte_unverified",
                "detail": "basename 相同（%s）但基线文件字节不可得 → 同源未证实（诚实：标 unresolved）" % rb}
    return {"same_source": False, "status": "different_file",
            "detail": "数据集文件不同（run=%s vs baseline=%s）" % (rb, bb)}


def gates_of(run, rows, derived, observed, data_same):
    gates = []
    # 1) 数据身份
    if data_same.get("same_source") is True:
        gates.append({"gate": "data_identity", "closed": True, "detail": data_same["detail"]})
    elif data_same.get("same_source") == "unverified":
        gates.append({"gate": "data_identity", "closed": False, "detail": data_same["detail"]})
    else:
        gates.append({"gate": "data_identity", "closed": False, "detail": data_same["detail"]})
    # 2) 样本序
    sh = next((r for r in rows if r["dim"] == "shuffle"), None)
    sh_off = sh is not None and sh["run"]["value"] is False
    if sh_off and data_same.get("same_source") is True:
        gates.append({"gate": "sample_order", "closed": True,
                      "detail": "shuffle=False + 数据同源 → 声明层同序（逐步 batch 建议 sampler_probe 复核）"})
    else:
        gates.append({"gate": "sample_order", "closed": False,
                      "detail": "shuffle=%s / 数据同源=%s → 每步 batch 组成未证实相同"
                      % (sh["run"]["value"] if sh else "?", data_same.get("status"))})
    # 3) kernel 证据（声明 triton 但回落 → 数值路径矛盾）
    if observed is not None:
        kern = (observed.get("observed") or {}).get("kernel", {})
        decl_gdn = None
        for d in rows:
            if d["dim"] == "gdn_causal" and isinstance(d["run"]["value"], dict):
                decl_gdn = d["run"]["value"].get("gdn")
        if decl_gdn == "triton" and kern.get("effective") is False:
            gates.append({"gate": "kernel_evidence", "closed": False,
                          "detail": "声明 gdn=triton 但日志实测 roll back to CPU（%d 处）→ triton 未上 NPU，"
                          "数值路径与声明不符（R2-SK03 前置：升 3.2.1）" % len(kern.get("rollback_to_cpu", []))})
        elif decl_gdn == "triton" and kern.get("effective") is None:
            gates.append({"gate": "kernel_evidence", "closed": False,
                          "detail": "声明 gdn=triton 但日志无 NPU triton 生效行 → 生效证据缺失"})
        else:
            gates.append({"gate": "kernel_evidence", "closed": True,
                          "detail": "kernel 证据核验通过（effective=%s, declared=%s）"
                          % (kern.get("effective_status"), decl_gdn)})
    else:
        gates.append({"gate": "kernel_evidence", "closed": False,
                      "detail": "未提供 --observed（无法核验 kernel 生效/回落）"})
    # loss_band 门在 finalize() 里追加（需要基线 derived 的 loss_band + observed step1）
    return gates


def finalize(run, rows, derived, data_same, base_doc, observed, gates):
    """颜色/裁决五态 + reasons + verdict_id。"""
    bad = [r for r in rows if r["status"] == "mismatch"]
    unk = [r for r in rows if r["status"] == "unverified"]
    neff_eq = derived["N_eff"]["equal"]
    gbs_eq = derived["GBS_eff"]["equal"]
    base_derived = base_doc["derived"]

    # ---- 可达模式（重算，spec 口径）
    if not bad and not unk and neff_eq:
        level = "pointwise_feasible"
        level_rule = "7 维全等 + N_eff 同（%s）" % derived["N_eff"]["run"]
    elif gbs_eq and derived["GBS_eff"]["run"] == 8:
        level = "window_feasible"
        level_rule = "GBS_eff=8（run=基线=8）→ 仅窗口均值口径（逐点不可达/未证实）"
    else:
        level = "none"
        level_rule = "N_eff/GBS 不可比（run=%s, baseline=%s）" % (derived["N_eff"]["run"],
                                                               derived["N_eff"]["baseline"])

    # ---- loss 带门（基线带 vs observed 实测带；LOSS_BAND_SPLIT=5.0 与 compare.py 同口径）
    base_band = base_derived.get("loss_band")
    obs_band = None
    s1 = {}
    if observed is not None:
        s1 = (observed.get("observed") or {}).get("step1_lr0", {})
        if s1.get("loss") is not None:
            obs_band = "A" if s1["loss"] >= 5.0 else "B"
    band_gate = {"gate": "loss_band", "closed": True,
                 "detail": "带判定: baseline=%s" % band_name(base_band)}
    if obs_band is not None:
        if obs_band == base_band:
            band_gate = {"gate": "loss_band", "closed": True,
                         "detail": "带一致：run step1 实测 %s(%s) == baseline %s" % (obs_band, s1.get("loss"), base_band)}
        else:
            band_gate = {"gate": "loss_band", "closed": False,
                         "detail": "带冲突：run step1 实测 %s(%s) ≠ baseline %s(%s) → 数值尺度不可比，"
                         "逐点 <2%% 数学上不可达（compare.py provenance 守卫同口径，勿喂 S-M3）"
                         % (obs_band, s1.get("loss"), base_band, band_name(base_band))}
    else:
        band_gate = {"gate": "loss_band", "closed": False,
                     "detail": "未提供 --observed：run 起点带未实测（step1(lr=0) loss）→ 带一致未证实"}
    gates.append(band_gate)
    open_gates = [g["gate"] for g in gates if not g["closed"]]

    # ---- 颜色与裁决
    hard_open = [g for g in open_gates if g in ("kernel_evidence", "loss_band")]
    band_conflict = obs_band is not None and base_band is not None and obs_band != base_band
    if level == "none":
        color, verdict = "red", "NOT_COMPARABLE"
    elif band_conflict:
        color, verdict = "red", "NOT_COMPARABLE"   # 带冲突优先红（不喂 S-M3）
    elif level == "pointwise_feasible" and not open_gates:
        color, verdict = "green", "POINTWISE_OK"
    elif level == "pointwise_feasible":
        color, verdict = "yellow", "NEEDS_EVIDENCE"  # 逐点候选但证据门未闭合
    elif hard_open:
        color, verdict = "yellow", "NEEDS_EVIDENCE"  # 窗口口径但数值路径/带未证实
    else:
        color, verdict = "yellow", "WINDOW_OK"       # 仅窗口均值口径（data 序类门不影响窗口均值）
    exit_hint = {"green": 0, "yellow": 3, "red": 4}[color]

    deviation_table = [{"dim": r["dim"], "label": r["label"],
                        "run": r["run"]["value"], "baseline": r["baseline"]["value"],
                        "equal": r["equal"], "status": r["status"],
                        "run_config_line": r["run"].get("config_line"),
                        "baseline_evidence": r["baseline"]["evidence"]} for r in rows]

    reasons = []
    if bad:
        reasons.append("配置偏离 %d 项（%s）→ 逐点不可达；仅窗口口径候选（GBS=%s）"
                       % (len(bad), ",".join(r["dim"] for r in bad), derived["GBS_eff"]["run"]))
    if unk:
        reasons.append("未钉死维度 %d 项（%s）：单侧缺省/声明未定 → 不作等值断言"
                       % (len(unk), ",".join(r["dim"] for r in unk)))
    if open_gates:
        reasons.append("未闭合证据门: %s" % ", ".join(open_gates))
    if verdict == "POINTWISE_OK":
        reasons.append("逐点 <2%% 可宣称（7 维全等 + N_eff 同 + 证据门全闭）；数字口径见 S-M3 compare.py")
    elif verdict == "WINDOW_OK":
        reasons.append("只许窗口均值口径对照（S-M3 窗口统计），禁止宣称逐点 PASS")
    elif verdict == "NOT_COMPARABLE":
        reasons.append("不喂 S-M3：配置不可比或 loss 带冲突（compare.py provenance 守卫同口径）")
    elif verdict == "NEEDS_EVIDENCE":
        reasons.append("候选可比但证据不足：补齐 observed/data 字节/官方确认后再判")

    return {
        "level": level, "level_rule": level_rule,
        "color": color, "verdict": verdict, "exit_hint": exit_hint,
        "gates": gates, "open_gates": open_gates,
        "deviation_table": deviation_table,
        "reasons": reasons,
    }


def main():
    ap = argparse.ArgumentParser(description="R2-SK04 可比性判定机（纯函数）")
    ap.add_argument("--run", required=True, help="fingerprint_cfg.py 输出 JSON（runA 指纹快照）")
    ap.add_argument("--observed", default=None, help="fingerprint_observed.py 输出 JSON（可选）")
    ap.add_argument("--data-cmp", default=None, help="data_id.py --compare 输出 JSON（可选）")
    ap.add_argument("--baseline", default=None,
                    help="目标基线 officialA|officialB|<yaml>（缺省取快照里记录的 baseline）")
    ap.add_argument("--out", default=None, help="verdict JSON 输出路径（默认 stdout）")
    ap.add_argument("--gate", action="store_true",
                    help="按裁决退出：0=绿/3=黄/4=红；默认 0（裁决成功产出即 0）")
    args = ap.parse_args()

    try:
        run, observed, data_cmp, run_path, obs_path, data_cmp_path = load_inputs(args)
        baseline_ref = args.baseline or run["baseline"]["id"]
        base_doc, rows, derived = re_derive(run, baseline_ref)
        data_same = data_same_source(run, data_cmp)
        gates = gates_of(run, rows, derived, observed, data_same)
        res = finalize(run, rows, derived, data_same, base_doc, observed, gates)

        # verdict_id = 证据内容摘要（无墙钟）
        parts = [Path(base_doc["_file"]).read_bytes(), run_path.read_bytes()]
        if obs_path:
            parts.append(obs_path.read_bytes())
        if data_cmp_path:
            parts.append(data_cmp_path.read_bytes())
        verdict_id = sha256_bytes(b"|".join(parts))

        out = {
            "schema": "sk04_judge_verdict.v1",
            "verdict_id": verdict_id,
            "generated_by": "judge_comparable.py",
            "run": {"fingerprint_file": str(run_path), "baseline_id": base_doc.get("baseline_id")},
            "baseline": {"id": base_doc.get("baseline_id"),
                         "officiality": base_doc.get("officiality"),
                         "officiality_note": base_doc.get("officiality_note"),
                         "role": base_doc.get("role")},
            "config_deviation": {"count_mismatch": len([r for r in rows if r["status"] == "mismatch"]),
                                 "count_unverified": len([r for r in rows if r["status"] == "unverified"]),
                                 "table": res["deviation_table"]},
            "data_same_source": data_same,
            "n_eff": derived["N_eff"],
            "reachability": {"level": res["level"], "rule": res["level_rule"],
                             "color": res["color"], "verdict": res["verdict"],
                             "exit_hint": res["exit_hint"]},
            "officiality": base_doc.get("officiality"),
            "evidence_gates": res["gates"],
            "open_gates": res["open_gates"],
            "reasons": res["reasons"],
        }
    except FileNotFoundError as e:
        print("FATAL %s" % e, file=sys.stderr)
        return 2
    except ValueError as e:
        print("FATAL %s" % e, file=sys.stderr)
        return 2

    text = json.dumps(out, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    devs = out["config_deviation"]["count_mismatch"]
    print("JUDGE_OK verdict=%s level=%s color=%s officiality=%s deviations=%d unverified=%d "
          "data_same=%s neff=%s/%s open_gates=[%s] verdict_id=%s exit_hint=%d"
          % (out["reachability"]["verdict"], out["reachability"]["level"],
             out["reachability"]["color"], out["officiality"], devs,
             out["config_deviation"]["count_unverified"],
             out["data_same_source"].get("status"),
             out["n_eff"]["run"], out["n_eff"]["baseline"],
             ",".join(out["open_gates"]), verdict_id[:16],
             out["reachability"]["exit_hint"]))
    return out["reachability"]["exit_hint"] if args.gate else 0


if __name__ == "__main__":
    sys.exit(main())
