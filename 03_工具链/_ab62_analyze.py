#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""§4-2 A/B **本地判据**：解析与统计一律用**交付副本**的实现（远端只负责跑）。

为什么这样分工：远端 `/root/qwen35-ascend-migrator/scripts/59_config_ab.py` 是旧版
（md5 `218e635b…`，无 `win_ms_of`），而 `paired_stats` 依赖它 ⇒ 若在远端统计，
**判据语义会随副本版本漂移**。所以：
  远端 = 跑 + 记录原始日志；本地 = 用交付副本的 `PAT`/`compute_valid`/`paired_stats` 解析统计。

产出（供 `62_reportability.py` 直接消费）：
  null_summary.json          —— N 臂重复基线（p59null.v1）
  C1_vs_N__summary.json      —— （p59ab.v2）
  C2_vs_N__summary.json      —— （p59ab.v2）

用法：
  python _ab62_analyze.py <runs_index.json> <输出目录>
"""
import glob
import importlib
import json
import os
import re
import statistics
import sys

SKILL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "qwen35-ascend-migrator_整合版")
sys.path.insert(0, os.path.join(SKILL, "scripts"))
_p59 = importlib.import_module("59_config_ab")      # ★ 交付副本，单一来源
PAT = _p59.PAT
compute_valid = _p59.compute_valid
paired_stats = _p59.paired_stats

WINDOW_LO = 11      # 与 59 的窗口约定一致：11..steps（前 10 步不纳入中位数）


def parse_log(path, steps):
    text = open(path, encoding="utf-8", errors="replace").read()
    rows = [(int(a), float(b), float(c), float(d)) for a, b, c, d in PAT.findall(text)]
    ids = [r[0] for r in rows]
    win = [ms for s, ms, _lo, _gn in rows if WINDOW_LO <= s <= steps]
    return rows, ids, (statistics.median(win) if win else None)


def resolve_local(local_dir, remote_path):
    """把远端路径映射到 pull_evidence 落盘后的本地名字。

    `pull_evidence` 会用父目录消歧（`<祖父>__<父>__<basename>`），因此这里按
    "末两段 + basename" 去 glob，并要求**唯一命中**（歧义即失败，不猜）。
    """
    parts = remote_path.strip("/").split("/")
    base = parts[-1]
    tag = parts[-2] if len(parts) >= 2 else None
    pats = []
    if tag:
        pats.append(os.path.join(local_dir, "*__%s__%s" % (tag, base)))
    pats.append(os.path.join(local_dir, "*__%s" % base))
    pats.append(os.path.join(local_dir, base))
    for pat in pats:
        hits = sorted(glob.glob(pat))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1 and tag:
            exact = [h for h in hits if ("__%s__" % tag) in os.path.basename(h)]
            if len(exact) == 1:
                return exact[0]
    return None


def main():
    if len(sys.argv) < 4:
        print("USAGE: _ab62_analyze.py <runs_index.json> <outdir> <local_logs_dir>")
        return 2
    index_path, outdir, local_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(outdir, exist_ok=True)
    idx = json.load(open(index_path, encoding="utf-8-sig"))
    steps = idx["steps"]
    print("ANALYZE out=%s steps=%d arms=%d" % (idx["out"], steps, len(idx["runs"])))

    runs = []
    for r in idx["runs"]:
        logp = resolve_local(local_dir, r["train_log"])
        if logp is None:
            print("ANALYZE_FAIL 解析不到本地日志：%s（拒绝继续）" % r["train_log"])
            return 1
        rows, ids, med = parse_log(logp, steps)
        drvp = resolve_local(local_dir, r["driver_log"])
        if drvp is None:
            print("ANALYZE_FAIL 解析不到本地 driver.log：%s（拒绝继续：run_valid 要求 train_rc==['0']）"
                  % r["driver_log"])
            return 1
        dtext = open(drvp, encoding="utf-8", errors="replace").read()
        train_rc = re.findall(r"train_rc=(\d+)", dtext)
        valid, steps_complete = compute_valid(r["rc"], train_rc, ids, steps)
        rec = {
            "tag": r["tag"], "arm": r["arm"], "MM_CAND": r["MM_CAND"],
            "variant": "A" if r["arm"] == "N" else "B",     # 62 按 variant 分臂
            "valid": bool(valid), "steps_complete": bool(steps_complete),
            "window_median_ms": med, "wall_seconds": r["wall_seconds"],
            "rc": r["rc"], "rows": len(rows), "step_ids_tail": ids[-5:],
            "loss_first": rows[0][2] if rows else None,
            "gradnorm_first": rows[0][3] if rows else None,
            "loss_head": [round(x[2], 9) for x in rows[:6]],
            "gradnorm_head": [round(x[3], 6) for x in rows[:6]],
            "identity": r.get("identity", {}),
        }
        runs.append(rec)
        print("  %-8s arm=%-3s valid=%-5s steps_complete=%-5s win_med=%-10s rows=%d gn1=%s"
              % (rec["tag"], rec["arm"], rec["valid"], rec["steps_complete"],
                 rec["window_median_ms"], rec["rows"], rec["gradnorm_first"]))

    invalid = [r for r in runs if not r["valid"]]
    if invalid:
        print("ANALYZE_FAIL 有 %d 个无效运行 ⇒ 拒绝出统计（fail-closed）: %s"
              % (len(invalid), [r["tag"] for r in invalid]))
        return 1

    # ---- 噪声底噪：全部来自 N 臂（同批次） ----
    N = [r for r in runs if r["arm"] == "N"]
    meds = [r["window_median_ms"] for r in N if r["window_median_ms"] is not None]
    if len(meds) < 2:
        print("ANALYZE_FAIL N 臂有效中位数不足 2 个 ⇒ 量不出底噪")
        return 1
    spread = (max(meds) - min(meds)) / min(meds) * 100
    null = {"schema": "p59null.v1", "runs": len(meds), "medians_ms": meds,
            "mean_ms": round(sum(meds) / len(meds), 2), "min_ms": min(meds), "max_ms": max(meds),
            "spread_pct": round(spread, 2),
            "resolution_note": ("同批次基线重复 %d 次的最大相对差 = %.2f%%；"
                                "A/B 差异小于它则不可判定。" % (len(meds), spread)),
            "source": "§4-2 A/B 的 N 臂（MM_CAND 未设，env 门控未进入）",
            "identity": N[0].get("identity", {})}
    with open(os.path.join(outdir, "null_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(null, fh, ensure_ascii=False, indent=1)
    print("NULL_RESULT n=%d medians=%s spread_pct=%.2f" % (len(meds), meds, null["spread_pct"]))

    # ---- 每臂 vs N ----
    for arm in ("C1", "C2"):
        B = [r for r in runs if r["arm"] == arm]
        if not B:
            continue
        pairs = paired_stats(N, B, "A,B")        # ★ 交付副本的成对统计
        d = {"schema": "p59ab.v2", "key": "MM_CAND=%s" % arm,
             "official_comparable": False,
             "comparability": {"reason": "同机同批次、mock 数据（official_comparable=false）"},
             "order_used": "latin-square(N,C1,C2 / C1,C2,N / C2,N,C1)",
             "steps": steps, "n_A": len(N), "n_B": len(B),
             "paired": pairs, "runs": N + B,
             "verdict": "PENDING_LOCAL_62",
             "verdict_reason": "由 62_reportability.py 按冻结判据判定（阈值=max(2×底噪,5%)）"}
        with open(os.path.join(outdir, "%s_vs_N__summary.json" % arm), "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=1)
        print("AB62_%s pairs=%s mean_delta_pct=%s improvement_pct=%s ci95=%s"
              % (arm, pairs["pairs"], pairs["mean_delta_pct"], pairs["improvement_pct"],
                 pairs["ci95_delta_pct"]))

    # ---- 注入自证：C1 臂的 grad_norm 应被改写为 0.0；C2 臂的 loss 应与基线比有(无)差异 ----
    print("INJECTION_CHECK markers=%s" % json.dumps(idx.get("markers", {})))
    for r in runs:
        print("  %-8s arm=%-3s gradnorm_head=%s" % (r["tag"], r["arm"], r["gradnorm_head"][:3]))
    with open(os.path.join(outdir, "runs_parsed.json"), "w", encoding="utf-8") as fh:
        json.dump({"runs": runs, "markers": idx.get("markers", {})}, fh,
                  ensure_ascii=False, indent=1)
    print("ANALYZE_OK out=%s" % outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
