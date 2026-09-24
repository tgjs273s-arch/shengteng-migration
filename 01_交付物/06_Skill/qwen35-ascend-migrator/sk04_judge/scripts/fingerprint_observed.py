#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fingerprint_observed.py — R2-SK04 · 运行实测指纹（可比性机器化三件套之三，来源 R2-C6-03）

功能
----
从训练日志（MindSpeed-MM FSDP trainer 迭代行，R1-SK01/compare.py 同口径）提取【实测指纹】：
  1. 实际 GBS（每步 consumed samples 增量唯一值集 + 日志 global batch size 行）；
  2. LR 序列哈希（逐 iter 解析 learning rate → 规范序列 → sha256；可再与参考 CSV 的
     LR 列逐 iter 比对 --lr-ref-csv）；
  3. kernel 生效证据：检测 "roll back to CPU" / "not supported on current platform"（回落）
     与 "use NPU triton ops" / "use NPU triton fused ops"（生效标记），输出命中行号+原文；
  4. 最后 iter / 迭代行数 / traceback 数 / samples 终值；
  5. 墙钟：首末迭代行时间戳与耗时（分钟）；
  6. step1(lr=0) 采样（loss 带判定：≥5 → A 带(从零尺度) / <5 → B 带(预训练续训尺度)，
     与 compare.py LOSS_BAND_SPLIT 同口径）+ dcp checkpoint 加载证据行。
与声明指纹对账：--declared 传入 fingerprint_cfg.py 的 JSON 输出（或其子集）→
  逐项 reconcile（GBS 声明 vs 实测、gdn/causal 声明 vs kernel 证据、N_eff 声明 vs samples
  增量）→ 输出 observed_vs_declared.json（含 discrepancies 表）。

确定性：输出完全由日志/输入文件内容决定，不含墙钟/随机 —— 同输入字节必同输出。

纯 CPU；仅 Python 标准库。不依赖 torch/A2/pyyaml。

用法
----
  python3 scripts/fingerprint_observed.py --log runs/run_<tag>/train.log \
      [--declared <fingerprint_cfg.json>] [--lr-ref-csv <官方CSV(LR 列)>] \
      [--lr-ref-label officialB] [--out observed_vs_declared.json]
退出码：0=产出成功；2=参数/IO 错误；3=日志无可解析迭代行。
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------- 迭代行正则
# 与 skill_round2_SM3/compare.py 的 _ITER_RE 及 R1-SK01 parse_train_log.py 完全一致
# （三处同源，勿改，改动会口径漂移）。官方 2026-07-24 日志实例见 compare.py docstring。
_ITER_RE = re.compile(
    r"iteration\s+(?P<iter>\d+)\s*/\s*(?P<total>\d+)"
    r"\s*\|\s*consumed samples:\s*(?P<samples>\d+)"
    r"\s*\|\s*elapsed time per iteration \(ms\):\s*(?P<timems>[\d.]+)"
    r"\s*\|\s*learning rate:\s*(?P<lr>[\d.Ee+\-]+)"
    r"\s*\|\s*global batch size:\s*(?P<gbs>\d+)"
    r"\s*\|\s*loss:\s*(?P<loss>[\d.Ee+\-]+)"
    r"\s*\|\s*grad norm:\s*(?P<gradnorm>[\d.]+)"
    r"\s*\|"
)
_TS_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})")
_ROLLBACK_RES = [
    re.compile(r"roll\s*back\s*to\s*CPU", re.I),
    re.compile(r"not\s+supported\s+on\s+current\s+platform", re.I),
    re.compile(r"falling?\s*back\s*to\s*CPU", re.I),
    re.compile(r"triton[^\n]{0,80}CPU", re.I),
]
_NPU_TRITON_RE = re.compile(r"use\s+NPU\s+triton\s+(?:fused\s+)?ops", re.I)
_LOSS_BAND_SPLIT = 5.0     # 与 compare.py LOSS_BAND_SPLIT 同口径


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_lines(items) -> str:
    """规范字符串序列哈希（逐项 repr(float) 确定性）。"""
    body = "\n".join(repr(x) for x in items).encode("utf-8")
    return sha256_bytes(body)


def parse_log_rows(path):
    """返回 rows{iter: dict(含 line/ts)}。行号供证据指针。"""
    rows = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            m = _ITER_RE.search(line)
            if not m:
                continue
            it = int(m.group("iter"))
            rows[it] = {"Iter": it,
                        "Samples": int(m.group("samples")),
                        "TimeMs": float(m.group("timems")),
                        "LR": float(m.group("lr")),
                        "GBS": int(m.group("gbs")),
                        "Loss": float(m.group("loss")),
                        "GradNorm": float(m.group("gradnorm")),
                        "line": lineno,
                        "ts": _ts_of_line(line)}
    return rows


def _ts_of_line(line):
    m = _TS_RE.search(line)
    if not m:
        return None
    try:
        return datetime(*[int(x) for x in m.groups()])
    except Exception:
        return None


def kernel_evidence(lines):
    """扫描 kernel 生效/回落证据。返回 dict（命中行号从 1 起）。"""
    rollback, npu = [], []
    for i, line in enumerate(lines, 1):
        if any(r.search(line) for r in _ROLLBACK_RES):
            rollback.append({"line": i, "text": line.strip()[:200]})
        elif _NPU_TRITON_RE.search(line):
            npu.append({"line": i, "text": line.strip()[:200]})
    if rollback:
        effective, status = False, "rollback_observed"
    elif npu:
        effective, status = True, "npu_triton_active"
    else:
        effective, status = None, "no_kernel_marker"
    return {"rollback_to_cpu": rollback, "npu_triton": npu,
            "effective": effective, "effective_status": status}


def world_from_log(lines):
    """Configuration Details 区里的 world_size/data_parallel_size（首个迭代行之前的首次出现）。"""
    world = dp = None
    world_line = dp_line = None
    for i, line in enumerate(lines, 1):
        if _ITER_RE.search(line):
            break
        m = re.match(r"^\s*world_size:\s*(\d+)", line)
        if m and world is None:
            world, world_line = int(m.group(1)), i
        m = re.match(r"^\s*data_parallel_size:\s*(\d+)", line)
        if m and dp is None:
            dp, dp_line = int(m.group(1)), i
    return {"world_size": world, "world_line": world_line,
            "data_parallel_size": dp, "dp_line": dp_line}


def load_declared(json_path):
    """--declared：fingerprint_cfg.py 输出 JSON（取其 dims/derived 做对账源）。"""
    doc = json.loads(Path(json_path).read_text(encoding="utf-8-sig"))
    out = {"schema": doc.get("schema"), "path": str(json_path)}
    for d in doc.get("dims", []):
        out[d["dim"]] = {"value": d.get("run", {}).get("value"),
                         "present": d.get("run", {}).get("present", True)}
    out["n_eff"] = (doc.get("derived") or {}).get("N_eff", {}).get("run")
    out["gbs_eff"] = (doc.get("derived") or {}).get("GBS_eff", {}).get("run")
    return out


def load_ref_lr(csv_path, label=None):
    """参考 CSV（Iter,LR 列）→ {iter: lr}。兼容官方 A/B 两份 CSV。"""
    ref = {}
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            try:
                it = int(row["Iter"])
                lr = float(row["LR"])
            except Exception:
                continue
            ref[it] = lr
    return ref


def main():
    ap = argparse.ArgumentParser(description="R2-SK04 运行实测指纹 + 声明对账")
    ap.add_argument("--log", required=True, help="训练日志路径")
    ap.add_argument("--declared", default=None, help="fingerprint_cfg.py JSON（声明指纹）")
    ap.add_argument("--lr-ref-csv", default=None, help="参考 CSV（逐 iter LR 比对）")
    ap.add_argument("--lr-ref-label", default=None, help="参考 CSV 标签（如 officialB）")
    ap.add_argument("--out", default=None, help="JSON 输出路径（默认 stdout）")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.is_file():
        print("FATAL log_not_found %s" % log_path, file=sys.stderr)
        return 2
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = parse_log_rows(log_path)
    if not rows:
        print("FATAL no_iter_rows %s（无可解析迭代行，检查是否为 MindSpeed-MM trainer 日志）"
              % log_path, file=sys.stderr)
        return 3

    # ---- observed 组装
    iters = sorted(rows)
    sample_deltas = sorted({rows[iters[i]]["Samples"] - rows[iters[i - 1]]["Samples"]
                            for i in range(1, len(iters))})
    gbs_values = sorted({rows[i]["GBS"] for i in iters})
    lr_seq = [rows[i]["LR"] for i in iters]
    first_row, last_row = rows[iters[0]], rows[iters[-1]]
    wallclock = {"first_row_ts": first_row["ts"].strftime("%Y-%m-%d %H:%M:%S") if first_row["ts"] else None,
                 "last_row_ts": last_row["ts"].strftime("%Y-%m-%d %H:%M:%S") if last_row["ts"] else None,
                 "elapsed_min": None}
    if first_row["ts"] and last_row["ts"]:
        wallclock["elapsed_min"] = round((last_row["ts"] - first_row["ts"]).total_seconds() / 60.0, 1)

    # step1 采样（R1-SK01 口径：iter=1 且 lr==0 优先）
    s1 = rows.get(1)
    if s1 is not None and s1["LR"] != 0.0:
        lr0 = [rows[i] for i in iters if rows[i]["LR"] == 0.0]
        s1 = min(lr0, key=lambda r: r["Iter"]) if lr0 else s1
    step1 = {"iter": s1["Iter"], "lr": s1["LR"], "loss": s1["Loss"],
             "grad_norm": s1["GradNorm"], "line": s1["line"]}
    step1["loss_band"] = "A(从零尺度)" if s1["Loss"] >= _LOSS_BAND_SPLIT else "B(预训练续训尺度)"

    loaded_ckpt = [{"line": i + 1, "text": line.strip()[:200]}
                   for i, line in enumerate(lines) if "Loaded checkpoint from" in line]

    kern = kernel_evidence(lines)
    wld = world_from_log(lines)
    observed = {
        "iter_rows": {"count": len(rows), "min_iter": iters[0], "max_iter": iters[-1],
                      "contiguous": list(range(iters[0], iters[-1] + 1)) == iters,
                      "last_row_line": last_row["line"]},
        "gbs": {"values": gbs_values, "sample_delta_per_step": sample_deltas,
                "actual_gbs": gbs_values[0] if len(gbs_values) == 1 else None,
                "note": "global batch size 行与 consumed samples 增量（官方口径每步 +8）"},
        "lr": {"row_count": len(lr_seq), "first": lr_seq[0], "last": lr_seq[-1],
               "sha256": sha256_lines(lr_seq)},
        "kernel": kern,
        "world_from_log": wld,
        "last_iter": rows[iters[-1]]["Iter"],
        "samples_final": rows[iters[-1]]["Samples"],
        "traceback_count": sum(1 for l in lines if "Traceback (most recent call last)" in l),
        "wallclock": wallclock,
        "step1_lr0": step1,
        "loaded_checkpoint_evidence": loaded_ckpt,
    }

    # ---- 声明指纹加载（可选）
    declared = None
    if args.declared:
        try:
            declared = load_declared(args.declared)
        except Exception as e:
            print("WARN declared 加载失败（跳过对账）: %s" % e, file=sys.stderr)

    # ---- 对账 / discrepancies
    disc = []
    actual_gbs = observed["gbs"]["actual_gbs"]
    if declared is not None:
        decl_gbs = declared.get("gbs_eff")
        if decl_gbs is not None and actual_gbs is not None and decl_gbs != actual_gbs:
            disc.append({"item": "gbs", "declared": decl_gbs, "observed": actual_gbs,
                         "severity": "hard",
                         "detail": "声明 GBS_eff=%s ≠ 日志实测每步 %s 样本 → 声明与实测冲突" % (decl_gbs, actual_gbs)})
        decl_gdn = declared.get("gdn_causal", {}).get("value") if isinstance(declared.get("gdn_causal"), dict) else None
        if isinstance(decl_gdn, dict) and decl_gdn.get("gdn") == "triton":
            if kern["effective"] is False:
                disc.append({"item": "gdn_impl", "declared": "triton",
                             "observed": "roll back to CPU",
                             "severity": "hard",
                             "detail": "声明 gdn=triton 但日志出现 roll back to CPU（L%s）→ triton 未上 NPU（3.2.0 回落）"
                             % kern["rollback_to_cpu"][0]["line"] if kern["rollback_to_cpu"] else ""})
            elif kern["effective"] is None:
                disc.append({"item": "gdn_impl", "declared": "triton", "observed": "no_kernel_marker",
                             "severity": "warn",
                             "detail": "声明 gdn=triton 但日志无 NPU triton 生效行也无回落行 → 生效证据缺失（静默路径风险）"})
    if actual_gbs is not None and actual_gbs != 8:
        disc.append({"item": "gbs_official", "declared": 8, "observed": actual_gbs,
                     "severity": "hard",
                     "detail": "实测 GBS≠8 → 与官方参考 CSV（GBS=8）不可比（S-M3 红牌）"})

    # ---- LR vs 参考 CSV
    lr_cmp = None
    if args.lr_ref_csv:
        ref = load_ref_lr(args.lr_ref_csv)
        common = sorted(set(iters) & set(ref.keys()))
        diffs = [abs(rows[i]["LR"] - ref[i]) for i in common]
        lr_cmp = {"ref_label": args.lr_ref_label or Path(args.lr_ref_csv).name,
                  "iters_common": len(common), "iters_log": len(iters),
                  "max_abs_diff": (max(diffs) if diffs else None),
                  "seq_equal": all(abs(rows[i]["LR"] - ref[i]) < 1e-12 for i in common)
                  if common else False,
                  "note": "逐 iter 浮点解析比对（同解析口径）"}
        if lr_cmp["seq_equal"] is False and diffs:
            disc.append({"item": "lr_seq", "declared": lr_cmp["ref_label"],
                         "observed": "differs",
                         "severity": "warn",
                         "detail": "LR 序列与参考 %s 不一致（max_abs_diff=%s）→ 调度不同 → 曲线可比性存疑"
                         % (lr_cmp["ref_label"], lr_cmp["max_abs_diff"])})

    out = {
        "schema": "sk04_fingerprint_observed.v1",
        "generated_by": "fingerprint_observed.py",
        "log": {"path": str(log_path), "sha256": sha256_bytes(log_path.read_bytes()),
                "lines": len(lines)},
        "observed": observed,
        "declared": declared,
        "reconciliation": {"gbs_declared_vs_observed": ("match" if declared is None or actual_gbs is None
                                                        or declared.get("gbs_eff") == actual_gbs else "conflict"),
                           "lr_vs_ref_csv": lr_cmp},
        "discrepancies": disc,
    }
    text = json.dumps(out, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)

    print("OBSERVED_OK last_iter=%d rows=%d gbs=%s lr_sha256=%s kernel_status=%s "
          "rollback_lines=%d npu_triton_lines=%d wallclock_min=%s step1_loss=%s band=%s"
          % (observed["last_iter"], observed["iter_rows"]["count"], actual_gbs,
             observed["lr"]["sha256"][:12], kern["effective_status"],
             len(kern["rollback_to_cpu"]), len(kern["npu_triton"]),
             wallclock["elapsed_min"], step1["loss"], step1["loss_band"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
