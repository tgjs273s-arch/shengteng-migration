#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""55_validate_prediction.py — **预登记预测**的第 4 点验证（三点模型的可证伪性检验）

背景与动机（为什么必须有这一步）
--------------------------------
step54 用三点解三个未知量 `(a, f, F)` —— **三点三参数必然精确拟合，残差恒为 0**，
所以它本身**不能**证明模型对。按本项目纪律（INV-2：判据必须可证伪），
必须再用**一个新点**检验预测值。

预登记预测（由 step54 在 boot 7bcced6f… 上的解给出，先写死在此处，跑完再比对）
  a = 70.700 ms/样本   f = 146.900 ms/micro-batch   F = 3.000 ms/步
  D: mbs=8, gas=2（GBS=32）→ 预测  t_D = F + gas×(f + a×mbs) = 3.0 + 2×(146.9 + 565.6) = **1428.0 ms**
  容差：±10%（[1285.2, 1570.8] ms）

判据（跑之前就定好）
  · PASS：实测窗口 11–30 中位落在容差内 → 三点模型成立 → `f` 是真实可攻击的固定开销
  · FAIL 且实测明显更低 → 模型高估（f 有饱和/并行效应）→ "去掉 f 可得 1.5×"的推论要打折
  · FAIL 且实测明显更高 → 模型低估（存在非线性）→ 需重做标定，不得引用外推值

用法：python3 55_validate_prediction.py --out /root/ops/p2fit_D_20260917
"""
import argparse
import copy
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time

PAT = re.compile(r"iteration\s+(\d+)\s*/\s*\d+.*?elapsed time per iteration \(ms\):\s*([\d.]+)", re.S)
PRED = {"a": 70.700, "f": 146.900, "F": 3.000, "mbs": 8, "gas": 2, "tol_pct": 10.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", default="/root/qwen35-ascend-migrator")
    ap.add_argument("--config", default="/root/qwen35-ascend-migrator/out/plan/train_config.yaml")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", default="/root/ops/p2fit_D")
    ap.add_argument("--mind", default="/root/MindSpeed-MM")
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()

    if os.path.exists(args.out):
        print("FATAL 输出目录已存在（拒绝覆盖）：%s" % args.out)
        return 3
    import yaml
    doc = yaml.safe_load(open(args.config, encoding="utf-8"))
    dp = int(doc["parallel"]["data_parallel_size"])
    if dp != 2:
        print("FATAL 需 dp=2（当前 %d）" % dp)
        return 3
    pred = PRED["F"] + PRED["gas"] * (PRED["f"] + PRED["a"] * PRED["mbs"])
    lo, hi = pred * (1 - PRED["tol_pct"] / 100.0), pred * (1 + PRED["tol_pct"] / 100.0)

    os.makedirs(args.out)
    cfg = copy.deepcopy(doc)
    cfg["training"]["train_iters"] = args.steps
    cfg["training"]["micro_batch_size"] = PRED["mbs"]
    cfg["training"]["gradient_accumulation_steps"] = PRED["gas"]
    cfg["training"]["save"] = os.path.join(args.out, "checkpoint")
    text = yaml.safe_dump(cfg, sort_keys=False)
    back = yaml.safe_load(text)["training"]
    assert back["micro_batch_size"] == PRED["mbs"] and back["gradient_accumulation_steps"] == PRED["gas"]
    config = os.path.join(args.out, "config.yaml")
    open(config, "w", encoding="utf-8").write(text)

    log, drv = os.path.join(args.out, "train.log"), os.path.join(args.out, "driver.log")
    env_json = os.path.join(args.skill, "out", "probe", "env.json")
    boot = open("/proc/sys/kernel/random/boot_id").read().strip()
    print("D_START point=mbs%d_gas%d_gbs%d predicted=%.1f ms tol=[%.1f, %.1f] boot=%s"
          % (PRED["mbs"], PRED["gas"], PRED["mbs"] * PRED["gas"] * dp, pred, lo, hi, boot), flush=True)
    cmd = [sys.executable, os.path.join(args.skill, "scripts", "50_train.py"),
           "--config", config, "--env", env_json, "--log", log, "--workdir", args.mind,
           "--steps", str(args.steps), "--world-size", "2", "--timeout", "900",
           "--foreground", "--allow-gbs-mismatch"]
    t0 = time.time()
    with open(drv, "w", encoding="utf-8") as fh:
        r = subprocess.run(cmd, cwd=args.skill, env=os.environ.copy(),
                           stdout=fh, stderr=subprocess.STDOUT, timeout=args.timeout)
    wall = time.time() - t0
    dtext = open(drv, encoding="utf-8", errors="replace").read()
    ttext = open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else ""
    rows = [(int(i), float(ms)) for i, ms in PAT.findall(ttext)]
    train_rc = re.findall(r"train_rc=(\d+)", dtext)
    valid = (r.returncode == 0 and train_rc == ["0"] and len(rows) == args.steps)
    win = [ms for i, ms in rows if 11 <= i <= args.steps]
    med = statistics.median(win) if win else None
    ok = bool(valid and med is not None and lo <= med <= hi)
    rec = {"schema": "p2fit_pointD.v1", "valid": valid, "driver_rc": r.returncode, "train_rc": train_rc,
           "steps_parsed": len(rows), "wall_seconds": round(wall, 1),
           "window": [11, args.steps], "window_median_ms": med,
           "iter_ms_all": [ms for _, ms in rows],
           "predicted_ms": round(pred, 3), "tol_band_ms": [round(lo, 3), round(hi, 3)],
           "prediction_pass": ok,
           "prediction_inputs": PRED,
           "source_sha256": hashlib.sha256(
               open(os.path.join(args.skill, "scripts", "50_train.py"), "rb").read()).hexdigest(),
           "identity": {"boot": boot, "hostname": __import__("platform").node() or "unknown",  # 坑 154
                        "affinity": sorted(os.sched_getaffinity(0))}}
    json.dump(rec, open(os.path.join(args.out, "result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("D_RESULT valid=%s median=%.1f ms predicted=%.1f ms verdict=%s wall=%.1fs"
          % (valid, med if med else -1, pred, "PASS" if ok else "FAIL", wall), flush=True)
    print("P2D_%s out=%s" % ("OK" if ok else "FAIL", args.out), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
