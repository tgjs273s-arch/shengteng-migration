#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""54_three_point_fit.py — 快时段"每步固定开销 / 每 micro-batch 开销 / 每样本计算"三点分离

为什么做这个
------------
A1（9/16 慢时段）测得 `步时长 = F + n·a`，F≈171 ms/步、a≈94.3 ms/样本，
但按本项目纪律**不可跨时段外推**。要在快时段（boot 7bcced6f…）判断"余量在固定开销还是每样本计算"，
必须重做，而且要用**能区分三种开销**的设计，而不是两点了事。

三点设计（dp=2 固定；每步每卡样本数 S = mbs × gas）
--------------------------------------------------
  A  mbs=4 gas=1 → S=4  GBS=8    ← **官方几何**（唯一可参与精度/性能对标的点）
  B  mbs=8 gas=1 → S=8  GBS=16   ← 需 --allow-gbs-mismatch（**仅测性能性质**）
  C  mbs=4 gas=2 → S=8  GBS=16   ← 与 B 同为 GBS=16，但每步有 **2 个 micro-batch**

模型：`t = F_opt + gas × (f + a × mbs)`
  A: F + 1×(f+4a)   B: F + 1×(f+8a)   C: F + 2×(f+4a)
⇒ `a = (tB − tA)/4`；`f = (tC − tA) − 4a`；`F = tA − (f + 4a)`

**判据（先定好，避免事后编故事）**
  · `f ≥ 60 ms`（每个 micro-batch 的固定开销很大）→ 下发/建图/host 侧为主 → **图捕获/融合有余量**
  · `f < 30 ms` 且 a 主导 → 余量在算 → 图捕获帮不上，要走 kernel 级优化
  · `F_opt` 大（≥80 ms）→ 每步的优化器/通信固定成本高 → 走通算重叠

纪律
----
* 输出目录**必须不存在**（拒绝覆盖；长任务产物是最贵的证据）。
* 三点缺一即 FAIL；每组必须有退出证据（driver_rc=0 且 train_rc=['0']）；解析步数必须等于 --steps。
* GBS≠8 的点**必须**显式 --allow-gbs-mismatch，并在结果里带 `official_comparable: false`。
* 采样间隔目标 0.5 s（此前 4 s 与步长 0.43 s 混叠，AICore 读数不可当稳态值，这次把它压下去）。

用法（真机）：
  python3 54_three_point_fit.py --skill /root/qwen35-ascend-migrator \
      --config /root/qwen35-ascend-migrator/out/plan/train_config.yaml \
      --steps 30 --out /root/ops/p2fit_20260917
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
import threading
import time

PAT = re.compile(r"iteration\s+(\d+)\s*/\s*\d+.*?"
                 r"elapsed time per iteration \(ms\):\s*([\d.]+)", re.S)
ENVLIB = ("export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:"
          "/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:"
          "$LD_LIBRARY_PATH; ")
CASES = [("A_official_mbs4_gas1", 4, 1, True),
         ("B_mbs8_gas1", 8, 1, False),
         ("C_mbs4_gas2", 4, 2, False)]
SAMPLE_INTERVAL = 0.5


def sh(cmd, timeout=60):
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    return (r.stdout or "") + (r.stderr or "")


def npu_id():
    txt = sh(ENVLIB + "npu-smi info -l 2>&1")
    m = re.search(r"NPU ID\s*:\s*(\d+)", txt)
    if not m:
        raise RuntimeError("取不到 NPU ID: " + txt[:200])
    return m.group(1)


def sample_once(nid):
    u = sh(ENVLIB + "npu-smi info -t usages -i %s -c 0 2>&1" % nid)

    def grab(key):
        m = re.search(r"%s\s*:\s*([\d.]+)" % re.escape(key), u)
        return float(m.group(1)) if m else None

    return {"t": time.time(), "aicore": grab("Aicore Usage Rate(%)"),
            "aivector": grab("Aivector Usage Rate(%)"),
            "hbm_bw": grab("HBM Bandwidth Usage Rate(%)"),
            "npu_util": grab("NPU Utilization(%)")}


def fit(tA, tB, tC):
    """由三点解出 (a, f, F)。返回 ms。"""
    a = (tB - tA) / 4.0
    f = (tC - tA) - 4.0 * a
    F = tA - (f + 4.0 * a)
    return a, f, F


def selftest():
    """合成数据正例 + 坏例：可证伪（INV-2）。"""
    a, f, F = 10.0, 50.0, 100.0
    tA = F + (f + 4 * a)          # 190
    tB = F + (f + 8 * a)          # 230
    tC = F + 2 * (f + 4 * a)      # 280
    ra, rf, rF = fit(tA, tB, tC)
    assert abs(ra - a) < 1e-9 and abs(rf - f) < 1e-9 and abs(rF - F) < 1e-9, (ra, rf, rF)
    bad = copy.deepcopy(CASES)
    bad[0] = ("A_official_mbs4_gas1", 4, 2, True)      # 官方点被改坏（GBS=16）
    try:
        check_cases(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("坏几何被接受")
    try:
        check_cases([CASES[0], CASES[1]])              # 缺点
    except AssertionError:
        pass
    else:
        raise AssertionError("缺点被接受")
    print("P2FIT_SELFTEST_OK cases=3 fit_exact=1 bad_geometry_rejected=1 missing_point_rejected=1")


def check_cases(cases):
    """设计点必须成立：官方点 GBS=8，另两点 GBS=16 且其中一点 gas=2、一点 mbs 翻倍。"""
    assert len(cases) == 3, "必须三个点"
    seen = {(c[1], c[2]) for c in cases}
    assert seen == {(4, 1), (8, 1), (4, 2)}, "三点必须是 (mbs4,gas1)/(mbs8,gas1)/(mbs4,gas2)"
    assert cases[0][3] is True and cases[1][3] is False and cases[2][3] is False, \
        "只有官方点可标注为 comparable"


class Sampler(threading.Thread):
    def __init__(self, nid):
        super().__init__(daemon=True)
        self.nid, self.rows, self.on = nid, [], True

    def run(self):
        while self.on:
            try:
                self.rows.append(sample_once(self.nid))
            except Exception as e:
                self.rows.append({"t": time.time(), "error": str(e)[:120]})
            time.sleep(SAMPLE_INTERVAL)

    def stop(self):
        self.on = False
        self.join(timeout=10)


def run_case(tag, mbs, gas, comparable, doc, args, env_json, nid):
    import yaml
    case = os.path.join(args.out, tag)
    os.makedirs(case, exist_ok=False)
    cfg = copy.deepcopy(doc)
    cfg["training"]["train_iters"] = args.steps
    cfg["training"]["micro_batch_size"] = mbs
    cfg["training"]["gradient_accumulation_steps"] = gas
    cfg["training"]["save"] = os.path.join(case, "checkpoint")
    gbs = mbs * gas * int(doc["parallel"]["data_parallel_size"])
    text = yaml.safe_dump(cfg, sort_keys=False)
    back = yaml.safe_load(text)
    assert back["training"]["micro_batch_size"] == mbs
    assert back["training"]["gradient_accumulation_steps"] == gas
    assert back["training"]["train_iters"] == args.steps
    config = os.path.join(case, "config.yaml")
    open(config, "w", encoding="utf-8").write(text)

    log = os.path.join(case, "train.log")
    drv = os.path.join(case, "driver.log")
    cmd = [sys.executable, os.path.join(args.skill, "scripts", "50_train.py"),
           "--config", config, "--env", env_json, "--log", log,
           "--workdir", args.mind, "--steps", str(args.steps),
           "--world-size", "2", "--timeout", "900", "--foreground"]
    if not comparable:
        cmd.append("--allow-gbs-mismatch")
    env = os.environ.copy()
    if args.omp:
        env["OMP_NUM_THREADS"] = args.omp

    sampler = Sampler(nid)
    sampler.start()
    t0 = time.time()
    with open(drv, "w", encoding="utf-8") as fh:
        r = subprocess.run(cmd, cwd=args.skill, env=env, stdout=fh,
                           stderr=subprocess.STDOUT, timeout=args.timeout)
    wall = time.time() - t0
    sampler.stop()

    dtext = open(drv, encoding="utf-8", errors="replace").read()
    ttext = open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else ""
    rows = [(int(i), float(ms)) for i, ms in PAT.findall(ttext)]
    train_rc = re.findall(r"train_rc=(\d+)", dtext)
    valid = (r.returncode == 0 and train_rc == ["0"] and len(rows) == args.steps)
    win = [ms for i, ms in rows if 11 <= i <= args.steps]
    rec = {"tag": tag, "mbs": mbs, "gas": gas, "gbs": gbs,
           "official_comparable": bool(comparable),
           "source_sha256": hashlib.sha256(
               open(os.path.join(args.skill, "scripts", "50_train.py"), "rb").read()).hexdigest(),
           "config_sha256": hashlib.sha256(open(config, "rb").read()).hexdigest(),
           "driver_rc": r.returncode, "train_rc": train_rc, "valid": valid,
           "steps_parsed": len(rows), "wall_seconds": round(wall, 1),
           "iter_ms_all": [ms for _, ms in rows],
           "window": [11, args.steps],
           "window_median_ms": statistics.median(win) if win else None,
           "window_mean_ms": round(sum(win) / len(win), 3) if win else None,
           "all_median_ms": statistics.median([ms for _, ms in rows]) if rows else None,
           "identity": {"boot": open("/proc/sys/kernel/random/boot_id").read().strip(),
                        "hostname": __import__("platform").node() or "unknown",   # ★ 坑 154：不读 /etc/hostname
                        "affinity": sorted(os.sched_getaffinity(0))},
           "samples": sampler.rows}
    json.dump(rec, open(os.path.join(case, "result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("END %s gbs=%d valid=%s wall=%.1fs win_med=%s"
          % (tag, gbs, valid, wall, rec["window_median_ms"]), flush=True)
    return rec


def summarise(results, args):
    by = {r["tag"]: r for r in results}
    tA = by[CASES[0][0]]["window_median_ms"]
    tB = by[CASES[1][0]]["window_median_ms"]
    tC = by[CASES[2][0]]["window_median_ms"]
    a, f, F = fit(tA, tB, tC)
    act = []
    for r in results:
        v = [s["aicore"] for s in r["samples"] if s.get("aicore") is not None]
        r["aicore_samples"] = {"n": len(v), "median": statistics.median(v) if v else None,
                               "max": max(v) if v else None}
        act.append(r)
    out = {"schema": "p2fit.v1", "steps": args.steps, "window": [11, args.steps],
           "cases": {r["tag"]: {k: r[k] for k in
                                ("mbs", "gas", "gbs", "official_comparable", "valid",
                                 "window_median_ms", "window_mean_ms", "wall_seconds",
                                 "aicore_samples", "source_sha256", "identity")}
                     for r in results},
           "solution_ms": {"a_per_sample": round(a, 3), "f_per_microbatch": round(f, 3),
                           "F_per_step": round(F, 3)},
           "interpretation_rule": {
               "f_ge_60": "每 micro-batch 固定开销大 → 下发/建图/host 侧为主 → 图捕获/融合有余量",
               "f_lt_30": "余量在每样本计算 → 图捕获帮不上，需 kernel 级优化",
               "F_ge_80": "每步固定成本高 → 走通信重叠/优化器融合"},
           "hint": ("f = %.1f ms、a = %.3f ms/样本、F = %.1f ms/步 —— 仅在 identity.boot=%s 上成立，"
                    "不得跨时段外推" % (f, a, F, results[0]["identity"]["boot"]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--skill", default="/root/qwen35-ascend-migrator")
    ap.add_argument("--config", default="/root/qwen35-ascend-migrator/out/plan/train_config.yaml")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", default="/root/ops/p2fit")
    ap.add_argument("--mind", default="/root/MindSpeed-MM")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--omp", default=None)
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return 0

    check_cases(CASES)
    if os.path.exists(args.out):
        print("FATAL 输出目录已存在（拒绝覆盖）：%s" % args.out)
        return 3
    trainer = os.path.join(args.skill, "scripts", "50_train.py")
    src = open(trainer, encoding="utf-8", errors="replace").read()
    if "--allow-gbs-mismatch" not in src:
        print("FATAL 远端 50_train.py 没有 --allow-gbs-mismatch（版本不对，拒绝跑 B/C 点）")
        return 3
    import yaml
    doc = yaml.safe_load(open(args.config, encoding="utf-8"))
    dp = int(doc["parallel"]["data_parallel_size"])
    if dp != 2:
        print("FATAL 诊断要求 dp=2（当前 %d）" % dp)
        return 3
    env_json = os.path.join(args.skill, "out", "probe", "env.json")
    if not os.path.isfile(env_json):
        print("FATAL 缺 %s" % env_json)
        return 3
    nid = npu_id()
    os.makedirs(args.out)
    print("P2FIT_START out=%s npu=%s steps=%d dp=%d boot=%s"
          % (args.out, nid, args.steps, dp,
             open("/proc/sys/kernel/random/boot_id").read().strip()), flush=True)

    results = []
    try:
        for tag, mbs, gas, comparable in CASES:
            results.append(run_case(tag, mbs, gas, comparable, doc, args, env_json, nid))
            json.dump(results, open(os.path.join(args.out, "results.json"), "w",
                                    encoding="utf-8"), ensure_ascii=False, indent=1)
    except Exception as e:
        print("P2FIT_FAIL 中断: %s" % e, flush=True)
        return 1

    bad = [r["tag"] for r in results if not r["valid"]]
    if bad:
        print("P2FIT_FAIL 无退出证据/步数不符: %s" % bad)
        return 1
    s = summarise(results, args)
    json.dump(s, open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    sol = s["solution_ms"]
    txt = ["P2FIT out=%s steps=%d boot=%s" % (args.out, args.steps,
                                              results[0]["identity"]["boot"]),
           "  A mbs4 gas1 GBS8  win_med=%.1f ms  wall=%.1fs  aicore_med=%s max=%s"
           % (s["cases"][CASES[0][0]]["window_median_ms"], s["cases"][CASES[0][0]]["wall_seconds"],
              s["cases"][CASES[0][0]]["aicore_samples"]["median"],
              s["cases"][CASES[0][0]]["aicore_samples"]["max"]),
           "  B mbs8 gas1 GBS16 win_med=%.1f ms  wall=%.1fs  aicore_med=%s max=%s"
           % (s["cases"][CASES[1][0]]["window_median_ms"], s["cases"][CASES[1][0]]["wall_seconds"],
              s["cases"][CASES[1][0]]["aicore_samples"]["median"],
              s["cases"][CASES[1][0]]["aicore_samples"]["max"]),
           "  C mbs4 gas2 GBS16 win_med=%.1f ms  wall=%.1fs  aicore_med=%s max=%s"
           % (s["cases"][CASES[2][0]]["window_median_ms"], s["cases"][CASES[2][0]]["wall_seconds"],
              s["cases"][CASES[2][0]]["aicore_samples"]["median"],
              s["cases"][CASES[2][0]]["aicore_samples"]["max"]),
           "  fit: a=%.3f ms/样本  f=%.3f ms/micro-batch  F=%.3f ms/步" % (sol["a_per_sample"],
                                                                        sol["f_per_microbatch"],
                                                                        sol["F_per_step"]),
           "  B_on_A(mbs 翻倍): %.3f×   C_on_B(同 GBS、两个 micro-batch): %.3f×"
           % (s["cases"][CASES[1][0]]["window_median_ms"] / s["cases"][CASES[0][0]]["window_median_ms"],
              s["cases"][CASES[2][0]]["window_median_ms"] / s["cases"][CASES[1][0]]["window_median_ms"])]
    open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8").write("\n".join(txt) + "\n")
    print("\n".join(txt))
    print("P2FIT_OK out=%s" % args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
