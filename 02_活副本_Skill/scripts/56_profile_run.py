#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""step56_profile_run.py — 官方几何下的 profiling 运行：把 `f = 146.9 ms/micro-batch` 归因

为什么做
--------
step54 三点（+step55 第四点预测命中）证明：官方几何每步 432.7 ms 里
**146.9 ms（34%）是"每个 micro-batch 的固定开销"**，且与样本数无关。
但那 147 ms 是"宿主下发间隙 / 通信未重叠 / 数据等待"里的哪一个，**step54 判不出来**
（它的 AICore 采样在迭代期恒为 0，已作废）。

Mspeed-MM 自带 profiler（`mindspeed_mm/fsdp/tools/profiler.py`），由**配置项**开启：
`tools.profile.enable / ranks / static_param.{level,start_step,end_step,save_path,with_cpu,...}`
→ 它用 `torch_npu.profiler` 采 step 级 trace，产出 `step_trace_time.csv`，
其中直接给出 **Stage / Computing / Communication / Communication(Not Overlapped) / Free** 五个桶。
这正是把 f 归因所需的口径。

本脚本做什么
------------
1. 用**官方几何**（mbs=4, gas=1, dp=2 → GBS=8）跑 25 步，只对 **rank 0** 在第 15–20 步开 profiler；
2. 校验退出证据（driver_rc=0、train_rc=['0']、解析步数==25）；
3. 解析 `step_trace_time.csv` 求各桶均值 + `kernel_details.csv` 的 kernel 行数（发射次数）；
4. 落盘 `result.json`，打印 `P2PROF_OK`。

纪律
----
* **profiling 会拖慢训练**，所以 profiling 运行的步时长**不得**用于与 432.7 ms 对标；
  只用于"桶的**比例**"归因（这一点在报告里必须写明）。
* 只跑官方几何，**不加** --allow-gbs-mismatch（GBS=8 红线）；配置回读校验后才跑。
* 找不到 `step_trace_time.csv` 就**如实报失败并列目录**，不猜数、不用 kernel_details 硬凑。
"""
import argparse
import copy
import csv
import glob
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time

PAT = re.compile(r"iteration\s+(\d+)\s*/\s*\d+.*?elapsed time per iteration \(ms\):\s*([\d.]+)", re.S)
BUCKETS = ["Stage", "Computing", "Communication", "Communication(Not Overlapped)", "Free"]


def apply_single_die(doc, single):
    """★ 单 die 受限环境：保留 GBS=8 红线，但把几何改成 world1/mbs8/gas1。

    目的与边界（写进产物，避免误用）：
      * 只有**一块 die** 时无法复刻官方 dp2 几何；此时用 mbs=8/gas=1/world=1 仍满足
        `GBS = mbs x gas x world = 8`（红线不破），可继续做**发射/调用点归因**与
        **与并行无关的配置 A/B**；
      * 但 **几何可比性 = False**：与官方几何不同（每 rank micro-batch 8 vs 4、
        无 FSDP 通信），**不得**用于"优于官方"或通信类结论。
    """
    if not single:
        return doc, int(doc["parallel"]["data_parallel_size"]), True
    doc["parallel"]["data_parallel_size"] = 1
    tr = doc["training"]
    tr["micro_batch_size"] = 8
    tr["gradient_accumulation_steps"] = 1
    return doc, 1, False


def official_comparable(geometry_ok, data_identity):
    """★ `official_comparable` 必须同时满足 **几何一致** 与 **数据一致**。

    为什么改（外部复核指出的误判）：原实现只按**卡数/几何**推导 ⇒ 在 dp2 上跑 **mock 数据**
    会被标成 `official_comparable = True`，于是产物自己声称"与官方可比"——而我们的数据是 mock
    （1 图/样本、cutoff_len=1024、非官方 COCO），官方窗口的中位数根本不能当分母。
    ⇒ 现在**默认 False**（fail-closed），只有显式传入非空的 `--official-data-identity`
    才可能为 True；无论真假都记录判断理由（reason），不让读者去猜。
    """
    reasons = []
    if not geometry_ok:
        reasons.append("几何与官方不同（单 die：world1/mbs8/gas1）")
    if not str(data_identity or "").strip():
        reasons.append("数据身份未声明（默认视为 mock ⇒ 不可与官方比较）")
    ok = geometry_ok and bool(str(data_identity or "").strip())
    return ok, ("；".join(reasons) if reasons else "几何与数据均已声明为官方口径：%s" % data_identity)


def selftest_official():
    """`--selftest-official`：`official_comparable` 判据的三例自检（**含坏样本**）。

    为什么必须有：外部复核指出"仍按卡数推导 ⇒ mock 在双卡下可能被错误标为 true"。
    改完若不自检，就等于用一句"我改好了"替代判据（本项目已因这类自说自话付出过代价）。
    """
    cases = [
        ("坏样本①：dp2 几何 + **mock 数据（未声明身份）** ⇒ 必须 False",
         (True, ""), False),
        ("坏样本②：dp2 几何 + 空白身份声明 ⇒ 必须 False（空白不算声明）",
         (True, "   "), False),
        ("坏样本③：单 die 几何 + 已声明官方身份 ⇒ 仍必须 False（几何不可比）",
         (False, "official_coco_train2017_llava150k"), False),
        ("正对照：dp2 几何 + 显式官方身份 ⇒ True",
         (True, "official_coco_train2017_llava150k"), True),
    ]
    fails = 0
    for name, (geom, ident), want in cases:
        got, reason = official_comparable(geom, ident)
        ok = (got == want)
        fails += 0 if ok else 1
        print("  [%s] %-56s got=%-5s want=%-5s | %s"
              % ("OK" if ok else "**不符**", name, got, want, reason[:80]))
    print("PROF_OFFICIAL_SELFTEST_%s cases=%d failed=%d"
          % ("OK" if not fails else "FAIL", len(cases), fails))
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", default="/root/qwen35-ascend-migrator")
    ap.add_argument("--config", default="/root/qwen35-ascend-migrator/out/plan/train_config.yaml")
    ap.add_argument("--steps", type=int, default=25, help="总步数（必须 > profile end_step）")
    ap.add_argument("--prof-start", type=int, default=15)
    ap.add_argument("--prof-end", type=int, default=20)
    ap.add_argument("--with-stack", action="store_true", default=False,
                    help="采集算子调用栈（用于把发射归因到 Python 调用点；会显著增大产物）")
    ap.add_argument("--record-shapes", action="store_true", default=False)
    # ★ 2026-09-22 新增：显存维度。原实现把 `with_memory` **硬编码为 False** ⇒ 无法回答
    #   "激活/logits 占用是否突出"与"显存是否跨步增长"这两类问题（而 profile 协议里
    #   它们正是“仍未决定”的第二项）。默认仍为 False（保持既有行为不变），显式开才采。
    ap.add_argument("--with-memory", action="store_true", default=False,
                    help="采集显存信息（torch_npu profiler 的 with_memory；会增大产物、拖慢运行）")
    ap.add_argument("--single-die", action="store_true", default=False,
                    help="只有 1 块 die 时使用：world1/mbs8/gas1（GBS=8 不变，但几何可比性=False）")
    ap.add_argument("--official-data-identity", default="",
                    help="★ 数据身份声明：**只有**显式给出非空值（例如官方 COCO 数据集标识）"
                         "才可能 official_comparable=True；不传即视为 mock ⇒ False（fail-closed）")
    ap.add_argument("--selftest-official", action="store_true", default=False,
                    help="只跑 official_comparable 判据的自检（4 例，含 3 个坏样本）后退出")
    ap.add_argument("--out", default="/root/ops/p2prof")
    ap.add_argument("--mind", default="/root/MindSpeed-MM")
    ap.add_argument("--timeout", type=int, default=1500)
    args = ap.parse_args()

    if args.selftest_official:            # ★ 自检必须在**任何副作用之前**（不建目录、不起训练）
        return selftest_official()

    if os.path.exists(args.out):
        print("FATAL 输出目录已存在（拒绝覆盖）：%s" % args.out)
        return 3
    if not (0 < args.prof_start < args.prof_end < args.steps):
        print("FATAL profiling 窗口必须落在训练步数内")
        return 3
    import yaml
    doc = yaml.safe_load(open(args.config, encoding="utf-8"))
    doc, _w, geom_ok = apply_single_die(doc, args.single_die)
    # ★ official_comparable = 几何一致 **且** 数据身份已声明（默认 False，fail-closed）
    official, official_reason = official_comparable(geom_ok, args.official_data_identity)
    dp = int(doc["parallel"]["data_parallel_size"])
    tr = doc["training"]
    mbs, gas = int(tr["micro_batch_size"]), int(tr["gradient_accumulation_steps"])
    gbs = mbs * gas * dp
    if gbs != 8:
        print("FATAL 本运行只做官方几何（GBS=8），当前 mbs=%d gas=%d dp=%d → GBS=%d" % (mbs, gas, dp, gbs))
        return 3

    os.makedirs(args.out)
    cfg = copy.deepcopy(doc)
    cfg["training"]["train_iters"] = args.steps
    cfg["training"]["save"] = os.path.join(args.out, "checkpoint")
    prof_path = os.path.join(args.out, "prof")
    cfg.setdefault("tools", {})
    cfg["tools"]["profile"] = {
        "enable": True, "profile_type": "static", "ranks": [0],
        "static_param": {"level": "level1", "with_stack": bool(args.with_stack),
                         "with_memory": bool(args.with_memory),
                         "record_shapes": bool(args.record_shapes), "with_cpu": True, "save_path": prof_path,
                         "start_step": args.prof_start, "end_step": args.prof_end,
                         "data_simplification": False, "aic_metrics_type": "PipeUtilization",
                         "analyse_flag": True},
    }
    text = yaml.safe_dump(cfg, sort_keys=False)
    back = yaml.safe_load(text)
    assert back["tools"]["profile"]["static_param"]["end_step"] == args.prof_end
    assert back["training"]["micro_batch_size"] * back["training"]["gradient_accumulation_steps"] * dp == 8
    config = os.path.join(args.out, "config.yaml")
    open(config, "w", encoding="utf-8").write(text)

    log, drv = os.path.join(args.out, "train.log"), os.path.join(args.out, "driver.log")
    env_json = os.path.join(args.skill, "out", "probe", "env.json")
    boot = open("/proc/sys/kernel/random/boot_id").read().strip()
    print("PROF_START out=%s steps=%d window=[%d,%d] GBS=%d dp=%d official_comparable=%s "
          "geom_ok=%s data_identity=%r boot=%s"
          % (args.out, args.steps, args.prof_start, args.prof_end, gbs, dp, official,
             geom_ok, args.official_data_identity or "(未声明)", boot), flush=True)
    print("PROF_OFFICIAL_COMPARABLE_REASON %s" % official_reason, flush=True)
    cmd = [sys.executable, os.path.join(args.skill, "scripts", "50_train.py"),
           "--config", config, "--env", env_json, "--log", log, "--workdir", args.mind,
           "--steps", str(args.steps), "--world-size", str(dp),
           "--timeout", "900", "--foreground"]
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

    rec = {"schema": "p2prof.v1", "valid": valid, "driver_rc": r.returncode, "train_rc": train_rc,
           "steps_parsed": len(rows), "wall_seconds": round(wall, 1),
           "geometry": {"mbs": mbs, "gas": gas, "dp": dp, "gbs": gbs,
                     "official_comparable": official,
                     "official_comparable_reason": official_reason,
                     "geometry_matches_official": geom_ok,
                     "data_identity": args.official_data_identity or None},
           "steps_all_median_ms": statistics.median([ms for _, ms in rows]) if rows else None,
           "window_11_end_median_ms": statistics.median(win) if win else None,
           "profiling_window": [args.prof_start, args.prof_end], "prof_save_path": prof_path,
           "run_dir": args.out,
           "identity": {"boot": boot, "hostname": __import__("platform").node() or "unknown",  # 坑 154
                        "affinity": sorted(os.sched_getaffinity(0))},
           "note": "profiling 会拖慢训练：本运行的步时长不得与 432.7 ms 对标，只用于桶比例归因"}

    # ---- 解析 profiling 产物
    files = glob.glob(os.path.join(prof_path, "**", "*.csv"), recursive=True)
    stt = [f for f in files if os.path.basename(f) == "step_trace_time.csv"]
    kdt = [f for f in files if os.path.basename(f) == "kernel_details.csv"]
    rec["prof_files_found"] = [os.path.relpath(f, prof_path) for f in files][:40]
    if not stt:
        print("P2PROF_FAIL 未找到 step_trace_time.csv；目录内容=%s" % rec["prof_files_found"][:12], flush=True)
        json.dump(rec, open(os.path.join(args.out, "result.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        return 1

    with open(stt[0], newline="", encoding="utf-8", errors="replace") as fh:
        rdr = list(csv.DictReader(fh))
    rec["step_trace_rows"] = len(rdr)
    rec["step_trace_header"] = list(rdr[0].keys()) if rdr else []
    buckets = {}
    for b in BUCKETS:
        key = next((k for k in (rdr[0].keys() if rdr else []) if k.strip() == b), None)
        if key is None:
            key = next((k for k in (rdr[0].keys() if rdr else []) if b.lower().replace("(", "").replace(")", "")
                        in k.lower().replace("(", "").replace(")", "")), None)
        if key is None:
            continue
        vals = []
        for row in rdr:
            try:
                vals.append(float(row[key]))
            except (TypeError, ValueError):
                pass
        if vals:
            buckets[b] = {"key": key, "n": len(vals), "mean": round(sum(vals) / len(vals), 3),
                          "median": round(statistics.median(vals), 3), "max": round(max(vals), 3)}
    rec["buckets"] = buckets
    if kdt:
        with open(kdt[0], newline="", encoding="utf-8", errors="replace") as fh:
            n_k = sum(1 for _ in fh) - 1
        rec["kernel_details_rows"] = n_k
        rec["kernel_launches_per_profiled_step"] = round(n_k / max(1, args.prof_end - args.prof_start), 1)

    if "Stage" in buckets and "Computing" in buckets:
        st = buckets["Stage"]["mean"]
        rec["shares_pct"] = {b: round(buckets[b]["mean"] / st * 100, 1) for b in buckets if b != "Stage"}
        rec["stage_over_steady"] = round(st / rec["window_11_end_median_ms"], 3) if rec["window_11_end_median_ms"] else None
    json.dump(rec, open(os.path.join(args.out, "result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    def g(b, k="mean"):
        return buckets.get(b, {}).get(k)
    print("PROF_RESULT valid=%s steps=%s wall=%.0fs" % (valid, len(rows), wall), flush=True)
    print("  Stage=%s  Computing=%s  Comm=%s  Comm_not_overlapped=%s  Free=%s (ms, 均值)"
          % (g("Stage"), g("Computing"), g("Communication"), g("Communication(Not Overlapped)"), g("Free")), flush=True)
    print("  占比: %s" % json.dumps(rec.get("shares_pct", {}), ensure_ascii=False), flush=True)
    print("  kernel 发射/被采步: %s（来自 %d 行 kernel_details）"
          % (rec.get("kernel_launches_per_profiled_step"), rec.get("kernel_details_rows", 0)), flush=True)
    print("P2PROF_OK out=%s" % args.out, flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())

