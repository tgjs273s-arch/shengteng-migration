#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""step57_fsdp_group_ablation.py — FSDP 分组结构 A/B/A 消融（减少"多而小"发射的正面进攻）

为什么做这个（数据从哪来）
--------------------------
归因结论（`性能余量_20260917/README_性能余量.md` §五之三）：
官方几何每步 **6,994 次 kernel 发射**、固定开销 146.4 ms/步，其中
**`aclnnInplaceCopy_{Transpose,Cast,Slice}` 合计 ≈2,112 次/步**（占全部发射的 30%）。
这类"inplace copy + 布局转换"是 **FSDP2 参数分片/重组**的典型形态，
而当前配置 `parallel.fsdp_plan.apply_modules` 里**同时列出了父模块与子模块**：

    - model.visual
    - model.visual.blocks.{*}
    - model.language_model
    - model.language_model.layers.{*}      ← 与上一行**嵌套**
    - lm_head
    - mtp

**嵌套分组 ⇒ 嵌套 FSDP ⇒ 每个 micro-batch 在两层层级上各做一遍分片重组**。
把层级压平（只保留最细一层）在理论上能显著减少这类 copy 发射。

判据（跑之前先定好，避免事后解释）
----------------------------------
* 变体 B（压平分组）的窗口中位 ≤ **0.95 ×** 基线 A 的均值 → **有效（≥5%）**
* 两次基线 A1/A2 的相对差 > **3%** → 会话漂移过大 → **不确定，不下结论**
* 任一组无退出证据/步数不足/GBS≠8 → **整轮作废**
* 与官方 431.3 ms 的对比**只在 A 点讨论**（B 点仍是同一 GBS=8 几何，可比）

红线
----
* 只改 `parallel.fsdp_plan.apply_modules`（**单一变量**），其余逐字节不动，并回读校验；
* 几何保持 mbs4/gas1/dp2 → GBS=8（脚本内断言，不加 --allow-gbs-mismatch）；
* 每组独立目录 + 退出证据；结果带 `identity.boot`；**不得**把 profiling 的时长当性能。

用法：bash scripts/p57_launch.sh /root/ops/<新目录>
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

# 变体定义：A = 原样；B = 去掉父级条目（压平为最细一层）
NESTED_PARENTS = ("model.visual", "model.language_model")


def flatten_modules(mods):
    """把"父+子"嵌套列表压平：去掉父级条目，只保留最细一层。"""
    out = [m for m in mods if m not in NESTED_PARENTS]
    return out


def plan(runs):
    """runs: [(tag, variant)]；记录每个变体的 config 差异，供报告引用。"""
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", default="/root/qwen35-ascend-migrator")
    ap.add_argument("--config", default="/root/qwen35-ascend-migrator/out/plan/train_config.yaml")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", default="/root/ops/p57")
    ap.add_argument("--mind", default="/root/MindSpeed-MM")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--order", default="A,B,A", help="默认 A/B/A：两次基线用于估计会话漂移")
    args = ap.parse_args()

    if os.path.exists(args.out):
        print("FATAL 输出目录已存在（拒绝覆盖）：%s" % args.out)
        return 3
    import yaml
    raw = open(args.config, "rb").read()
    doc0 = yaml.safe_load(raw)
    tr = doc0["training"]
    dp = int(doc0["parallel"]["data_parallel_size"])
    gbs = int(tr["micro_batch_size"]) * int(tr["gradient_accumulation_steps"]) * dp
    if gbs != 8:
        print("FATAL 只做官方几何（GBS=8），当前 GBS=%d" % gbs)
        return 3
    mods0 = list(doc0["parallel"]["fsdp_plan"]["apply_modules"])
    mods_flat = flatten_modules(mods0)
    if mods_flat == mods0:
        print("FATAL apply_modules 里没有可压平的父级条目：%s" % mods0)
        return 3
    os.makedirs(args.out)

    variants = {"A": mods0, "B": mods_flat}
    order = [x.strip().upper() for x in args.order.split(",") if x.strip()]
    boot = open("/proc/sys/kernel/random/boot_id").read().strip()
    env_json = os.path.join(args.skill, "out", "probe", "env.json")
    print("P57_START out=%s order=%s steps=%d GBS=%d boot=%s" % (args.out, order, args.steps, gbs, boot), flush=True)
    print("  A(原样) = %s" % mods0, flush=True)
    print("  B(压平) = %s" % mods_flat, flush=True)

    results = []
    for i, v in enumerate(order):
        tag = "%02d_%s" % (i, v)
        case = os.path.join(args.out, tag)
        os.makedirs(case, exist_ok=False)
        cfg = copy.deepcopy(doc0)
        cfg["parallel"]["fsdp_plan"]["apply_modules"] = variants[v]
        cfg["training"]["train_iters"] = args.steps
        cfg["training"]["save"] = os.path.join(case, "checkpoint")
        text = yaml.safe_dump(cfg, sort_keys=False)
        back = yaml.safe_load(text)
        assert back["parallel"]["fsdp_plan"]["apply_modules"] == variants[v], "回读不一致"
        assert back["training"]["micro_batch_size"] * back["training"]["gradient_accumulation_steps"] * dp == 8
        # 单一变量校验：除 apply_modules / train_iters / save 外，其余必须与原配置逐键相同
        b2 = copy.deepcopy(back)
        b2["parallel"]["fsdp_plan"]["apply_modules"] = mods0
        b2["training"]["train_iters"] = tr["train_iters"]
        b2["training"]["save"] = tr.get("save")
        d0 = copy.deepcopy(doc0)
        if d0["training"].get("save") is None:
            d0["training"].pop("save", None)
            b2["training"].pop("save", None)
        same_keys = json.dumps(b2, sort_keys=True) == json.dumps(d0, sort_keys=True)
        config = os.path.join(case, "config.yaml")
        open(config, "w", encoding="utf-8").write(text)

        log, drv = os.path.join(case, "train.log"), os.path.join(case, "driver.log")
        cmd = [sys.executable, os.path.join(args.skill, "scripts", "50_train.py"),
               "--config", config, "--env", env_json, "--log", log, "--workdir", args.mind,
               "--steps", str(args.steps), "--world-size", "2", "--timeout", "900", "--foreground"]
        t0 = time.time()
        with open(drv, "w", encoding="utf-8") as fh:
            r = subprocess.run(cmd, cwd=args.skill, env=os.environ.copy(),
                               stdout=fh, stderr=subprocess.STDOUT, timeout=args.timeout)
        wall = time.time() - t0
        dtext = open(drv, encoding="utf-8", errors="replace").read()
        ttext = open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else ""
        rows = [(int(x), float(ms)) for x, ms in PAT.findall(ttext)]
        train_rc = re.findall(r"train_rc=(\d+)", dtext)
        valid = (r.returncode == 0 and train_rc == ["0"] and len(rows) == args.steps)
        win = [ms for x, ms in rows if 11 <= x <= args.steps]
        rec = {"tag": tag, "variant": v, "apply_modules": variants[v],
               "single_variable_ok": same_keys, "valid": valid, "driver_rc": r.returncode,
               "train_rc": train_rc, "steps_parsed": len(rows), "wall_seconds": round(wall, 1),
               "window": [11, args.steps], "window_median_ms": statistics.median(win) if win else None,
               "window_mean_ms": round(sum(win) / len(win), 3) if win else None,
               "iter_ms_all": [ms for _, ms in rows],
               "config_sha256": hashlib.sha256(open(config, "rb").read()).hexdigest(),
               "source_sha256": hashlib.sha256(
                   open(os.path.join(args.skill, "scripts", "50_train.py"), "rb").read()).hexdigest(),
               "identity": {"boot": boot, "hostname": __import__("platform").node() or "unknown"}}  # 坑 154
        json.dump(rec, open(os.path.join(case, "result.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        results.append(rec)
        json.dump(results, open(os.path.join(args.out, "results.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("END %s variant=%s valid=%s single_var=%s win_med=%s wall=%.0fs"
              % (tag, v, valid, same_keys, rec["window_median_ms"], wall), flush=True)
        if not valid:
            print("P57_FAIL 无退出证据，停止（不继续消耗算力）", flush=True)
            return 1

    A = [r for r in results if r["variant"] == "A"]
    B = [r for r in results if r["variant"] == "B"]
    if not A or not B:
        print("P57_FAIL 缺少 A 或 B", flush=True)
        return 1
    a_mean = sum(r["window_median_ms"] for r in A) / len(A)
    b_mean = sum(r["window_median_ms"] for r in B) / len(B)
    drift = (max(r["window_median_ms"] for r in A) - min(r["window_median_ms"] for r in A)) \
        / min(r["window_median_ms"] for r in A) * 100 if len(A) > 1 else None
    ratio = b_mean / a_mean
    if drift is not None and drift > 3.0:
        verdict = "UNCERTAIN: 基线漂移 %.2f%% > 3%%，本轮不下结论" % drift
    elif ratio <= 0.95:
        verdict = "WIN: 压平分组使窗口中位降 %.1f%%（%.1f → %.1f ms）" % ((1 - ratio) * 100, a_mean, b_mean)
    else:
        verdict = "NO_WIN: 压平分组未带来 ≥5%% 收益（%.1f → %.1f ms，%.2f×）" % (a_mean, b_mean, ratio)
    out = {"schema": "p57.v1", "order": order, "baseline_A_mean_ms": round(a_mean, 2),
           "variant_B_mean_ms": round(b_mean, 2), "ratio_B_over_A": round(ratio, 4),
           "baseline_drift_pct": round(drift, 2) if drift is not None else None,
           "verdict": verdict, "official_median_ms": 431.3,
           "A_vs_official": round(431.3 / a_mean, 3), "B_vs_official": round(431.3 / b_mean, 3),
           "runs": [{k: r[k] for k in ("tag", "variant", "valid", "single_variable_ok",
                                       "window_median_ms", "wall_seconds", "identity")} for r in results],
           "note": "同一 boot 内 A/B/A；窗口 11–30；GBS=8；只有 A 点可与官方对标（B 点同为 GBS=8 亦可比）"}
    json.dump(out, open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("P57_RESULT baselineA=%.1f ms  variantB=%.1f ms  ratio=%.3f  drift=%s"
          % (a_mean, b_mean, ratio, ("%.2f%%" % drift) if drift is not None else "n/a"), flush=True)
    print("VERDICT %s" % verdict, flush=True)
    print("P57_OK out=%s" % args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
