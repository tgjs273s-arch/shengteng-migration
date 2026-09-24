# -*- coding: utf-8 -*-
r"""63_two_config_ab.py —— **双配置**配对对照（dp1 vs dp2 这类"多键联合变化"）

为什么不能复用 `59_config_ab.py`
--------------------------------
`59` 是**单键** A/B，并用 `ab_comparable()` 要求"除登记键外完全一致"。
而 dp1↔dp2 必须**同时**改三个键：
    parallel.data_parallel_size        2 → 1
    training.gradient_accumulation_steps 1 → 2
    parallel.fully_shard_parallel_size 2 → 1
因为 `GBS = mbs × gas × world` 是硬约束（`50_train.py` 非 8 直接 rc=3 拒跑）。
单键模型表达不了这种变化 ⇒ 本工具用**显式声明的差异清单**表达"到底变了什么"，
并要求实际差异与声明**逐键相等**（多一个键就拒绝）。

判据**不另起一套**
------------------
直接复用 `59_config_ab.py` 的 `compute_valid` / `paired_stats` / `pointwise_all` / `verdict_of`
—— 同一语义只留一处实现。收益判定口径与 59 完全一致：
**改善区间的下界 ≥5% 才算 WIN**；区间跨 0 → UNCERTAIN；数值不合格先于性能结论报出。

如实声明的边界（写进产物，不靠记忆）
------------------------------------
* `official_comparable = False`：dp1 组不是官方几何，**两组都不能对官方**。
* **每步全局样本身份未做字节级验证**（需要 sampler hook 才能落 sample id）；
  本案只能声明"同数据文件 + 同 seed + 同 shuffle:false + 同 GBS"，属**未闭合项**。
* **实际 token 吞吐不予估计**：日志未提供每步有效 token 数时记 `null` 并写明原因，
  **不编数**（编出来的吞吐比没有更糟）。
* dataloader `num_workers` 的口径（每 rank / 全局）由配置本身决定并写进产物。

用法
----
    python3 63_two_config_ab.py --config-a <dp1.yaml> --config-b <dp2.yaml> \
        --declare parallel.data_parallel_size,training.gradient_accumulation_steps,\
parallel.fully_shard_parallel_size --order A,B,B,A,A,B --steps 60 --numeric-steps 2 --out <dir>
    python3 63_two_config_ab.py --selftest      # 只跑差异声明/GBS 自检（不跑训练）
"""
import argparse
import hashlib
import importlib
import json
import os
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import yaml
except ImportError:
    print("T63_FAIL no_pyyaml")
    sys.exit(2)

# ★ 复用 59 的判据实现（模块名以数字开头 → 用 importlib）
_p59 = importlib.import_module("59_config_ab")
PAT = _p59.PAT
compute_valid = _p59.compute_valid
paired_stats = _p59.paired_stats
pointwise_all = _p59.pointwise_all
verdict_of = _p59.verdict_of

CGROUP_MEM = "/sys/fs/cgroup/memory"


def dig(doc, dotted, default=None):
    cur = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def put(doc, dotted, value):
    parts = dotted.split(".")
    cur = doc
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    if not isinstance(cur, dict):
        return False
    cur[parts[-1]] = value
    return True


def flatten(doc, prefix=""):
    out = {}
    if isinstance(doc, dict):
        for k, v in doc.items():
            out.update(flatten(v, "%s.%s" % (prefix, k) if prefix else str(k)))
    elif isinstance(doc, list):
        out[prefix] = repr(doc)
    else:
        out[prefix] = doc
    return out


# 每次实验按设计就会不同、且与"平行布局"无关的键（不构成未声明差异）
IGNORE_KEYS = ("training.save", "training.train_iters", "tools", "profile", "memory_profile")


def diff_keys(cfg_a, cfg_b):
    fa, fb = flatten(cfg_a), flatten(cfg_b)
    out = []
    for k in sorted(set(fa) | set(fb)):
        if any(k == ig or k.startswith(ig + ".") for ig in IGNORE_KEYS):
            continue
        if fa.get(k, "<absent>") != fb.get(k, "<absent>"):
            out.append(k)
    return out


def gbs_of(cfg):
    return (int(dig(cfg, "training.micro_batch_size"))
            * int(dig(cfg, "training.gradient_accumulation_steps"))
            * int(dig(cfg, "parallel.data_parallel_size")))


def cgroup_peak():
    try:
        with open(os.path.join(CGROUP_MEM, "memory.max_usage_in_bytes")) as fh:
            return int(fh.read().split()[0])
    except Exception:
        return None


def run_one(tag, variant, cfg, args, boot):
    case = os.path.join(args.out, tag)
    os.makedirs(case, exist_ok=False)
    cfg = json.loads(json.dumps(cfg))
    cfg["training"]["train_iters"] = args.steps
    cfg["training"]["save"] = os.path.join(case, "checkpoint")
    config = os.path.join(case, "config.yaml")
    # 先校验后落盘：GBS 红线必须在写盘前成立
    assert gbs_of(cfg) == 8, "GBS 不是 8"
    open(config, "w", encoding="utf-8").write(yaml.safe_dump(cfg, sort_keys=False))

    log, drv = os.path.join(case, "train.log"), os.path.join(case, "driver.log")
    env_json = os.path.join(args.skill, "out", "probe", "env.json")
    world = int(dig(cfg, "parallel.data_parallel_size"))
    cmd = [sys.executable, os.path.join(args.skill, "scripts", "50_train.py"),
           "--config", config, "--env", env_json, "--log", log, "--workdir", args.mind,
           "--steps", str(args.steps), "--world-size", str(world),
           "--timeout", "900", "--foreground"]
    t0 = time.time()
    with open(drv, "w", encoding="utf-8") as fh:
        r = subprocess.run(cmd, cwd=args.skill, env=os.environ.copy(),
                           stdout=fh, stderr=subprocess.STDOUT, timeout=args.timeout)
    wall = time.time() - t0
    dtext = open(drv, encoding="utf-8", errors="replace").read()
    ttext = open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else ""
    rows = [(int(i), float(ms), float(lo), float(gn)) for i, ms, lo, gn in PAT.findall(ttext)]
    ids = [i for i, _m, _l, _g in rows]
    train_rc = re.findall(r"train_rc=(\d+)", dtext)
    valid, steps_complete = compute_valid(r.returncode, train_rc, ids, args.steps)
    win = [ms for i, ms, _l, _g in rows if 11 <= i <= args.steps]
    med = statistics.median(win) if win else None
    # 设备显存：日志里的 `max allocated`（进程内分配器口径，比设备级更贴合本实验）
    mxa = [float(x) for x in re.findall(r"max allocated:\s*([\d.]+)", ttext)]
    rec = {
        "tag": tag, "variant": variant, "valid": valid, "steps_complete": steps_complete,
        # ★ 2026-09-21：**补齐字段形状**。本工具复用 59 的 `verdict_of()`，而它按 59 的
        #   record 形状读 `single_variable_ok` —— 63 原来不写这个字段，于是重算时
        #   `verdict_of` 判 `INVALID（存在非单变量运行）`，把**已经跑完的六组有效数据**判成无效
        #   （与 `window_median_ms` 那个 KeyError 同源：**复用别人的函数时默认了记录形状相同**）。
        #   本字段在 63 的语义下是**真**的：`--declare` 与实际差异在开跑前已逐键校验通过
        #   （不通过就直接 rc=3 退出，根本跑不到这里）⇒ 每组都是"只差被声明的那些键"。
        "single_variable_ok": True,
        "driver_rc": r.returncode, "train_rc": train_rc, "steps_parsed": len(rows),
        "step_ids_tail": ids[-5:], "wall_seconds": round(wall, 1),
        "window_11_end_median_ms": med,
        "median_step_s": (med / 1000.0 if med else None),
        "samples_per_s": (8.0 / (med / 1000.0) if med else None),
        "iter_ms": [ms for _i, ms, _l, _g in rows],
        "loss": [lo for _i, _m, lo, _g in rows],
        "grad_norm": [gn for _i, _m, _l, gn in rows],
        "max_allocated_mb": max(mxa) if mxa else None,
        "cgroup_peak_max_usage_bytes": cgroup_peak(),
        "world_size": world,
        "gbs": gbs_of(cfg),
        "config_sha256": hashlib.sha256(open(config, "rb").read()).hexdigest(),
        "identity": {"boot": boot, "hostname": __import__("platform").node() or "unknown"},
    }
    json.dump(rec, open(os.path.join(case, "result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("T63_END %s variant=%s valid=%s steps=%s win_med=%s samples_per_s=%s wall=%.0fs"
          % (tag, variant, valid, len(rows), med, rec["samples_per_s"], wall), flush=True)
    return rec


def selftest():
    """差异声明/GBS 的自检（不跑训练）。每个用例都是**真的坏样本**。"""
    fails = []

    def ck(name, cond, detail=""):
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                               "" if cond or not detail else "  ← " + detail))
        if not cond:
            fails.append(name)

    base = {"parallel": {"data_parallel_size": 2, "fully_shard_parallel_size": 2},
            "training": {"micro_batch_size": 4, "gradient_accumulation_steps": 1,
                         "save": "x", "train_iters": 100}}
    dp1 = {"parallel": {"data_parallel_size": 1, "fully_shard_parallel_size": 1},
           "training": {"micro_batch_size": 4, "gradient_accumulation_steps": 2,
                        "save": "y", "train_iters": 100}}
    d1 = diff_keys(base, dp1)
    ck("差异检测：dp2↔dp1 恰为 3 个键（save/train_iters 被忽略）",
       sorted(d1) == ["parallel.data_parallel_size", "parallel.fully_shard_parallel_size",
                      "training.gradient_accumulation_steps"], "diffs=%s" % d1)

    dp1_extra = json.loads(json.dumps(dp1))
    dp1_extra["training"]["micro_batch_size"] = 8
    d2 = diff_keys(base, dp1_extra)
    ck("差异检测：改动 mbs 会额外暴露出来（不被吞）",
       "training.micro_batch_size" in d2, "diffs=%s" % d2)

    ck("GBS：base=8", gbs_of(base) == 8, "gbs=%s" % gbs_of(base))
    ck("GBS：dp1=8（mbs4×gas2×world1）", gbs_of(dp1) == 8, "gbs=%s" % gbs_of(dp1))
    bad = json.loads(json.dumps(dp1))
    bad["training"]["gradient_accumulation_steps"] = 1  # → GBS=4
    ck("GBS：gas 改成 1 → 4，必须被拒", gbs_of(bad) != 8, "gbs=%s" % gbs_of(bad))
    print("T63_SELFTEST cases=%d failed=%d" % (5, len(fails)))
    return 0 if not fails else 1


def _geom_of(cfg):
    return (int(dig(cfg, "parallel.data_parallel_size")),
            int(dig(cfg, "training.micro_batch_size")),
            int(dig(cfg, "training.gradient_accumulation_steps")))


def _comparability_note(cfg_a, cfg_b, note):
    """按**实际配置**说明"为什么不能对官方"（不写死措辞，避免复用者被带偏）。"""
    ga, gb = _geom_of(cfg_a), _geom_of(cfg_b)
    official = (2, 4, 1)
    parts = []
    if ga == official and gb == official:
        parts.append("两组都是**官方几何** world2/mbs4/gas1")
        parts.append("`official_comparable` 仍为 False —— 原因是**数据不是官方 COCO**（本机是 mock），"
                     "几何一致**不能**替代数据一致")
    else:
        parts.append("几何非官方：A=world%d/mbs%d/gas%d，B=world%d/mbs%d/gas%d" % (ga + gb))
        parts.append("⇒ 一律不可对官方")
    parts.append("本对照回答的是：%s（须连同**显存峰值**一起读：本类差异常常是"
                 "「用显存换速度」而不是纯粹的更快）" % (note or "<未声明：调用者应给出 --note>"))
    return "；".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", default="/root/qwen35-ascend-migrator")
    ap.add_argument("--config-a", dest="config_a")
    ap.add_argument("--config-b", dest="config_b")
    ap.add_argument("--name-a", dest="name_a", default="A")
    ap.add_argument("--name-b", dest="name_b", default="B")
    ap.add_argument("--declare", default="",
                    help="逗号分隔的**期望差异键**；实际差异必须与之逐键相等，否则拒绝")
    ap.add_argument("--note", default="",
                    help="这次对照**回答什么问题**（写进 summary.comparability_note，避免下游误读）")
    ap.add_argument("--order", default="A,B,B,A,A,B")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--numeric-steps", type=int, default=2, dest="numeric_steps")
    ap.add_argument("--numeric-tol", type=float, default=2.0, dest="numeric_tol")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mind", default="/root/MindSpeed-MM")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    for n in ("config_a", "config_b", "out"):
        if not getattr(a, n):
            print("T63_FAIL 缺 --%s" % n.replace("_", "-"))
            return 2
    if os.path.exists(a.out):
        print("T63_FAIL 输出目录已存在（拒绝覆盖）：%s" % a.out)
        return 3

    cfg_a = yaml.safe_load(open(a.config_a, encoding="utf-8"))
    cfg_b = yaml.safe_load(open(a.config_b, encoding="utf-8"))
    declared = [x.strip() for x in a.declare.split(",") if x.strip()]
    actual = diff_keys(cfg_a, cfg_b)

    print("T63_START out=%s steps=%d order=%s" % (a.out, a.steps, a.order))
    print("T63_GBS a=%d b=%d（红线 8）" % (gbs_of(cfg_a), gbs_of(cfg_b)))
    print("T63_DECLARED %s" % declared)
    print("T63_ACTUAL   %s" % actual)
    if gbs_of(cfg_a) != 8 or gbs_of(cfg_b) != 8:
        print("T63_FAIL 两组 GBS 都必须为 8（GBS=mbs×gas×world）")
        return 3
    if int(dig(cfg_a, "training.micro_batch_size")) != int(dig(cfg_b, "training.micro_batch_size")):
        print("T63_FAIL 两组 micro_batch_size 必须一致"
              "（否则每步计算粒度不同，A/B 就不再「只差被声明的键」）")
        return 3
    if sorted(actual) != sorted(declared):
        only_decl = [k for k in declared if k not in actual]
        only_act = [k for k in actual if k not in declared]
        print("T63_FAIL 实际差异与声明不一致 —— 声明了没变=%s，变了没声明=%s" % (only_decl, only_act))
        return 3

    order = [x.strip().upper() for x in a.order.split(",") if x.strip()]
    if sorted(set(order)) != ["A", "B"] or len(order) < 3:
        print("T63_FAIL --order 必须同时含 A/B 且至少 3 组（示例 A,B,B,A,A,B）")
        return 3
    os.makedirs(a.out)
    boot = open("/proc/sys/kernel/random/boot_id").read().strip()

    results = []
    for i, v in enumerate(order):
        cfg = cfg_a if v == "A" else cfg_b
        rec = run_one("%02d_%s" % (i, v), v, cfg, a, boot)
        results.append(rec)
        json.dump(results, open(os.path.join(a.out, "results.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        if not rec["valid"] or not rec["steps_complete"]:
            print("T63_FAIL 该组无退出证据或步号不齐（valid=%s steps_complete=%s），停止"
                  % (rec["valid"], rec["steps_complete"]))
            return 1

    A = [r for r in results if r["variant"] == "A"]
    B = [r for r in results if r["variant"] == "B"]
    a_mean = sum(r["window_11_end_median_ms"] for r in A) / len(A) if A else None
    drift = ((max(r["window_11_end_median_ms"] for r in A)
              - min(r["window_11_end_median_ms"] for r in A))
             / min(r["window_11_end_median_ms"] for r in A) * 100) if len(A) > 1 else None
    pairs = paired_stats(A, B, order)
    pw, pw_per_pair = pointwise_all(A, B, max_step=(a.numeric_steps or None)) if (A and B) else ({}, [])
    verdict, why = verdict_of(A, B, pairs, pw, False, a.numeric_tol,
                              drift if drift is not None else float("nan"))
    s1a = A[0]["loss"][0] if A and A[0]["loss"] else None
    s1b = B[0]["loss"][0] if B and B[0]["loss"] else None
    anchor_ok = (s1a is not None and s1b is not None and abs(s1a - s1b) < 1e-6)
    if not anchor_ok and verdict not in ("INVALID", "REJECT_NUMERIC"):
        verdict, why = "REJECT_NUMERIC", "两组 step1 loss 不一致（A=%r B=%r）" % (s1a, s1b)

    out = {
        "schema": "t63_two_config_ab.v1",
        "config_a": a.config_a, "config_b": a.config_b,
        "name_a": a.name_a, "name_b": a.name_b,
        "declared_differences": declared, "actual_differences": actual,
        "order_used": order, "steps": a.steps, "numeric_steps": a.numeric_steps,
        "n_A": len(A), "n_B": len(B),
        "baseline_A_mean_ms": round(a_mean, 2) if a_mean else None,
        "baseline_drift_pct": round(drift, 2) if drift is not None else None,
        "paired": pairs, "pointwise_rel_diff_pct": pw, "pointwise_per_pair": pw_per_pair,
        "step1_loss": {"A": s1a, "B": s1b, "anchor_ok": anchor_ok},
        "verdict": verdict, "verdict_reason": why,
        "official_comparable": False,
        # ★ 2026-09-21：原文写死了「两组都不是官方几何（dp1 组）」—— 那是 dp1/dp2 那一案的措辞。
        #   本工具是**通用**双配置配对器（复用者会拿它比"省显存开关组 vs 我方配置"），
        #   写死的理由会把结论带偏 ⇒ 改为**按实际配置推导**：几何是否官方 + 不可对官方的真实原因。
        "comparability_note": _comparability_note(cfg_a, cfg_b, a.note),
        "unclosed_items": [
            "每步全局样本身份未做字节级验证（需 sampler hook 才能落 sample id）——"
            " 本案只声明：同数据文件、同 seed、同 shuffle:false、同 GBS；"
            "**不得**据此宣称「两组每步吃的是同一批样本」",
            "实际 token 吞吐未落盘：日志未提供每步有效 token 数 ⇒ **不予估计**（不编数）",
        ],
        "runs": [{k: r[k] for k in ("tag", "variant", "valid", "steps_complete", "world_size",
                                    "gbs", "window_11_end_median_ms", "median_step_s",
                                    "samples_per_s", "wall_seconds", "steps_parsed",
                                    "max_allocated_mb", "cgroup_peak_max_usage_bytes",
                                    "config_sha256", "identity")} for r in results],
    }
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("T63_RESULT n_A=%d n_B=%d drift=%s mean_delta_pct=%s ci95=%s"
          % (len(A), len(B), out["baseline_drift_pct"], pairs.get("mean_delta_pct"),
             pairs.get("ci95_delta_pct")))
    print("T63_VERDICT %s — %s" % (verdict, why))
    print("T63_OK out=%s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
