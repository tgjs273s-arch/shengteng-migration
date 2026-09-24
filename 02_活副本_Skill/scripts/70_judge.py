#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
70_judge.py — P7 精度与配置判定（封装判定链三件套 + 哈希链账本）

流程（依次调用 scripts/judge/ 下的工具）：
  1. fingerprint_cfg     配置指纹门（7+2 维偏离 + 可达性谓词）
  2. fingerprint_observed 运行实测指纹（GBS / kernel 生效性 / step1 loss 带）
  3. judge_comparable    五态裁决（输出 verdict_id）
  4. registry_append     追加到 append-only 哈希链账本（可选）
  5. 汇总：额外计算逐点与窗口双轨精度指标（配合官方验收口径不确定性）

用法：
  python3 scripts/70_judge.py --log out/train/train.log \
      --config <cfg.yaml> --data-json <data.json> \
      --baseline officialB --out out/judge/ [--registry evidence/registry.json --tag <tag>]

退出码：0 产出成功（黄/红也属成功产出）；2 输入/IO 错误；3 工具执行失败；4 --gate 时按裁决
"""

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import shutil
import tempfile

SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JUDGE_DIR = os.path.join(SKILL_ROOT, "sk04_judge", "scripts")
BASELINE_DIR = os.path.join(SKILL_ROOT, "sk04_judge", "configs", "baselines")

ITER_RE = re.compile(
    r"iteration\s+(\d+)\s*/\s*(\d+)\s*\|\s*consumed samples:\s*(\d+)"
    r"\s*\|\s*elapsed time per iteration \(ms\):\s*([\d.]+)"
    r"\s*\|\s*learning rate:\s*([\d.Ee+\-]+)"
    r"\s*\|\s*global batch size:\s*(\d+)"
    r"\s*\|\s*loss:\s*([\d.Ee+\-]+)"
    r"\s*\|\s*grad norm:\s*([\d.]+)")


def run_tool(script, args, timeout=300):
    """执行 judge 工具，返回 (rc, stdout, stderr)。"""
    cmd = [sys.executable, os.path.join(JUDGE_DIR, script)] + args
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except Exception as e:
        return 99, "", str(e)


def write_status(directory, state, **details):
    path = os.path.join(directory, "judge_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"execution_state": state, **details}, f, ensure_ascii=False, indent=1)


def checked_tool(script, args, outdir):
    rc, so, se = run_tool(script, args)
    try:
        if rc != 0:
            raise ValueError("exit=%s %s" % (rc, se.strip()[-300:]))
        schemas = {"fingerprint_cfg.py": "sk04_fingerprint_cfg.v1",
                   "fingerprint_observed.py": "sk04_fingerprint_observed.v1",
                   "judge_comparable.py": "sk04_judge_verdict.v1"}
        if script in schemas:
            with open(args[args.index("--out") + 1], encoding="utf-8") as f:
                doc = json.load(f)
            if not isinstance(doc, dict) or doc.get("schema") != schemas[script]:
                raise ValueError("missing or invalid artifact schema")
            if script == "judge_comparable.py":
                reach = doc.get("reachability")
                if (not isinstance(reach, dict) or not reach.get("verdict")
                        or not reach.get("level") or reach.get("color") not in ("green", "yellow", "red")
                        or not doc.get("verdict_id") or not isinstance(doc.get("open_gates"), list)):
                    raise ValueError("incomplete verdict")
        elif script == "registry_append.py" and not any(
                line.startswith("REGISTRY_APPEND_OK ") for line in so.splitlines()):
            raise ValueError("missing registry append acknowledgement")
    except (OSError, ValueError) as exc:
        write_status(os.path.dirname(outdir), "FAILED", attempt_dir=outdir,
                     failed_tool=script, error=str(exc))
        raise RuntimeError("%s: %s" % (script, exc)) from exc
    return rc, so, se


def parse_log(path):
    rows = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = ITER_RE.search(line)
        if m:
            it = int(m.group(1))
            rows[it] = {"loss": float(m.group(7)), "gn": float(m.group(8)), "ms": float(m.group(4))}
    return rows


def pointwise_metrics(ours, official):
    """逐点相对误差 4 指标（Mean/MSE/Max/Min）+ 超 2% 步数。"""
    common = sorted(set(ours) & set(official))
    if not common:
        return None
    def rel(a, b):
        return abs(a - b) / abs(b) * 100.0 if b else 0.0
    le = [rel(ours[i]["loss"], official[i]["loss"]) for i in common]
    ge = [rel(ours[i]["gn"], official[i]["gn"]) for i in common]
    def m(e):
        return {"mean": statistics.mean(e), "mse": sum(x * x for x in e) / len(e),
                "max": max(e), "min": min(e), "median": statistics.median(e)}
    return {"n_common": len(common),
            "loss": m(le), "grad_norm": m(ge),
            "loss_over_2pct_iters": [i for i in common if rel(ours[i]["loss"], official[i]["loss"]) > 2.0],
            "gn_over_2pct_iters": [i for i in common if rel(ours[i]["gn"], official[i]["gn"]) > 2.0],
            "step1_loss_ours": ours[min(common)]["loss"],
            "step1_loss_official": official[min(common)]["loss"]}


def window_metrics(ours, official, lo=50, hi=100):
    def w(rows, key):
        v = [rows[i][key] for i in sorted(rows) if lo <= i <= hi]
        return (statistics.mean(v), statistics.median(v)) if v else (None, None)
    ol, om = w(ours, "loss"); fl, fm = w(official, "loss")
    og, _ = w(ours, "gn"); fg, _ = w(official, "gn")
    out = {}
    if ol and fl:
        out["loss"] = {"ours_mean": ol, "official_mean": fl,
                       "rel_dev_pct": abs(ol - fl) / fl * 100.0,
                       "ours_median": om, "official_median": fm}
    if og and fg:
        out["grad_norm"] = {"ours_mean": og, "official_mean": fg,
                            "rel_dev_pct": abs(og - fg) / fg * 100.0}
    return out


# ---------------------------------------------------------------- world_size 同源解析
# ★ 坑 110：本脚本曾把 `--world-size 1` **硬编码** 传进 fingerprint_cfg（该工具的该参数
#   默认是 None，本意是"从配置/日志取"）。而 fingerprint_cfg 里 `world = world_cli if
#   world_cli is not None else dp_dim["value"]` —— 于是硬编码的 1 **连配置里显式写的
#   data_parallel_size: 2 都会被覆盖**，N_eff 恒为 1×mbs×gas，任何多卡运行必然
#   verdict=NOT_COMPARABLE / neff=4/8。而 run_from_zero.sh 的 P7 正是带 --config 调用
#   → **零到交付链的最后一环永远是红的**，且看起来像"精度不达标"。
#   修法：不再猜 world，改为**同源解析 + 交叉核对 + 拿不到就拒判**（三选一，不猜）。
GBS_RE = re.compile(r"global batch size:\s*(\d+)")
DETAILS_HEADER = "Configuration Details"


def _details_text(path):
    """从训练日志切出 Configuration Details 段文本；找不到 → None。"""
    try:
        raw = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    lines = raw.splitlines()
    i0 = next((i for i, l in enumerate(lines) if DETAILS_HEADER in l), None)
    if i0 is None:
        return None
    i1 = i0 + 1
    while i1 < len(lines) and "============" not in lines[i1]:
        i1 += 1
    return "\n".join(lines[i0 + 1:i1])


def _yaml_doc(text):
    if not text:
        return {}
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except Exception:
        return {}


def _int_at(doc, *path):
    """取整数叶子；bool/auto/缺失 → None。"""
    cur = doc
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    if isinstance(cur, bool):
        return None
    if isinstance(cur, int):
        return cur
    if isinstance(cur, str) and cur.strip().isdigit():
        return int(cur.strip())
    return None


def _facts(doc):
    """从一份配置 doc 里取 world 相关事实。"""
    dp = _int_at(doc, "parallel", "data_parallel_size")
    if dp is None:
        dp = _int_at(doc, "training", "world_size")
    if dp is None:
        dp = _int_at(doc, "world_size")
    return {"dp": dp,
            "fs": _int_at(doc, "parallel", "fully_shard_parallel_size"),
            "mbs": _int_at(doc, "training", "micro_batch_size"),
            "gas": _int_at(doc, "training", "gradient_accumulation_steps")}


def collect_world_evidence(args):
    """收集 world_size 的全部来源（只收集，不判定）。

    ★ 坑 114：**日志内嵌的 Configuration Details 是运行态真相**，而 `--config` 指向的
    文件可能是一份**陈旧产物**（实测机器上的 `out/plan/train_config.yaml` 停留在
    "坑 103 修复前"的 `dp=1/mbs=2/gas=4`，而真实运行是 `dp=2/mbs=4/gas=1`；
    两者 GBS 都恰好 = 8，所以只看 GBS 永远发现不了）。
    因此这里**两个来源都收集**，并在下游把日志侧作为主源、配置侧作为对照。
    """
    cfg_facts = None
    if args.config and os.path.isfile(args.config):
        cfg_facts = _facts(_yaml_doc(open(args.config, encoding="utf-8-sig",
                                         errors="replace").read()))
    log_text = _details_text(args.log)
    log_facts = _facts(_yaml_doc(log_text)) if log_text else None
    log_detail_present = bool(log_text)
    if log_facts is None:
        log_facts = {"dp": None, "fs": None, "mbs": None, "gas": None}

    gbs = None
    try:
        m = GBS_RE.search(open(args.log, encoding="utf-8", errors="replace").read())
        gbs = int(m.group(1)) if m else None
    except OSError:
        pass

    # 主源 = 日志运行态（若存在），否则用户给的 --config
    primary = log_facts if log_detail_present else (cfg_facts or log_facts)
    primary_src = "log_details" if log_detail_present else "config_file"
    log_world = None
    if gbs and primary["mbs"] and primary["gas"]:
        unit = primary["mbs"] * primary["gas"]
        if unit > 0 and gbs % unit == 0:
            log_world = gbs // unit
    return {"cli": args.world_size, "gbs": gbs, "log_world": log_world,
            "primary": primary, "primary_src": primary_src,
            "cfg": cfg_facts, "log": log_facts,
            "log_detail_present": log_detail_present}


def resolve_world_size(ev):
    """决定 world。返回 (value|None, source, notes, fatal, use_log_config)。

    同源原则（ROBUSTNESS.md · INV-2 推论 / 坑 110/114）：
      * 以**日志运行态配置**为主源（`--config-from-log`），它是"实际跑了什么"的真相；
      * `--config` 若与之矛盾 → 告警 `STALE_CONFIG` 并**以日志为准**（不静默采信文件）；
      * 主源显式声明 dp/world → 不传 `--world-size`（保持 source=config）；
      * 否则用主源 fs 整数，再否则用 `gbs ÷ (mbs×gas)` 反推；
      * 任一来源互相矛盾 → **拒判**（rc=2）；都拿不到 → **拒判**（不默认 1）。
    """
    notes, fatal = [], None
    cli, gbs, log_world = ev["cli"], ev["gbs"], ev["log_world"]
    P, C, L = ev["primary"], ev["cfg"], ev["log"]
    use_log = ev["log_detail_present"]

    def _conflict(msg):
        return None, "conflict", notes, msg, use_log

    # ---- 陈旧配置告警（★ 坑 114：文件与运行态不一致，且可能被补偿性参数掩盖）
    if C and ev["log_detail_present"]:
        stale = []
        for k, label in (("dp", "data_parallel_size/world_size"),
                         ("mbs", "micro_batch_size"), ("gas", "gradient_accumulation_steps")):
            if C[k] is not None and L[k] is not None and C[k] != L[k]:
                stale.append("%s: 配置=%s vs 日志=%s" % (label, C[k], L[k]))
        # 日志未显式声明 dp 时，把配置的 dp 与"日志反推的 world"也比一遍
        ref_dp = L["dp"] if L["dp"] is not None else log_world
        if C["dp"] is not None and ref_dp is not None and C["dp"] != ref_dp \
                and not any(s.startswith("data_parallel_size") for s in stale):
            stale.append("data_parallel_size/world_size: 配置=%s vs 日志反推=%s" % (C["dp"], ref_dp))
        if stale:
            notes.append("★ STALE_CONFIG 传入的 --config 与实际运行不同源（以日志为准）：" + "；".join(stale))
            if C["dp"] is not None and ref_dp is not None and C["dp"] != ref_dp:
                notes.append("  该文件的 GBS 可能因补偿性参数（mbs/gas）而与运行态相同 → "
                             "**只看 GBS 红线查不出这类漂移**，必须比对 dp 本身")
    elif C and not ev["log_detail_present"]:
        notes.append("⚠ 日志中无 Configuration Details → 判定只能基于 --config 文件；"
                     "若该文件是早期产物（dp 漂移 + mbs/gas 补偿），本判定**无法自证**，"
                     "请核对它的生成时间与 env.json 档位")

    # ---- 交叉核对（三来源两两）
    dp_primary = P["dp"]
    if cli is not None and dp_primary is not None and cli != dp_primary:
        return _conflict("WORLD_CONFLICT --world-size=%d 与%s 声明的 dp/world=%d 不一致；拒判。"
                         % (cli, ("日志运行态" if ev["primary_src"] == "log_details" else "配置"),
                            dp_primary))
    if cli is not None and log_world is not None and cli != log_world:
        return _conflict("WORLD_CONFLICT --world-size=%d 与日志反推 %d 不一致；拒判。" % (cli, log_world))
    if dp_primary is not None and log_world is not None and dp_primary != log_world:
        return _conflict("WORLD_CONFLICT %s 声明 dp/world=%d，但日志 global batch size=%s ÷ "
                         "(mbs=%s×gas=%s)=%d —— 不同源，拒判。"
                         % (ev["primary_src"], dp_primary, gbs, P["mbs"], P["gas"], log_world))

    if cli is not None:
        return cli, "cli", notes, None, use_log
    if dp_primary is not None:
        notes.append("主源(%s)已显式声明 dp/world=%d → 不传 --world-size（指纹从配置取值，source=config）"
                     % (ev["primary_src"], dp_primary))
        if log_world is not None:
            notes.append("日志交叉核对一致（gbs=%s ÷ %s×%s = %d）" % (gbs, P["mbs"], P["gas"], log_world))
        return None, "config", notes, None, use_log
    if P["fs"] is not None:
        notes.append("主源未声明 dp，但有整数 fully_shard_parallel_size=%d → 以它为准" % P["fs"])
        return P["fs"], "config:fully_shard_parallel_size", notes, None, use_log
    if log_world is not None:
        notes.append("主源未声明 dp → 用日志反推：gbs=%s ÷ (mbs=%s×gas=%s) = %d"
                     % (gbs, P["mbs"], P["gas"], log_world))
        return log_world, "log:gbs/(mbs*gas)", notes, None, use_log
    return None, "unknown", notes, (
        "WORLD_UNKNOWN 无法确定 world_size：日志无 Configuration Details（或其中缺 mbs/gas/dp），"
        "且 --config 未声明 parallel.data_parallel_size / training.world_size。"
        "**拒绝按 1 猜测**（旧版正是因此让多卡运行必判 NOT_COMPARABLE）。请显式传 --world-size <N>。"), use_log


def selftest_world_resolve():
    """world_size 同源解析自检（可证伪：好用例过 / 坏用例拒）。"""
    def ev(cli=None, cfg=None, log=None, gbs=None, present=True):
        base = {"dp": None, "fs": None, "mbs": None, "gas": None}
        C, L = (dict(base, **cfg) if cfg else None), dict(base, **(log or {}))
        P = L if present else (C or L)
        lw = None
        if gbs and P["mbs"] and P["gas"] and (P["mbs"] * P["gas"]) > 0 and gbs % (P["mbs"] * P["gas"]) == 0:
            lw = gbs // (P["mbs"] * P["gas"])
        return {"cli": cli, "gbs": gbs, "log_world": lw, "primary": P,
                "primary_src": "log_details" if present else "config_file",
                "cfg": C, "log": L, "log_detail_present": present}

    cases = [
        ("日志运行态 dp2 + gbs8 一致",
         ev(cfg={"dp": 2, "mbs": 4, "gas": 1}, log={"dp": 2, "mbs": 4, "gas": 1}, gbs=8),
         (None, "config", False)),
        ("★陈旧配置 dp1/mbs2/gas4 vs 日志 dp2/mbs4/gas1",
         ev(cfg={"dp": 1, "mbs": 2, "gas": 4}, log={"dp": 2, "mbs": 4, "gas": 1}, gbs=8),
         (None, "config", False)),
        ("★陈旧配置 dp2 但日志反推 1（以日志为准）",
         ev(cfg={"dp": 2, "mbs": 4, "gas": 1}, log={"mbs": 4, "gas": 1}, gbs=4),
         (1, "log:gbs/(mbs*gas)", False)),
        ("日志无 dp → 用 gbs 反推 2",
         ev(cfg={"mbs": 4, "gas": 1}, log={"mbs": 4, "gas": 1}, gbs=8),
         (2, "log:gbs/(mbs*gas)", False)),
        ("日志无 dp 但有整数 fully_shard=2",
         ev(log={"fs": 2, "mbs": 4, "gas": 1}, gbs=None),
         (2, "config:fully_shard_parallel_size", False)),
        ("全缺 → 拒判（不得默认 1）", ev(present=False), (None, "unknown", True)),
        ("单卡 dp1 + gbs8 一致",
         ev(cfg={"dp": 1, "mbs": 8, "gas": 1}, log={"dp": 1, "mbs": 8, "gas": 1}, gbs=8),
         (None, "config", False)),
        ("--world-size 与日志 dp 冲突 → 拒判",
         ev(cli=1, log={"dp": 2, "mbs": 4, "gas": 1}, gbs=8), (None, "conflict", True)),
        ("--world-size 与 gbs 反推冲突 → 拒判",
         ev(cli=1, log={"mbs": 4, "gas": 1}, gbs=8), (None, "conflict", True)),
    ]
    ok = True
    for name, e, (want_v, want_src, want_fatal) in cases:
        v, src, notes, fatal, _ul = resolve_world_size(e)
        got = (v == want_v and src == want_src and bool(fatal) == want_fatal)
        # 陈旧配置场景还必须产生 STALE_CONFIG 告警（否则"以日志为准"是空话）
        if "陈旧配置" in name:
            got = got and any("STALE_CONFIG" in n for n in notes)
        ok = ok and got
        print("  [%s] %-40s world=%-6s src=%-31s fatal=%s%s"
              % ("PASS" if got else "FAIL", name, str(v), src, bool(fatal),
                 " stale_warn=Y" if any("STALE" in n for n in notes) else ""))
    print("WORLD_RESOLVE_SELFTEST_%s cases=%d" % ("OK" if ok else "FAIL", len(cases)))
    return 0 if ok else 1


def _main():
    ap = argparse.ArgumentParser(description="P7 判定链（配置指纹 + 实测指纹 + 五态裁决 + 账本）")
    ap.add_argument("--log", default=None, help="训练日志（从启动开始的完整日志）")
    ap.add_argument("--config", default=None, help="生效配置 yaml（缺省则从日志内嵌 Configuration Details 取）")
    ap.add_argument("--data-json", default=None, help="数据集 json（用于字节/顺序身份核对）")
    ap.add_argument("--baseline", default="officialB", help="officialA | officialB | <yaml 路径>")
    ap.add_argument("--baseline-log", default=None, help="官方基线日志（用于逐点/窗口双轨指标）")
    ap.add_argument("--out", default="out/judge", help="输出目录")
    ap.add_argument("--registry", default=None, help="哈希链账本路径（提供则追加）")
    ap.add_argument("--tag", default=None, help="账本条目 tag（需唯一）")
    ap.add_argument("--world-size", type=int, default=None,
                    help="world_size（缺省=同源解析：配置显式 dp → 用配置；否则用日志 gbs÷(mbs×gas)；"
                         "两者矛盾或都缺 → 拒判 rc=2，不做默认 1 的猜测）")
    ap.add_argument("--selftest", action="store_true", help="只跑 world_size 同源解析自检并退出")
    ap.add_argument("--gate", action="store_true", help="按裁决色退出（0 绿/3 黄/4 红）")
    args = ap.parse_args()

    if args.selftest:
        return selftest_world_resolve()
    if not args.log:
        print("FATAL 需要 --log（或用 --selftest）", file=sys.stderr)
        return 2

    if not os.path.isfile(args.log):
        print("FATAL 日志不存在：%s" % args.log, file=sys.stderr)
        return 2
    if bool(args.registry) != bool(args.tag):
        print("FATAL --registry 与 --tag 必须同时提供", file=sys.stderr)
        return 2
    publishdir = os.path.abspath(args.out)
    os.makedirs(publishdir, exist_ok=True)
    outdir = tempfile.mkdtemp(prefix="attempt-", dir=publishdir)
    write_status(publishdir, "RUNNING", attempt_dir=outdir)
    base = args.baseline
    if os.path.isfile(base) is False and os.path.isfile(os.path.join(BASELINE_DIR, base + ".yaml")):
        base = os.path.join(BASELINE_DIR, base + ".yaml")

    print("== P7 判定链 ==")
    print("日志     : %s" % args.log)
    print("基线     : %s" % base)

    # ---- 1) 配置指纹门（★ 坑 110/114：world 必须同源解析 + 交叉核对，禁止硬编码；
    #        主源取**日志运行态**而不是可能陈旧的 --config 文件）
    ev = collect_world_evidence(args)
    ws, ws_src, ws_notes, ws_fatal, use_log = resolve_world_size(ev)
    P = ev["primary"]
    print("world 证据: cli=%s | 配置文件=%s | 日志运行态=%s | gbs=%s"
          % (ev["cli"], ev["cfg"], ev["log"], ev["gbs"]))
    print("  主源=%s（mbs=%s gas=%s dp=%s）→ gbs 反推 world=%s"
          % (ev["primary_src"], P["mbs"], P["gas"], P["dp"], ev["log_world"]))
    print("world_size : %s（来源 %s）"
          % (ws if ws is not None else ("未传·由主源取值" if not ws_fatal else "未定"), ws_src))
    for n in ws_notes:
        print("           · %s" % n)
    if ws_fatal:
        print("FATAL %s" % ws_fatal, file=sys.stderr)
        write_status(publishdir, "FAILED", attempt_dir=outdir, error=ws_fatal)
        return 2

    fp_args = ["--baseline", base, "--out", os.path.join(outdir, "fp.json")]
    if use_log:
        # ★ 坑 114：日志内嵌 Configuration Details 才是"实际跑了什么"
        fp_args += ["--config-from-log", os.path.abspath(args.log)]
        print("指纹配置源: --config-from-log（日志运行态）")
    elif args.config:
        fp_args += ["--config", os.path.abspath(args.config)]
        print("指纹配置源: --config %s" % os.path.abspath(args.config))
    else:
        fp_args += ["--config-from-log", os.path.abspath(args.log)]
        print("指纹配置源: --config-from-log（未给 --config）")
    if ws is not None:
        fp_args += ["--world-size", str(ws)]
    if args.data_json:
        fp_args += ["--data-json", args.data_json]
    rc, so, se = checked_tool("fingerprint_cfg.py", fp_args, outdir)
    print("\n[1/4] fingerprint_cfg rc=%s" % rc)
    line = [l for l in so.splitlines() if l.startswith("FINGERPRINT_")]
    print("  " + (line[-1] if line else (se.strip()[-300:] or "无输出")))
    if rc != 0 or not os.path.isfile(os.path.join(outdir, "fp.json")):
        print("FATAL 配置指纹门失败", file=sys.stderr)
        return 3

    # ---- 2) 实测指纹
    rc, so, se = checked_tool("fingerprint_observed.py",
                          ["--log", os.path.abspath(args.log),
                           "--declared", os.path.join(outdir, "fp.json"),
                           "--out", os.path.join(outdir, "obs.json")], outdir)
    print("\n[2/4] fingerprint_observed rc=%s" % rc)
    line = [l for l in so.splitlines() if l.startswith("OBSERVED_")]
    print("  " + (line[-1] if line else (se.strip()[-300:] or "无输出")))

    # ---- 3) 裁决
    rc, so, se = checked_tool("judge_comparable.py",
                          ["--run", os.path.join(outdir, "fp.json"),
                           "--observed", os.path.join(outdir, "obs.json"),
                           "--baseline", base,
                           "--out", os.path.join(outdir, "verdict.json")], outdir)
    print("\n[3/4] judge_comparable rc=%s" % rc)
    vline = [l for l in so.splitlines() if l.startswith("JUDGE_")]
    print("  " + (vline[-1] if vline else (se.strip()[-300:] or "无输出")))
    with open(os.path.join(outdir, "verdict.json"), encoding="utf-8") as f:
        verdict = json.load(f)
    reach = verdict["reachability"]
    verdict_name = reach["verdict"]
    level = reach["level"]
    open_gates = verdict["open_gates"]
    color = reach["color"]

    # ---- 5) 双轨精度指标（逐点 + 窗口）
    summary = {"execution_state": "COMPLETED", "attempt_dir": outdir,
               "verdict_id": verdict.get("verdict_id"),
               "verdict": verdict_name,
               "color": color,
               "level": level,
               "officiality": verdict.get("officiality"),
               "deviations": (verdict.get("config_deviation") or {}).get("count_mismatch", 0),
               "open_gates": open_gates}
    if args.baseline_log and os.path.isfile(args.baseline_log):
        ours = parse_log(args.log)
        official = parse_log(args.baseline_log)
        pw = pointwise_metrics(ours, official)
        wm = window_metrics(ours, official)
        summary["pointwise"] = pw
        summary["window_50_100"] = wm
        if pw:
            # 逐点对比 CSV（供精度报告作图）
            with open(os.path.join(outdir, "loss_compare.csv"), "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["iter", "official_loss", "ours_loss", "loss_rel_err_pct",
                            "official_gn", "ours_gn", "gn_rel_err_pct"])
                for i in sorted(set(ours) & set(official)):
                    lr = abs(ours[i]["loss"] - official[i]["loss"]) / official[i]["loss"] * 100
                    gr = abs(ours[i]["gn"] - official[i]["gn"]) / official[i]["gn"] * 100
                    w.writerow([i, official[i]["loss"], ours[i]["loss"], round(lr, 6),
                                official[i]["gn"], ours[i]["gn"], round(gr, 6)])
            print("\n—— 双轨精度指标 ——")
            print("  逐点 loss : Mean %.4f%% | Max %.4f%% | MSE %.6f | 超2%%步数 %d/%d" % (
                pw["loss"]["mean"], pw["loss"]["max"], pw["loss"]["mse"],
                len(pw["loss_over_2pct_iters"]), pw["n_common"]))
            print("  逐点 gn   : Mean %.4f%% | Max %.4f%% | 超2%%步数 %d/%d" % (
                pw["grad_norm"]["mean"], pw["grad_norm"]["max"],
                len(pw["gn_over_2pct_iters"]), pw["n_common"]))
            print("  step1 loss: 我方 %.6f vs 官方 %.6f" % (pw["step1_loss_ours"], pw["step1_loss_official"]))
        if wm.get("loss"):
            print("  窗口50-100: loss 偏差 %.4f%% | gn 偏差 %.4f%%" % (
                wm["loss"]["rel_dev_pct"], wm["grad_norm"]["rel_dev_pct"]))
        print("  对比 CSV  : %s" % os.path.join(outdir, "loss_compare.csv"))
    else:
        print("\n（未提供 --baseline-log，跳过逐点/窗口双轨指标）")

    # ---- 4) 账本
    if args.registry and args.tag:
        rc, so, se = checked_tool("registry_append.py",
                              ["--registry", os.path.abspath(args.registry), "--tag", args.tag,
                               "--run-dir", outdir,
                               "--manifest", os.path.join(outdir, "verdict.json"),
                               "--fingerprint", os.path.join(outdir, "fp.json"),
                               "--observed", os.path.join(outdir, "obs.json")], outdir)
        print("\n[4/4] registry_append rc=%s" % rc)
        print("  " + ((so.strip().splitlines() or [se.strip()[-300:]])[-1]))
    else:
        print("\n[4/4] registry_append 跳过（未指定 --registry/--tag）")

    with open(os.path.join(outdir, "judge_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    # Publish compatibility outputs only after all requested steps succeed; summary last.
    for name in ("fp.json", "obs.json", "verdict.json", "loss_compare.csv", "judge_summary.json"):
        source = os.path.join(outdir, name)
        if os.path.isfile(source):
            shutil.copyfile(source, os.path.join(publishdir, name))

    print("\n=== 判定汇总 ===")
    print("verdict    : %s" % summary["verdict"])
    print("level      : %s" % summary["level"])
    print("verdict_id : %s" % summary["verdict_id"])
    print("open gates : %s" % summary["open_gates"])
    print("产出目录   : %s（fp.json / obs.json / verdict.json / judge_summary.json）" % outdir)
    print("JUDGE_DONE verdict=%s verdict_id=%s" % (summary["verdict"], summary["verdict_id"]))

    if args.gate:
        return {"green": 0, "yellow": 3, "red": 4}.get(color, 3)
    return 0


def main():
    try:
        return _main()
    except (RuntimeError, OSError, ValueError) as exc:
        print("JUDGE_FAILED %s" % exc, file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
