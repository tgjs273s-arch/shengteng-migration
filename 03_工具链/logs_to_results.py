#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""logs_to_results.py —— 把真机 `train.log` 转成 `numeric_diag.py` 能吃的 results.json

为什么需要：真机产物是**原始日志**（每个 run 一份 `train.log`），而 `numeric_diag.py` 吃的是
结构化 `results.json`。直接写第二份解析器有重复实现的风险，所以：
  · **只用 Rank 0 行**（多 rank 日志里每 rank 都会打印，混起来会把同一步算多次）；
  · 字段口径与既有工具一致：`loss` / `grad_norm` / `iter_ms` / `step`；
  · **解析不到就报错**，绝不产出"看起来完整"的空序列（宁可失败，不要假数据）。

用法：
  python logs_to_results.py --out results.json <run_id>=<train.log> [<run_id>=<train.log> ...]
  # run_id 里用 `#A` / `#B` 标明变体，例如： P0_run1#A=.../train.log
"""
import argparse
import hashlib
import json
import re
import sys

# 与既有工具一致的行格式（实测样例见坑表）：
# [Rank 0 | Local Rank 0] 2026-09-21 15:51:59 INFO [...train_engine:48] =>  [..] iteration      100/     100 |
#   consumed samples: 800 | elapsed time per iteration (ms): 416.9 | learning rate: ... | global batch size: 8 |
#   loss: 1.772759E-05 | grad norm: 0.012 |
LINE = re.compile(
    r"^\[Rank\s+(\d+)\s*\|.*?iteration\s+(\d+)\s*/\s*(\d+).*?"
    r"elapsed time per iteration \(ms\):\s*([\d.]+).*?"
    r"loss:\s*([-\d.eE+]+).*?grad norm:\s*([-\d.eE+]+)", re.S)


FATAL_MARKERS = ("Traceback (most recent call last)", "RuntimeError", "out of memory",
                 "ERR99999", "Aborted", "ChildFailedError")
DUMP_KEY = re.compile(r"^\s*(?:\[[A-Z]+\]\s*)?([A-Za-z_][\w.]*)\s*:\s*(\S.*)$")


STAMP = re.compile(r"\d{8}_\d{6}")
RUNNAME = re.compile(r"(phase1b|aftermab|memab|AB_AVSB|p2_keepckpt)[A-Za-z0-9_]*")
# ★ 实测暴露的第二个易变项：`save: '/root/ops/<RUN>/00_A/checkpoint'` —— 差异键**只有一个**，
#   就是 checkpoint 输出路径里的臂标识（00_A / 03_A / 04_A）。它是**输出位置**，不改变数值语义，
#   但足以让"同配置的三次运行"算出三个不同的配置摘要（假区分）。故一并归一化。
ARMID = re.compile(r"\b\d{2}_[A-Za-z]\b|\brun\d+\b|\bcfg\b")
# ★ 纯"输出位置"的键：不改变数值语义，但随运行/臂变化 ⇒ 从身份摘要里**显式排除并登记**。
#   注意**不排除** load/checkpoint 载入类键：那是"是否从已有权重继续"的语义，排除会掩盖真差异。
IDENTITY_EXCLUDE = ("save",)


def _norm_ident(v):
    """归一化身份值：把"随运行变化"的部分抹平（时间戳/运行目录名）。

    ★ 为什么必须归一化（实测暴露）：三个 A 臂（应同配置）第一版给出**三个不同的配置摘要**
      （dump_34be7ee9 / dump_f4deba4c / dump_9a8c1204）—— 因为 dump 里含运行目录名与时间戳。
      若不归一化，"配置身份不同"会把**同配置的三次运行**判成三个不同配置，
      于是"配置效应"这类比较从一开始就被污染（假区分 = 另一种错判）。
    """
    v = STAMP.sub("<STAMP>", str(v))
    v = RUNNAME.sub("<RUN>", v)
    v = ARMID.sub("<RUN>", v)
    return v.strip()


def parse_identity(lines):
    """从**日志本身**导出：运行状态 + 配置身份（有效配置 dump 的归一化摘要）。

    为什么要有它：外部复核要求"运行状态、配置身份缺失时输出未验证"。这两项在日志里**本来就有**
    （L5 起是配置加载与逐字段 dump，L249 就有 `use_deter_comp: True`），所以**从产物导出**比让我
    口头声明更硬 —— 它随产物变化而变化，改不了口供。
    ★ 边界：这是"**该日志打印出来的有效配置**"的摘要，不是"实际生效配置"的证明；
      而且 dump 区段按"首个 iteration 行之前"截取（启发式），故把行数一起记下来供核对。
    """
    cfg_lines, fatal = [], []
    for ln in lines:
        s = ln.strip()
        if "iteration" in s and re.search(r"iteration\s+\d+\s*/", s):
            break                      # 训练开始 ⇒ dump 区段结束
        m = DUMP_KEY.match(ln.rstrip("\n"))
        if m and not s.startswith("["):
            cfg_lines.append("%s: %s" % (m.group(1), m.group(2).strip()))
        elif m and s.startswith("[INFO]") and ":" in s:
            body = s.split("]", 1)[-1].strip()
            m2 = DUMP_KEY.match(body)
            if m2:
                cfg_lines.append("%s: %s" % (m2.group(1), m2.group(2).strip()))
    for ln in lines:
        for mk in FATAL_MARKERS:
            if mk in ln:
                fatal.append(mk)
    uniq = sorted(set(cfg_lines))
    digest = hashlib.sha256("\n".join(uniq).encode("utf-8")).hexdigest()[:16]
    # 归一化后的"配置身份"（可跨运行比较）+ 落盘键值，供日后 diff 用（不靠回忆）
    cfg_map = {}
    for ln in uniq:
        k, _sep, v = ln.partition(":")
        cfg_map[k.strip()] = _norm_ident(v)
    excluded = sorted(k for k in cfg_map if k in IDENTITY_EXCLUDE)
    for k in excluded:
        cfg_map.pop(k)
    digest_n = hashlib.sha256(
        "\n".join("%s=%s" % kv for kv in sorted(cfg_map.items())).encode("utf-8")).hexdigest()[:16]
    return {"config_sha256": "dump_%s" % digest_n, "config_sha256_raw": "raw_%s" % digest,
            "config_dump_lines": len(uniq), "config_dump": cfg_map,
            "config_excluded_keys": excluded,
            "config_fatal_markers": sorted(set(fatal))[:3]}


def parse_log(path, tag, variant, data_identity=None):
    steps, ms, loss, gn = [], [], [], []
    totals = set()
    ranks = set()
    lines = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            lines.append(line)
            m = LINE.match(line.strip())
            if not m:
                continue
            rank = int(m.group(1))
            ranks.add(rank)
            if rank != 0:                      # ★ 只用 Rank 0
                continue
            steps.append(int(m.group(2)))
            totals.add(int(m.group(3)))        # 日志里声明的总步数（如 100）
            ms.append(float(m.group(4)))
            loss.append(float(m.group(5)))
            gn.append(float(m.group(6)))
    if not steps:
        raise SystemExit("LOGS_TO_RESULTS_FAIL %s：一行都没解析出来（格式变了？不猜，直接失败）" % path)
    if steps != sorted(steps) or len(set(steps)) != len(steps):
        raise SystemExit("LOGS_TO_RESULTS_FAIL %s：步号不单调或有重复：%s" % (path, steps[:12]))
    ident = parse_identity(lines)
    last, total = steps[-1], max(totals)
    status = "%s(%d/%d)" % ("completed" if last == total else "partial", last, total)
    if ident["config_fatal_markers"]:
        status += "+fatal:" + ",".join(ident["config_fatal_markers"])
    rec = {"tag": tag, "variant": variant, "valid": True, "steps_parsed": len(steps),
           "steps": steps,                                   # 全集，供分析器校验 1..N 连续
           "declared_total": sorted(totals),                 # 日志自称的总步数集合
           "step_ids_tail": steps[-5:], "iter_ms": ms, "loss": loss, "grad_norm": gn,
           "rank_seen": sorted(ranks), "source_log": path,
           # ---- 三层身份（分析器要求：缺任一 ⇒ 结论未认证）----
           "status": status,                                 # 运行状态（从日志导出）
           "config_sha256": ident["config_sha256"],          # 配置身份（**归一化后**，可跨运行比较）
           "config_sha256_raw": ident["config_sha256_raw"],  # 未归一化（含运行目录/时间戳）
           "config_dump_lines": ident["config_dump_lines"],
           "config_dump": ident["config_dump"],              # 键值落盘 ⇒ 日后可 diff（不靠回忆）
           "log_sha256": hashlib.sha256("".join(lines).encode("utf-8", "replace")).hexdigest()[:16]}
    if data_identity:
        rec["data_identity"] = data_identity                 # 数据身份：**必须显式声明**（见下）
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-identity", default="",
                    help="**必须显式声明**的数据身份（如 mock_1img_cutoff1024_dp2）——"
                         "分析器要求它存在；它是一项声明，不是从日志推断出来的")
    ap.add_argument("runs", nargs="+", help="形如 <tag>#<A|B>=<train.log 路径>")
    a = ap.parse_args()
    out = []
    for spec in a.runs:
        if "=" not in spec or "#" not in spec.split("=")[0]:
            raise SystemExit("LOGS_TO_RESULTS_FAIL 参数格式应为 <tag>#<A|B>=<path>：%s" % spec)
        left, path = spec.split("=", 1)
        tag, variant = left.split("#", 1)
        out.append(parse_log(path, tag, variant.upper(), a.data_identity or None))
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("LOGS_TO_RESULTS_OK runs=%d out=%s" % (len(out), a.out))
    for r in out:
        print("  %-12s variant=%s steps=%d status=%s cfg=%s(%d 行) data=%s loss[0..2]=%s" %
              (r["tag"], r["variant"], r["steps_parsed"], r["status"], r["config_sha256"],
               r["config_dump_lines"], r.get("data_identity", "**缺**"),
               [round(x, 8) for x in r["loss"][:3]]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
