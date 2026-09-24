#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""hash_consistency.py —— 远端**运行侧**代码 与 本地**判据侧**代码的哈希对照（自动）

起因（坑 216）
-------------
§4-2 的端到端 A/B 编排脚本原本在**远端** `import 59_config_ab` 复用判据，
结果崩了：`AttributeError: no attribute 'win_ms_of'`。核对发现**远端那份 Skill 是旧版**
（md5 `218e635b…`），而 `win_ms_of` 正是 `paired_stats` **内部**要用的函数。
崩掉其实是幸运（响亮失败）；真正危险的是它"能跑通但取值口径不同"——
那我会在**毫无报错**的情况下用**别人版本的判据**算出性能数字。

复核要求：**远端运行代码、本地解析代码分别记录哈希**，而且这件事
**不能只靠操作者记得**。本工具把它变成**可自动发现**的问题。

做的事
------
  ① 读一份**单一来源**的文件清单（远端路径 ↔ 本地路径 ↔ 角色）；
  ② 走会话守护取远端 `sha256sum`；
  ③ 本地算 `sha256`；
  ④ 逐文件比对 → 输出 `OK` / `MISMATCH`，并落盘 JSON 产物（可入证据清单）。

角色语义（★ 本工具存在的理由）
------------------------------
  `run_side`    远端**执行**用（`50_train.py`、框架 `train_engine.py` 等）
  `judge_side`  判定用（交付副本 `59_config_ab.py` / `62_reportability.py` / 本地分析器）
  ★ 规则：**判据必须来自 judge_side（交付副本）**。若某文件在远端**被当作判据**使用，
    而它与交付副本不一致 ⇒ 判 `JUDGE_SKEW`（**禁止在远端据此判据**），而非仅仅"提示不同"。

用法
----
    python hash_consistency.py                 # 对照 + 落盘 out/hash_consistency.json
    python hash_consistency.py --out <路径>
    python hash_consistency.py --selftest      # 判据自检（不连远端）
退出码：0 = 无 JUDGE_SKEW；6 = 存在 JUDGE_SKEW（判据偏斜）；5 = 拿不到远端哈希；2 = 用法错
"""
import argparse
import hashlib
import json
import os
import sys

from _project_paths import bootstrap_paths
PATHS = bootstrap_paths(parse_cli=__name__ == "__main__")

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = PATHS.root
ASSETS = PATHS.root
SKILL = PATHS.skill
RSKILL = "/root/qwen35-ascend-migrator"
MIND = "/root/MindSpeed-MM"

# ★ 单一来源：远端路径 ↔ 本地路径 ↔ 角色 ↔ 是否在远端被当作判据
FILES = [
    # --- 判据侧（本地判定必须用交付副本）---
    {"name": "59_config_ab.py", "role": "judge_side", "judge_on_remote": False,
     "remote": RSKILL + "/scripts/59_config_ab.py",
     "local": os.path.join(SKILL, "scripts", "59_config_ab.py")},
    {"name": "62_reportability.py", "role": "judge_side", "judge_on_remote": False,
     "remote": RSKILL + "/scripts/62_reportability.py",
     "local": os.path.join(SKILL, "scripts", "62_reportability.py")},
    {"name": "_envcompat.py", "role": "judge_side", "judge_on_remote": False,
     "remote": RSKILL + "/scripts/_envcompat.py",
     "local": os.path.join(SKILL, "scripts", "_envcompat.py")},
    # --- 运行侧（远端执行用；差异仅作提示，除非被当判据）---
    {"name": "50_train.py", "role": "run_side", "judge_on_remote": False,
     "remote": RSKILL + "/scripts/50_train.py",
     "local": os.path.join(SKILL, "scripts", "50_train.py")},
    {"name": "56_profile_run.py", "role": "run_side", "judge_on_remote": False,
     "remote": RSKILL + "/scripts/56_profile_run.py",
     "local": os.path.join(SKILL, "scripts", "56_profile_run.py")},
    {"name": "70_judge.py", "role": "run_side", "judge_on_remote": True,
     "remote": RSKILL + "/scripts/70_judge.py",
     "local": os.path.join(SKILL, "scripts", "70_judge.py")},
    # --- 框架（本地无对应；只记录哈希，供追溯）---
    {"name": "train_engine.py", "role": "framework", "judge_on_remote": False,
     "remote": MIND + "/mindspeed_mm/fsdp/train/train_engine.py", "local": None},
    {"name": "clip_grad_norm.py", "role": "framework", "judge_on_remote": False,
     "remote": MIND + "/mindspeed_mm/fsdp/optimizer/clip_grad_norm.py", "local": None},
]


def sha256_local(path):
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(entry, remote_hex, local_hex):
    """返回 (status, severity, note)。severity: OK / INFO / JUDGE_SKEW / UNKNOWN"""
    if remote_hex is None:
        return "REMOTE_UNREADABLE", "UNKNOWN", "远端取不到哈希（会话断了？路径不存在？）"
    if local_hex is None:
        return "LOCAL_MISSING", ("JUDGE_SKEW" if entry["role"] == "judge_side" else "INFO"), \
            "本地交付副本不存在 ⇒ 无法证明远端与交付一致"
    if remote_hex == local_hex:
        return "MATCH", "OK", "远端与交付副本逐字节一致"
    if entry.get("judge_on_remote"):
        return "MISMATCH", "JUDGE_SKEW", \
            "★ 该文件**在远端被当作判据使用**，且与交付副本不一致 ⇒ 禁止用远端这份下判据"
    if entry["role"] == "judge_side":
        return "MISMATCH", "INFO", \
            "判据侧副本不一致（只要**不在远端**用它判据即安全；判据一律回本地用交付副本）"
    return "MISMATCH", "INFO", "运行侧副本不同（仅信息；判据不依赖它）"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out", "hash_consistency.json"))
    ap.add_argument("--selftest", action="store_true", default=False)
    args = ap.parse_args()

    if args.selftest:
        cases = []
        e_judge = {"name": "x", "role": "judge_side", "judge_on_remote": False}
        e_run = {"name": "y", "role": "run_side", "judge_on_remote": False}
        e_remotejudge = {"name": "z", "role": "run_side", "judge_on_remote": True}
        cases.append(("一致 ⇒ MATCH/OK", classify(e_judge, "a", "a")[:2] == ("MATCH", "OK")))
        cases.append(("远端读不到 ⇒ UNKNOWN（不得当通过）",
                      classify(e_judge, None, "a")[:2] == ("REMOTE_UNREADABLE", "UNKNOWN")))
        cases.append(("本地缺失 + 判据侧 ⇒ JUDGE_SKEW",
                      classify(e_judge, "a", None)[1] == "JUDGE_SKEW"))
        cases.append(("本地缺失 + 运行侧 ⇒ INFO",
                      classify(e_run, "a", None)[1] == "INFO"))
        cases.append(("远端被当判据且不一致 ⇒ JUDGE_SKEW（★ 核心用例）",
                      classify(e_remotejudge, "a", "b")[1] == "JUDGE_SKEW"))
        cases.append(("判据侧不一致但不在远端判据 ⇒ INFO",
                      classify(e_judge, "a", "b")[1] == "INFO"))
        bad = [c for c in cases if not c[1]]
        for n, ok in cases:
            print("  %s %s" % ("PASS" if ok else "FAIL", n))
        if bad:
            print("HASH_CONSISTENCY_SELFTEST_FAIL cases=%d failed=%d" % (len(cases), len(bad)))
            return 1
        print("HASH_CONSISTENCY_SELFTEST_OK cases=%d failed=0" % len(cases))
        return 0

    sys.path.insert(0, HERE)
    import pull_evidence                                    # 复用会话守护客户端（单一来源）
    remote = {}
    for e in FILES:
        r = pull_evidence.run("sha256sum %s 2>/dev/null | awk '{print $1}'" % e["remote"])
        out = (r.get("stdout") or "").strip().split()
        remote[e["name"]] = out[0] if (r.get("ok") and out) else None

    rows, skew = [], 0
    for e in FILES:
        rh, lh = remote.get(e["name"]), sha256_local(e["local"])
        status, severity, note = classify(e, rh, lh)
        if severity == "JUDGE_SKEW":
            skew += 1
        rows.append({"name": e["name"], "role": e["role"],
                     "judge_on_remote": bool(e.get("judge_on_remote")),
                     "remote_path": e["remote"], "local_path": e["local"],
                     "remote_sha256": rh, "local_sha256": lh,
                     "status": status, "severity": severity, "note": note})
        print("  %-22s %-11s %-10s remote=%s local=%s"
              % (e["name"], e["role"], status,
                 (rh or "-")[:12], (lh or "-")[:12]))
        print("        %s" % note)

    doc = {"schema": "dsh.hash_consistency.v1",
           "purpose": "远端运行侧代码 与 本地判据侧代码 的哈希对照（复核要求：分别记录，不靠人记）",
           "rule": "★ 判据必须来自交付副本（judge_side）；远端那份只允许『跑』，不允许『判』",
           "files": rows, "judge_skew_count": skew}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    print("HASH_CONSISTENCY_OUT %s" % args.out)
    if skew:
        print("HASH_CONSISTENCY_JUDGE_SKEW n=%d ⇒ 存在被判据使用的偏斜副本，禁止在远端下判据" % skew)
        return 6
    print("HASH_CONSISTENCY_OK files=%d judge_skew=0" % len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
