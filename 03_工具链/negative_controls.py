#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""negative_controls.py —— 负向对照：**证明这些闸门真的会失败**（闸门 G6 的实现）

为什么必须有它：坑 35 的教训是"能通过比不能通过更危险" —— 一个**永远不会失败**的检查
等于没有检查。旧驱动器的 G6 内联了 16 条坏样本，随驱动器一起丢失（坑 180）；本脚本按
**现存工具**重建一批坏样本，每一条都要求"工具必须拒绝"。

四项对照（全部在临时目录里做，**不碰任何交付物**）：
  1. 坑表连续性检查：喂一份**编号重复**的假坑表 ⇒ 必须 FAIL
  2. 坑表追加工具：编号与 `--expect-max` 不符 ⇒ 必须 rc=3 且**不写盘**（写盘即判失败）
  3. 对齐审计：喂一份含**未登记偏离**的配置（把 model.gdn_implementation 改成 eager）⇒ 必须 FAIL
  4. 证据清单核对：把清单里某个文件的哈希改掉 ⇒ 必须 FAIL

用法：python negative_controls.py
输出：NEGATIVE_CONTROLS_OK cases=N failed=0 / NEGATIVE_CONTROLS_FAIL …
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

from _project_paths import bootstrap_paths
PATHS = bootstrap_paths(parse_cli=__name__ == "__main__")

HERE = PATHS.tools
PKG = PATHS.root
SKILL = PATHS.skill
PY = sys.executable


def run(argv, cwd=HERE):
    r = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=600)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main():
    fails = []
    td = tempfile.mkdtemp(prefix="negctl_")

    def ck(name, cond, extra=""):
        print("  [%s] %-56s %s" % ("PASS" if cond else "FAIL", name, extra))
        if not cond:
            fails.append(name)

    # ---- 1) 坑表连续性：重复编号必须被抓 ----
    fake = os.path.join(td, "PITFALLS.md")
    with open(fake, "w", encoding="utf-8") as fh:
        fh.write("| **1** | a |\n| **1** | b |\n| **3** | c |\n")
    rc, out = run([PY, os.path.join(HERE, "check_pitfalls.py"), "--path", fake])
    ck("坏坑表（重复+缺号）→ check_pitfalls 必须 FAIL",
       rc != 0 and "PITFALL_CHECK_FAIL" in out, "rc=%s" % rc)

    # ---- 2) 追加工具：编号不符必须拒绝且不写盘 ----
    # ★ 2026-09-22 修：夹具原来写成 `| **999** | 假行 |`（**只有 2 列**）。我给追加工具加了
    #   "行尾/列数"校验（坑 196/197 的教训）之后，这一条**被格式校验先拒掉**（rc=2），
    #   于是 rc=3 的断言失败 —— 而真实性质（拒绝 + 不写盘）其实一直成立。
    #   这暴露的是一个**更隐蔽的坏样本缺陷**：坏样本若因**另一个原因**被拒，就等于**没测到目标**
    #   （判据看起来有效，实际是假通过）。⇒ ① 夹具改为**规范的 5 列行**，让它只测"编号不符"；
    #   ② 断言里**同时检查拒绝原因**（必须提到编号/expect-max），杜绝"因别的原因被拒"混过去。
    row = os.path.join(td, "row.md")
    open(row, "w", encoding="utf-8").write("| **999** | 假坑 | 假因 | 假对策 | 假证据 |\n")
    before = os.path.getsize(os.path.join(SKILL, "docs", "PITFALLS_坑表.md"))
    rc, out = run([PY, os.path.join(HERE, "_append_pitfall.py"), "--row", row, "--expect-max", "999"])
    after = os.path.getsize(os.path.join(SKILL, "docs", "PITFALLS_坑表.md"))
    named = ("expect-max" in out) or ("新编号" in out)
    ck("编号 ≠ --expect-max → 追加工具 rc=3、不写盘、且**拒绝原因指出编号**",
       rc == 3 and before == after and named, "rc=%s 大小 %d→%d 原因命中=%s" % (rc, before, after, named))

    # ---- 2b) 追加工具：**半截行**（列数不足/行尾缺 |）必须拒绝（本轮新增判据的回归测试）----
    trunc = os.path.join(td, "row_trunc.md")
    open(trunc, "w", encoding="utf-8").write("| **999** | 半截行（列数不足） |\n")
    before2 = os.path.getsize(os.path.join(SKILL, "docs", "PITFALLS_坑表.md"))
    rc_t, out_t = run([PY, os.path.join(HERE, "_append_pitfall.py"), "--row", trunc,
                       "--expect-max", "999"])
    after2 = os.path.getsize(os.path.join(SKILL, "docs", "PITFALLS_坑表.md"))
    ck("半截行（列数不足）→ 拒绝且不写盘（这条正是坑 196 事故的回归测试）",
       rc_t == 2 and before2 == after2 and ("列数不足" in out_t or "行尾" in out_t),
       "rc=%s 大小 %d→%d" % (rc_t, before2, after2))

    # ---- 3) 对齐审计：未登记偏离必须被抓 ----
    import yaml
    d = yaml.safe_load(open(os.path.join(SKILL, "config", "templates",
                                         "qwen3_5_0_8B_recommended_A.yaml"), encoding="utf-8"))
    d["model"]["gdn_implementation"] = "eager"      # 官方 triton；且 LEDGER_MAP 里没有这个键的登记
    bad_cfg = os.path.join(td, "bad.yaml")
    yaml.safe_dump(d, open(bad_cfg, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
    rc, out = run([PY, os.path.join(HERE, "align_audit.py"), "--run", bad_cfg])
    ck("含未登记偏离的配置 → 对齐审计必须 FAIL",
       rc != 0 and "ALIGN_AUDIT_FAIL" in out, "rc=%s" % rc)

    # ---- 4) 证据清单：篡改哈希必须被抓 ----
    ev = PATHS.evidence
    if os.path.isdir(ev) and os.path.isfile(os.path.join(ev, "_manifest.json")):
        tmp_ev = os.path.join(td, "ev")
        shutil.copytree(ev, tmp_ev)
        mp = os.path.join(tmp_ev, "_manifest.json")
        man = json.load(open(mp, encoding="utf-8"))
        k = sorted(man)[0]
        man[k]["sha256"] = "0" * 64
        json.dump(man, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        rc, out = run([PY, os.path.join(HERE, "evidence_manifest.py"), "--verify", "--dir", tmp_ev])
        ck("清单哈希被篡改 → 证据核对必须 FAIL",
           rc != 0 and "EVIDENCE_MANIFEST_FAIL" in out, "rc=%s" % rc)
    else:
        ck("清单哈希被篡改 → 证据核对必须 FAIL", False,
           "跳过条件不满足：%s 缺 _manifest.json（先跑 --write）" % ev)

    print("负向对照产物保留: %s（不递归清理）" % td)
    print("NEGATIVE_CONTROLS_%s cases=5 failed=%d" % ("OK" if not fails else "FAIL", len(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
