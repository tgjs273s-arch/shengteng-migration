#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""check_source_consistency.py —— G18：交付「07_源代码」与 Skill 活副本的**逐文件哈希**一致性

为什么把它从人工项变成自动项（外部复核把 G18 列为人工确认，理由是旧的 `build_source_package.py`
已丢失）：**"同路径文件内容不同"是可以自动判定的**，而且它正是最危险的一类不一致
（评委拿到的源码与 Skill 里的实现不同）。所以本工具只自动判这一件事，并把不能自动判的交回人工。

判据（fail-closed）
  · 两侧**都存在且哈希不同** ⇒ **FAIL**（这是硬缺陷，不允许"预期差异"例外）
  · 只有一侧有的文件 ⇒ 打印清单 + 计数（**不判失败**，但要人工确认是否属预期裁剪）
  · 交付侧为空 / 目录缺失 ⇒ FAIL（不能"没东西可查"当通过）

用法：
  python check_source_consistency.py
输出：SOURCE_CONSISTENCY_OK same=N diff=0 only_deliv=A only_skill=B / ..._FAIL ...
"""
import hashlib
import os
import sys

from _project_paths import bootstrap_paths
PATHS = bootstrap_paths(parse_cli=__name__ == "__main__")

DELIV = os.path.join(PATHS.deliverables, "07_源代码")
SKILL = PATHS.skill
# 交付源码包与 Skill 的目录结构不同：交付侧是"源码包"，Skill 侧把代码放在 scripts/ 等子目录。
# 因此按**文件名**（basename）建立索引做对照 —— 同名文件内容必须一致。
SKIP_DIRS = {"__pycache__", ".git", "out", "refs", "tmp", "06_original_poc"}
# ★ 首跑抓到 9 处"同名不同内容"，但复核后其中两类是**我的匹配规则造成的假阳性**：
#   ① 交付包里 `06_original_poc/**` 自带的另一个 `SKILL.md`（PoC 的工件，不是本实现）被配到了 Skill 根目录；
#   ② `sk04_judge/tests/tmp/*.json` 是**运行时临时产物**（`tests/tmp`），本就该不同。
#   因此把这两类排除后再判 —— **先修尺子，再读结论**（否则会把假阳性当缺陷去改真东西）。
SKIP_EXT = {".pyc", ".log", ".safetensors", ".zip", ".pdf"}


def index(root):
    out = {}
    for r, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if os.path.splitext(f)[1] in SKIP_EXT:
                continue
            p = os.path.join(r, f)
            try:
                h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            except OSError:
                continue
            out.setdefault(f, []).append((os.path.relpath(p, root).replace("\\", "/"), h))
    return out


def main():
    for d, name in ((DELIV, "交付 07_源代码"), (SKILL, "Skill 活副本")):
        if not os.path.isdir(d):
            print("SOURCE_CONSISTENCY_FAIL 缺目录：%s（%s）" % (d, name))
            return 2
    a, b = index(DELIV), index(SKILL)
    if not a:
        print("SOURCE_CONSISTENCY_FAIL 交付侧没有可比文件（不能把'没东西可查'当通过）")
        return 1

    same, diff, only_deliv, only_skill = 0, [], [], []
    for fn, ha in sorted(a.items()):
        if fn not in b:
            only_deliv.append(fn)
            continue
        hb = b[fn]
        if any(h in [x[1] for x in hb] for _p, h in ha):
            same += 1
        else:
            diff.append((fn, ha[0][0], hb[0][0], ha[0][1][:12], hb[0][1][:12]))
    only_skill = [fn for fn in sorted(b) if fn not in a]

    print("交付 07_源代码 唯一文件名=%d  Skill 唯一文件名=%d" % (len(a), len(b)))
    print("  同名且内容一致=%d   **同名但内容不同=%d**" % (same, len(diff)))
    for fn, pa, pb, ha, hb in diff[:10]:
        print("    ! %s：交付 %s(%s) vs Skill %s(%s)" % (fn, pa, ha, pb, hb))
    print("  仅交付侧有=%d %s" % (len(only_deliv), only_deliv[:8]))
    print("  仅 Skill 侧有=%d %s" % (len(only_skill), only_skill[:8]))
    if diff:
        print("SOURCE_CONSISTENCY_FAIL 存在**同名不同内容**的源码文件 ⇒ 评委拿到的源码与 Skill 实现不一致")
        return 1
    print("  注：'仅一侧有'的文件**不判失败**，但必须人工确认是否属预期裁剪"
          "（本工具只自动判'同名不同内容'这一最危险的情形）")
    print("SOURCE_CONSISTENCY_OK same=%d diff=0 only_deliv=%d only_skill=%d"
          % (same, len(only_deliv), len(only_skill)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
