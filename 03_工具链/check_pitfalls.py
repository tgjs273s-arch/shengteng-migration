#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_pitfalls.py —— 坑表连续性检查（编号 1..N 无缺号、无重复、行数==N）

为什么单独成脚本：坑 142 的教训是"追加把锚点行覆盖掉，文件表面仍自洽"——
编号断号只有**独立检查**看得出来。`_append_pitfall.py` 内部也有同样的断言，
但那是在**写的时候**；本脚本是**随时可跑、只读**的那一份（推送前闸门 G28 用它）。
"""
import os
import re
import sys

from _project_paths import bootstrap_paths
PATHS = bootstrap_paths(parse_cli=__name__ == "__main__")

PKG = PATHS.root
P = os.path.join(PATHS.skill, "docs", "PITFALLS_坑表.md")
ROW = re.compile(r"(?m)^\| \*{0,2}(\d+)\*{0,2} \|")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=P, help="待检查的坑表")
    path = ap.parse_args().path
    if not os.path.isfile(path):
        print("PITFALL_CHECK_FAIL 缺 %s" % path)
        return 2
    text = open(path, encoding="utf-8").read()
    nums = [int(m.group(1)) for m in ROW.finditer(text)]
    uniq = sorted(set(nums))
    dup = sorted({n for n in nums if nums.count(n) > 1})
    if not uniq:
        print("PITFALL_CHECK_FAIL 一条表格行都没识别到（文件被改坏了？）")
        return 1
    miss = [n for n in range(1, uniq[-1] + 1) if n not in uniq]
    print("坑表 %s" % path)
    print("  条数=%d 唯一编号=%d 范围=1..%d 缺号=%s 重复=%s"
          % (len(nums), len(uniq), uniq[-1], miss or "无", dup or "无"))
    if miss or dup or len(nums) != len(uniq):
        print("PITFALL_CHECK_FAIL 编号不连续或重复 ⇒ 不要推送")
        return 1
    print("PITFALL_CHECK_OK items=%d max=%d" % (len(uniq), uniq[-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
