# -*- coding: utf-8 -*-
r"""_fix_pit_196_197.py —— 修复被截断的坑 196 行，并补写 197

事故：我把 196 的"现场证据"列里**嵌了一个换行**（进程命令行换行排版），
`_append_pitfall.py` 只校验**行首**格式 `| **N** |`，没校验**行尾**，于是：
  · 第 1 行被当成完整行写进了坑表（**行尾缺 `|`**，列数也少了一截）；
  · 续行（`17916 …`）被当成新的行去追加 197 ⇒ 报"行首不是表格行格式"。
判据只管一头 ⇒ 半截内容照样通过（这与坑 33/34「静默当通过」同族，只是发生在**写入端**）。

本脚本：① 备份；② 用**单行**重建 196（把续行并入，用"；"连接，确保以 `|` 收尾）；
③ 追加 197；④ 回读校验连续性 + 每行首尾格式 + 列数。
"""
import io
import os
import re
import shutil
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = r"C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\qwen35-ascend-migrator_整合版\docs\PITFALLS_坑表.md"
SRC = os.path.join(HERE, "_pits_196_197.txt")

raw = io.open(SRC, encoding="utf-8").read().splitlines()
parts = [p for p in raw if p.strip()]
row196 = parts[0]
row197 = parts[-1]
tail = [p.strip() for p in parts[1:-1]]
if tail:                                   # 把被换行拆断的续行并回 196
    row196 = row196.rstrip() + " " + "；".join(tail)
row196 = row196.rstrip()
if not row196.endswith("|"):
    row196 += " |"
assert row196.startswith("| **196** |") and row196.endswith("|"), "196 重建失败"
assert row197.startswith("| **197** |") and row197.endswith("|"), "197 格式不对"
assert "\n" not in row196 and "\n" not in row197, "不得含换行"

lines = io.open(TABLE, encoding="utf-8").read().splitlines()
idx = [i for i, l in enumerate(lines) if l.startswith("| **196** |")]
if len(idx) != 1:
    raise SystemExit("FIX_FAIL 期望恰好 1 行 196，实际 %d" % len(idx))
bak = TABLE + ".bak_%s" % time.strftime("%Y%m%d_%H%M%S")
shutil.copy2(TABLE, bak)
lines[idx[0]] = row196
lines.insert(idx[0] + 1, row197)
io.open(TABLE, "w", encoding="utf-8").write("\n".join(lines) + "\n")

back = io.open(TABLE, encoding="utf-8").read().splitlines()
ids = []
for l in back:
    m = re.match(r"^\|\s*\*\*(\d+)\*\*\s*\|", l)
    if m:
        ids.append(int(m.group(1)))
        if not l.rstrip().endswith("|"):
            raise SystemExit("FIX_FAIL 行 %s 未以 | 结尾（列被截断）" % m.group(1))
        if l.count("|") < 6:
            raise SystemExit("FIX_FAIL 行 %s 列数不足（%d 个 |）" % (m.group(1), l.count("|")))
if ids != list(range(1, len(ids) + 1)):
    raise SystemExit("FIX_FAIL 编号不连续：%s" % ids[-5:])
print("FIX_OK 条数=%d 编号=1..%d" % (len(ids), ids[-1]))
print("  196 长度=%d 字符；197 长度=%d 字符" % (len(row196), len(row197)))
print("  备份=%s" % bak)
