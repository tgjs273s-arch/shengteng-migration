# -*- coding: utf-8 -*-
r"""_patch_g18c.py —— 修我上一版的行式删除留下的悬空续行

上一版按"行"过滤 MANUAL_GATES，而 G18 条目**跨两行**，于是只删了 `("G18", ...` 那一行，
续行 `"人工确认：…"),` 变成孤儿 ⇒ `SyntaxError: closing parenthesis ')' does not match
opening parenthesis '['`。

教训（今天第 N 次，同一个形状）：**行式编辑不能用于多行结构**；要么按"结构"删（括号配对），
要么先确认目标只有一行。本脚本按**结构**处理：找到 MANUAL_GATES 区间，删掉其中不含 `(` 的孤儿续行。
"""
import io
import os
import py_compile

P = r"C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\工具_重建\prepush_check.py"
s = io.open(P, encoding="utf-8").read()
i, j = s.find("MANUAL_GATES = ["), s.find("]", s.find("MANUAL_GATES = ["))
assert i > 0 and j > i, "找不到 MANUAL_GATES 区间"
block = s[i:j]
lines = block.splitlines(True)
kept, dropped = [], []
for l in lines:
    ls = l.strip()
    # 孤儿判据：是一条以引号开头的续行（不含元组起始 '("G' 且以 '),' 结尾）
    orphan = ls.startswith('"') and ls.endswith("),")
    (dropped if orphan else kept).append(l)
if dropped:
    s = s[:i] + "".join(kept) + s[j:]
    io.open(P, "w", encoding="utf-8", newline="").write(s)
    print("孤儿续行已删除 %d 行：%s" % (len(dropped), [d.strip()[:60] for d in dropped]))
else:
    print("未发现孤儿续行")

try:
    py_compile.compile(P, doraise=True)
    print("COMPILE_OK")
except Exception as exc:
    print("COMPILE_FAIL %s" % exc)
    raise SystemExit(1)

# 复核：MANUAL_GATES 里剩余条目数 + G18 是否只出现在 GATES（自动）里
s2 = io.open(P, encoding="utf-8").read()
mi = s2.find("MANUAL_GATES = [")
mj = s2.find("]", mi)
block2 = s2[mi:mj]
print("MANUAL_GATES 现有条目：%s" % [l.strip().split(",")[0] for l in block2.splitlines() if l.strip().startswith('("')])
auto_g18 = '("G18"' in s2[:mi]
manual_g18 = '("G18"' in block2
print("G18 在自动 GATES 中=%s；在 MANUAL 中=%s（应为 True / False）" % (auto_g18, manual_g18))
