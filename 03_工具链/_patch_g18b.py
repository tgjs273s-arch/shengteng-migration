# -*- coding: utf-8 -*-
r"""_patch_g18b.py —— 修三件：①递归定位打包残留；②从 MANUAL_GATES 摘掉已自动化的 G18；③打印待办

② 的必要性：G18 升为自动闸门后，驱动器仍把它列在 `MANUAL_GATES` ⇒ 同一次输出里
   G18 既 `PASS` 又 `MANUAL`，是**自相矛盾的状态**（比缺一条更坏：它会让人以为"还有一项要人工确认"）。
"""
import io
import os
import shutil
import time

TOOLS = r"C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\工具_重建"
DL = r"C:\Users\HUAWEI\Desktop\C4AI复赛_交付材料"
BAK = os.path.join(DL, "_旧版本备份_20260922", "打包残留")

# ---- ① 递归找可疑文件 ----
found = []
for root, dirs, files in os.walk(os.path.join(DL, "07_源代码")):
    for f in files:
        if ".zip." in f or f.endswith(".zip"):
            found.append(os.path.join(root, f))
print("扫描 07_源代码 下的 zip/残留：%d 个" % len(found))
moved = 0
for p in found:
    name = os.path.basename(p)
    size = os.path.getsize(p)
    with open(p, "rb") as fh:
        head = fh.read(4)
    is_tmp = ".zip." in name                     # 形如 <x>.zip.<数字> ⇒ 打包中断残留
    print("  %-70s %10d B  头部=%r  %s" % (name, size, head, "**判为残留**" if is_tmp else "保留"))
    if is_tmp:
        os.makedirs(BAK, exist_ok=True)
        dst = os.path.join(BAK, name + "." + time.strftime("%H%M%S"))
        shutil.move(p, dst)
        print("    → 移入备份（未删除）：%s" % dst)
        moved += 1
print("STRAY_SCAN_DONE 找到 %d，移走 %d" % (len(found), moved))

# ---- ② 从 MANUAL_GATES 摘掉 G18 ----
P = os.path.join(TOOLS, "prepush_check.py")
s = io.open(P, encoding="utf-8").read()
i = s.find("MANUAL_GATES = [")
j = s.find("]", i)
if i < 0 or j < 0:
    print("MANUAL_FAIL 找不到 MANUAL_GATES")
else:
    block = s[i:j]
    lines = [l for l in block.splitlines(True)
             if not l.lstrip().startswith('("G18"')]
    newblock = "".join(lines)
    if newblock != block:
        s = s[:i] + newblock + s[j:]
        io.open(P, "w", encoding="utf-8", newline="").write(s)
        removed = len(block.splitlines()) - len(newblock.splitlines())
        print("MANUAL_FIXED 已从 MANUAL_GATES 摘掉 G18（%d 行）—— 避免同一次输出里自相矛盾" % removed)
    else:
        print("MANUAL_SKIP G18 不在 MANUAL_GATES 里")
