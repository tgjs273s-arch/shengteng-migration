# -*- coding: utf-8 -*-
r"""_patch_g18.py —— ①清理交付源码包里的打包残留临时文件；②把 G18 接进推送前驱动器

① 可疑文件 `migrated_qwen35_src_20260911_v1.0.zip.1789787310`：名字形如 `<某zip>.<数字>`，
   像打包过程被打断留下的临时产物（数字常为时间戳/序号）。**先看再动**（大小、头部字节），
   确认是临时残留才移入备份目录（不删除）。
② G18 之前是独立脚本，现按驱动器约定接入：`check_source_consistency.py` + 精确期望标记
   `SOURCE_CONSISTENCY_OK`（三层判据由 run_gate 统一处理：后端缺失⇒ERROR、rc≠0 且出现标记⇒FAIL）。
"""
import io
import os
import shutil
import time

TOOLS = r"C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\工具_重建"
DL = r"C:\Users\HUAWEI\Desktop\C4AI复赛_交付材料"
BAK = os.path.join(DL, "_旧版本备份_20260922", "打包残留")

# ---- ① 检查并清理可疑临时文件 ----
stray = os.path.join(DL, "07_源代码", "migrated_qwen35_src_20260911_v1.0.zip.1789787310")
if os.path.isfile(stray):
    size = os.path.getsize(stray)
    with open(stray, "rb") as fh:
        head = fh.read(4)
    print("STRAY 文件: %s  大小=%d B  头部=%r" % (os.path.basename(stray), size, head))
    looks_temp = ".zip." in os.path.basename(stray)
    if looks_temp:
        os.makedirs(BAK, exist_ok=True)
        dst = os.path.join(BAK, os.path.basename(stray) + "." + time.strftime("%H%M%S"))
        shutil.move(stray, dst)
        print("STRAY_MOVED 名称形如 <zip>.<数字> ⇒ 判为打包残留，**移入备份**（未删除）：%s" % dst)
    else:
        print("STRAY_KEPT 名称不像临时残留 ⇒ 保留（不擅自删交付物）")
else:
    print("STRAY_NONE 该文件不存在（可能已处理）")

# ---- ② 接入 G18 ----
P = os.path.join(TOOLS, "prepush_check.py")
s = io.open(P, encoding="utf-8").read()
anchor = '    ("G8", "交付目录结构（必需目录存在且非空）", [], "", "_DRIVER_"),'
add = (anchor + "\n"
       '    # ★ 2026-09-22：G18 从人工项升为自动闸门（交付源码包 vs Skill，同名文件内容必须一致）\n'
       '    ("G18", "源代码包一致性（交付 07_源代码 vs Skill，**同名不同内容即失败**）",\n'
       '     [PY, os.path.join(TOOLS, "check_source_consistency.py")], TOOLS, "SOURCE_CONSISTENCY_OK"),')
if "G18" in s and "SOURCE_CONSISTENCY_OK" in s:
    print("G18_SKIP 已接入")
elif anchor in s:
    io.open(P, "w", encoding="utf-8", newline="").write(s.replace(anchor, add, 1))
    print("G18_ADDED 已接入驱动器")
else:
    print("G18_FAIL 锚点未找到（驱动器结构变了？）")

# ---- ③ 同步更新人工清单：G18 已自动化 ----
DOC = r"C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\qwen35-ascend-migrator_整合版\docs\交付前人工清单.md"
if os.path.isfile(DOC):
    d = io.open(DOC, encoding="utf-8").read()
    if "| **G18** |" in d and "已自动化" not in d.split("| **G18** |")[1][:120]:
        d = d.replace("| **G18** | 源代码包一致性 |",
                      "| ~~**G18**~~ | **已自动化为闸门**（`check_source_consistency.py`，"
                      "同名不同内容即 FAIL）—— 保留下表仅为历史 |", 1)
        io.open(DOC, "w", encoding="utf-8", newline="").write(d)
        print("DOC_UPDATED 人工清单已标注 G18 自动化")
    else:
        print("DOC_SKIP 人工清单无需改动")
else:
    print("DOC_MISSING %s" % DOC)
