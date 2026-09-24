# -*- coding: utf-8 -*-
r"""_regen_perf_pdf.py —— 用**当前 md（已含更正横幅）**重新印 PDF，并按"抽取正文开头"验证

上一版 bug：脚本把"插入横幅**之前**读到的字符串"传给了 markdown 转换器 ⇒
**md 有横幅、PDF 没有**（而我还把该 PDF 改名成了规范名 ⇒ 交付里那个 PDF 仍含未披露口径的表述）。
本版：**重新从磁盘读 md**（确保带横幅），印完后**打印抽取正文的前 400 字**作为验证
（不比子串——中文 PDF 的抽取可能把字切开；直接看开头最可靠）。
"""
import io
import os
import subprocess
import sys

DL = r"C:\Users\HUAWEI\Desktop\C4AI复赛_交付材料"
FOLDER = os.path.join(DL, "04_性能测试报告")
MD = os.path.join(FOLDER, "性能优化报告.md")
PDF = os.path.join(FOLDER, "性能测试报告.pdf")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

s = io.open(MD, encoding="utf-8").read()          # ★ 从磁盘重读（带横幅）
print("md 含更正横幅=%s（长度 %d）" % ("更正说明（" in s, len(s)))
import markdown as mdlib
body = mdlib.markdown(s, extensions=["tables", "fenced_code"])
html = os.path.join(FOLDER, "_regen.html")
io.open(html, "w", encoding="utf-8").write(
    "<html><head><meta charset='utf-8'><style>"
    "body{font-family:'Microsoft YaHei',sans-serif;margin:32px;line-height:1.6;font-size:13px}"
    "table{border-collapse:collapse}td,th{border:1px solid #999;padding:4px 8px}"
    "h1,h2{border-bottom:1px solid #ccc}</style></head><body>" + body + "</body></html>")

tmp = os.path.join(FOLDER, "_regen.pdf")
r = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
                    "--print-to-pdf=" + tmp, "file:///" + html.replace("\\", "/")],
                   capture_output=True, text=True, timeout=240)
ok = os.path.isfile(tmp) and os.path.getsize(tmp) > 20000
print("Chrome rc=%s 生成=%s 大小=%s" % (r.returncode, ok, os.path.getsize(tmp) if os.path.isfile(tmp) else "-"))
if not ok:
    print("REGEN_FAIL **未覆盖交付 PDF**（fail-closed）")
    sys.exit(1)

import pypdf
txt = "\n".join((pg.extract_text() or "") for pg in pypdf.PdfReader(tmp).pages)
head = " ".join(txt.split())[:400]
print("---- 抽取正文开头 400 字（验证用）----")
print(head)
print("---- 判据：开头必须出现更正的痕迹 ----")
hit = ("更正" in txt) or ("不可比" in txt) or ("mock" in txt.lower())
print("含更正痕迹=%s ⇒ %s" % (hit, "**可以覆盖**" if hit else "**不得覆盖**（旧口径仍在）"))
if not hit:
    sys.exit(1)

os.replace(tmp, PDF)          # 原子替换
print("REGEN_OK 已用带横幅版本覆盖 %s（%d B）" % (PDF, os.path.getsize(PDF)))
try:
    os.remove(html)
except OSError:
    pass
