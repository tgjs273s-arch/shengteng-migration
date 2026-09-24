#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""_pdf_check.py —— 检查交付 PDF 是否与 v2 决策记录的**更正后数字**一致（G7 的核心风险）

G7 原本是"报告/PDF 与 md 是否一致"的人工项。人工比对整篇不现实，但**最关键的一类不一致可以自动查**：
交付的 PDF 若仍写着被证伪的旧数字（`557.0`、`24.391`、"前两步即保障"），就会与 v2 决策记录
**正面矛盾**——那是最容易让评委看出问题的形态。

本脚本：抽 PDF 文本 → 查"旧数字"与"新数字"是否出现 → 给出判定。
**没有 PDF 库时如实报告"无法自动化"**，不假装通过（人工项仍是人工项）。
"""
import os
import re
import sys

DL = r"C:\Users\HUAWEI\Desktop\C4AI复赛_交付材料"
TARGETS = [
    ("04_性能测试报告", "性能测试报告.pdf"),
    ("03_精度分析报告", "精度分析报告.pdf"),
    ("02_README", "README_镜像环境与运行说明.pdf"),
    ("01_项目创意书", "项目创意书_内容预览.pdf"),
]
OLD = ["557.0", "24.391", "24.39", "19.07"]          # v1 口径/旧数（部分已被证伪或降级）
NEW = ["19.606", "562.55", "22.411", "26.372"]       # v2 口径/更正后数字


def extractor():
    try:
        import pypdf
        return lambda p: "\n".join((pg.extract_text() or "") for pg in pypdf.PdfReader(p).pages), "pypdf"
    except Exception:
        pass
    try:
        import PyPDF2
        return lambda p: "\n".join((pg.extract_text() or "") for pg in PyPDF2.PdfReader(p).pages), "PyPDF2"
    except Exception:
        pass
    return None, None


def main():
    fn, name = extractor()
    if fn is None:
        print("PDF_CHECK_UNAVAILABLE 本机没有 pypdf/PyPDF2 ⇒ **无法自动化**，"
              "G7 保持人工项（不假装通过）")
        return 3
    print("抽文本库=%s" % name)
    for folder, f in TARGETS:
        p = os.path.join(DL, folder, f)
        if not os.path.isfile(p):
            print("  MISSING %s" % p)
            continue
        try:
            txt = fn(p)
        except Exception as exc:
            print("  FAIL %s 抽取失败：%s: %s" % (f, type(exc).__name__, exc))
            continue
        o = [x for x in OLD if x in txt]
        n = [x for x in NEW if x in txt]
        print("  %-34s 字符数=%-7d 旧数字命中=%s 新数字命中=%s" % (f, len(txt), o or "无", n or "无"))
        if o and not n:
            print("      ! 该 PDF 只含旧数字 ⇒ 与 v2 决策记录矛盾，需重生成或声明差异")
    return 0


if __name__ == "__main__":
    sys.exit(main())
