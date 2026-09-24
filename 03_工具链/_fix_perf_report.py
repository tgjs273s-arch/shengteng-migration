# -*- coding: utf-8 -*-
r"""_fix_perf_report.py —— 性能报告的**口径更正**（P0 交付缺陷）

缺陷（实测）：
  `04_性能测试报告/性能优化报告.md` 与同名 PDF 声称"50–100 步**性能优于官方基线**（419.0 ms vs 官方
  431.3 ms）"、"**快 2.8%**"，并在 L244 明写"419.0 ms 与官方 431.3 ms 的**可比性成立**"；
  而**全文 0 处提及 mock**。可是后续判据链的结论是：本机数据为 **mock** ⇒ `official_comparable=false`，
  **不得**与官方 431.3 ms 做比值（v2 决策记录与外部复核裁决都如此要求）。

处置原则（不删历史、只加更正、可追溯）：
  ① 在 md **顶部插入更正横幅**（保留原文，注明哪些表述被收紧及原因）；
  ② 用 **headless Chrome** 从更正后的 md 重新印 PDF（生成器 `md2pdf.py` 已随坑 180 丢失）；
     ★ 重新生成**失败就不动旧 PDF**（fail-closed），改为写更正说明文件；
  ③ 旧 PDF/md 先备份；
  ④ 写一份独立更正说明进交付物，供评委直接看到口径边界。
"""
import io
import os
import shutil
import subprocess
import sys
import time

DL = r"C:\Users\HUAWEI\Desktop\C4AI复赛_交付材料"
FOLDER = os.path.join(DL, "04_性能测试报告")
MD = os.path.join(FOLDER, "性能优化报告.md")
PDF = os.path.join(FOLDER, "性能测试报告.pdf")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BAK = os.path.join(DL, "_旧版本备份_20260922", "性能报告_更正前")
STAMP = time.strftime("%Y%m%d_%H%M%S")

BANNER = f"""> ## ⚠️ 更正说明（{time.strftime('%Y-%m-%d')} 追加，原文一字未删）
>
> 本报告的**性能数字**（419.0 ms、51.1× 等）是在 **mock 数据**（1 图/样本、`cutoff_len=1024`、
> 512 长度 mock、**非官方 COCO**）上测得，因此：
>
> | 原文表述 | 现在必须如何读 |
> |---|---|
> | "50–100 步性能**优于官方基线**（中位 419.0 ms vs 官方 431.3 ms）"、"**快 2.8%**" | **不可比**：本机数据为 mock，**不得**与官方窗口中位数 431.3 ms 做比值。`official_comparable = false` |
> | L244 "419.0 ms 与官方 431.3 ms 的**可比性成立**" | **收回**：可比性从未成立（数据不同）；差异只能与**同机同数据的对照臂**比较 |
> | L325 "相对官方 431.3 ms 是 **1.51×**" | **不作为结论**：同上前提不成立 |
>
> **仍然成立**的部分：① 本报告记录的**同机优化链路**（算子实现 3.44×、几何 2.04×、硬件 1.80×、
> 几何细化 1.71/1.32/1.37×、关闭过期显存开关 1.22×、pregather 1.08% 等）是**同机同数据**的对比，
> 与官方基线无关；② 文末"**性能的节点依赖性**"（同一代码在不同容器运行时段可差 2.27×）仍然有效，
> 且与本项目后续判据链一致。
>
> 权威口径见交付物内 `06_Skill/qwen35-ascend-migrator/docs/推荐配置与保底配置_决策记录_v2_20260922.md`
> （第 0 节口径声明、第 3 节正确性未闭合）与 `04_性能测试报告/口径更正说明.md`。

---

"""

NOTE = """# 性能报告口径更正说明（2026-09-22）

## 一句话

交付的《性能测试报告》里**所有与官方基线 431.3 ms 的比较**（"优于官方"、"快 2.8%"、"1.51×"、
"可比性成立"）**均不成立**，因为本机实验数据是 **mock**，而官方数字来自 **COCO 官方数据**。
两者的差异**不能互相比值**；本项目判据链的结论是 `official_comparable = false`。

## 为什么会出现

报告生成于 **2026-09-11**，早于"数据来源决定可比性"这条判据被确立（其后由外部复核裁决与
`推荐配置与保底配置_决策记录 v2` 明确）。报告本身**从未披露数据是 mock** —— 这是缺陷的核心：
**读者无法从报告里看出前提不成立**。

## 哪些仍然可用

1. **同机同数据的优化链路**（本报告主体）：算子实现 3.44× → 几何 2.04× → 硬件换代 1.80× →
   几何细化 1.71×/1.32×/1.37× → 关闭过期显存开关 1.22× → pregather 1.08%；这些是**同一台机器、
   同一数据**上的前后对比，与官方基线无关，**可以用**。
2. **节点依赖性**一节（同一代码在不同容器运行时段可差 2.27×）：与后续判据链一致，**可以用**。
3. **"50–100 步"窗口**：属于项目双口径之一，**可以用**，但只用于同机对比。

## 当前口径（权威）

- 出货口径的时间结论以 `推荐配置与保底配置_决策记录_v2_20260922.md` 为准：
  在**同机同 mock 数据**下，A 相对 B 的省时比例 **19.606%（95% 区间 [18.323%, 20.889%]）**，
  显存峰值 +21.8%；**正确性未闭合**（同配置重复未满足完整窗口逐点 2% 一致性，证据指向反向/梯度
  路径的归约类非确定性）。
- 与官方的比较：**不成立，不得给出**。
"""


def main():
    if not os.path.isfile(MD):
        print("FIX_FAIL 缺 %s" % MD)
        return 2
    os.makedirs(BAK, exist_ok=True)
    # ① 备份
    shutil.copy2(MD, os.path.join(BAK, "性能优化报告.md." + STAMP))
    if os.path.isfile(PDF):
        shutil.copy2(PDF, os.path.join(BAK, "性能测试报告.pdf." + STAMP))
    print("① 已备份 md/pdf 到 %s" % BAK)

    # ② 插入更正横幅（幂等）
    s = io.open(MD, encoding="utf-8").read()
    if "更正说明（" in s and "不可比" in s[:1200]:
        print("② md 已含更正横幅（幂等）")
    else:
        io.open(MD, "w", encoding="utf-8", newline="").write(BANNER + s)
        print("② md 顶部已插入更正横幅（原文保留）")

    # ③ 更正说明文件
    io.open(os.path.join(FOLDER, "口径更正说明.md"), "w", encoding="utf-8", newline="").write(NOTE)
    print("③ 已写 口径更正说明.md")

    # ④ 用 headless Chrome 重新印 PDF（失败即不动旧 PDF）
    if not os.path.isfile(CHROME):
        print("④ PDF_SKIP 找不到 Chrome ⇒ **不动旧 PDF**（旧 PDF 仍含未披露口径的表述，需人工处理）")
        return 1
    html = os.path.join(FOLDER, "_report_tmp.html")
    try:
        import markdown as mdlib
        html_body = mdlib.markdown(s, extensions=["tables", "fenced_code"])
    except Exception as exc:
        print("④ PDF_SKIP markdown 转换失败：%s" % exc)
        return 1
    io.open(html, "w", encoding="utf-8").write(
        "<html><head><meta charset='utf-8'><style>"
        "body{font-family:'Microsoft YaHei',sans-serif;margin:32px;line-height:1.6;font-size:13px}"
        "table{border-collapse:collapse}td,th{border:1px solid #999;padding:4px 8px}"
        "h1,h2{border-bottom:1px solid #ccc}</style></head><body>"
        + html_body + "</body></html>")
    out_pdf = os.path.join(FOLDER, "性能测试报告_更正版.pdf")
    r = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
                        "--print-to-pdf=" + out_pdf, "file:///" + html.replace("\\", "/")],
                       capture_output=True, text=True, timeout=180)
    if os.path.isfile(out_pdf) and os.path.getsize(out_pdf) > 20000:
        print("④ PDF_OK 已生成更正版 PDF：%s（%d B）" % (out_pdf, os.path.getsize(out_pdf)))
    else:
        print("④ PDF_FAIL 生成失败（rc=%s）⇒ **未覆盖旧 PDF**；请人工处理" % r.returncode)
    try:
        os.remove(html)
    except OSError:
        pass
    print("FIX_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
