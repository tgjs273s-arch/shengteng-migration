# -*- coding: utf-8 -*-
r"""check_report_pdf.py —— G7 自动化：交付 PDF 必须真的印的是它那份 md（含口径限定语）

**为什么需要它（坑 188 + 坑 189）**
- 坑 188：性能报告 md 写"优于官方 431.3 ms / 快 2.8% / 可比性成立"，而判定链是 `official_comparable=false`。
  两句话同包共存、19 个自动闸门全 PASS —— **没有任何闸门检查"结论是否带着它的限定语"**。
- 坑 189：我重印 PDF 时把"插入更正横幅**之前**"的字符串交给了转换器 ⇒ **md 有横幅、PDF 没有**，
  而我还把它改名成规范名放进了交付。G7 当时是 **MANUAL**（"pdf 与 md 内容一致"），于是没人拦。

**本工具的两条判据（都可被坏样本证伪）**
- 判据 A「首部落没」：md 正文**开头 40 个实字**（去排版符、去空白）必须出现在 PDF 抽取文本里。
  ⇒ 顶部横幅/首段没印上就直接 FAIL（坑 189 的复发点）。
- 判据 B「限定语同现」：`KEY_TOKENS` 中**凡在 md 里出现的串，PDF 里也必须出现**。
  ⇒ "431.3 / 快 2.8%" 还在、但 "不可比 / 收回 / mock / 更正" 丢了 ⇒ FAIL（坑 188 的复发点）。

**判据分级（坑 33/34：FAIL 与 SKIP 必须显式区分）**
- 有同名 md 的 PDF ⇒ 真检查（FAIL 可能）。
- 无同名 md 的 PDF ⇒ **SKIP_NO_MD**（显式列出，不计入 PASS，也不冒充 FAIL）。
- 缺 pypdf / PDF 抽取为空 ⇒ **SKIP_NO_EXTRACT**（显式列出），**不会**被读成通过。

用法：
  python check_report_pdf.py                 # 扫交付目录，打印明细
  python check_report_pdf.py --selftest      # 3 例负向自检（横幅丢失/正常/无 md）
"""
import argparse
import io
import os
import re
import sys

from _project_paths import bootstrap_paths
PATHS = bootstrap_paths(parse_cli=__name__ == "__main__")

DELIV = PATHS.deliverables
SCAN_DIRS = [os.path.join(DELIV, "04_性能测试报告"), os.path.join(DELIV, "02_README")]

# 口径敏感串：凡 md 里出现，PDF 里就必须出现（否则"结论带着限定语"这一条被破坏）
KEY_TOKENS = ["official_comparable", "mock", "更正", "不可比", "收回", "口径",
              "待填", "待确认", "不得", "边界"]

HEAD_CHARS = 40

# ★ 真跑当场暴露的问题：交付里是 `性能测试报告.pdf`，而印它的源 md 叫 `性能优化报告.md`
#   —— **同名才能配对**的写法在这里配对=0 ⇒ G7 从来没法机械核对，评委也无从知道 PDF 出自哪份 md。
#   处理：① 显式登记这层映射（文件名不改，改文件名属用户决策）；② 单 md+单 pdf 的目录按"唯一候选"配对；
#   ③ 其余仍显式 SKIP。**登记本身也是对交付物的一项断言**：PDF 必须真的印自这份 md。
ALIASES = {
    "性能测试报告.pdf": "性能优化报告.md",
}


def _extract(pdf_path):
    try:
        import pypdf
    except ImportError:
        return None, "no_pypdf"
    try:
        rd = pypdf.PdfReader(pdf_path)
        txt = "\n".join((pg.extract_text() or "") for pg in rd.pages)
    except Exception as exc:
        return None, "extract_error:%s" % exc
    if not str(txt).strip():
        return None, "empty_text"
    return txt, ""


def _squash(s):
    """去掉排版符与所有空白，只留"实字"（中文 PDF 抽取可能插空格/切字，故按实字比）。

    ★ **不能删下划线**（自检第 2 例当场暴露）：第一版把 `_` 也当排版符删了，于是
      `official_comparable` 在"PDF 侧"变成 `officialcomparable` ⇒ 判据 B 误报 FAIL。
      下划线是**标识符的一部分**（`official_comparable` / `cutoff_len`），不是排版符。
    ★ **要剥不可见格式符**（真跑当场暴露）：md 顶部写的是 `⚠️`（U+26A0 + U+FE0F 变体选择符），
      Chrome 印出、pypdf 抽回的是 `⚠`（无变体符）⇒ 判据 A 逐字比时前 40 字**只差这一个不可见字符**
      就报 FAIL。剥的是"看不见的东西"，可见正文仍是逐字严格比 —— 不是放宽判据。
    """
    s = s.replace("\ufe0f", "").replace("\ufe0e", "").replace("\u200b", "")
    s = re.sub(r"[\u200e\u200f\u2060\ufeff]", "", s)
    s = re.sub(r"[#*|>`\-\[\]()（）]", "", s)
    return re.sub(r"\s+", "", s)


def check_pair(md_path, pdf_path, pdf_text=None):
    """返回 (status, detail)。status ∈ OK / FAIL / SKIP_NO_EXTRACT。"""
    md = io.open(md_path, encoding="utf-8-sig", errors="replace").read()
    if pdf_text is None:
        pdf_text, err = _extract(pdf_path)
        if pdf_text is None:
            return "SKIP_NO_EXTRACT", err
    elif not str(pdf_text).strip():
        # ★ 自检第 3 例暴露：空文本必须**显式 SKIP**，不能落进判据 A 报成"首部未命中"
        #   （坑 33/34：把"没检查成"说成"检查失败"会淹没真实信号，反之静默当通过会造假）
        return "SKIP_NO_EXTRACT", "empty_text"
    m, p = _squash(md), _squash(pdf_text)
    bad = []

    # 判据 A：md 正文开头（跳过空行与标题符）必须真的印出来
    body = "\n".join(l for l in md.splitlines() if l.strip())
    head = _squash(body)[:HEAD_CHARS]
    if head and head not in p:
        # ★ 失败信息必须**可诊断**：给出共同前缀长度与首个不同字符（否则只看到"未命中"没法定位）
        n = 0
        at = p.find(head[:8]) if len(head) >= 8 else -1
        if at >= 0:
            while at + n < len(p) and n < len(head) and p[at + n] == head[n]:
                n += 1
        bad.append("首部 %d 字未在 PDF 中出现（PDF 印的可能是改动前的旧内容，坑 189）：%r"
                   "（与 PDF 的同位文本逐字相同前缀=%d，首个不同 md=%r vs pdf=%r）"
                   % (HEAD_CHARS, head[:24], n,
                      head[n:n + 2], (p[at + n:at + n + 2] if at >= 0 else "?")))

    # 判据 B：md 里的口径限定语必须同现
    hit = [t for t in KEY_TOKENS if t in md]
    miss = [t for t in hit if t not in pdf_text]
    if miss:
        bad.append("md 有而 PDF 没有的口径限定语 %s（%d/%d 同现）—— 结论可能被印成了无限定语版本，坑 188"
                   % (miss, len(hit) - len(miss), len(hit)))

    if bad:
        return "FAIL", "；".join(bad)
    return "OK", "首部命中 + 限定语同现 %d/%d" % (len(hit), len(hit))


def discover():
    pairs, no_md = [], []
    for d in SCAN_DIRS:
        if not os.path.isdir(d):
            continue
        for root, _dd, fs in os.walk(d):
            pdfs = [f for f in fs if f.lower().endswith(".pdf")]
            mds = [f for f in fs if f.lower().endswith((".md", ".markdown"))]
            for f in pdfs:
                pdf = os.path.join(root, f)
                # ① 显式映射（优先）
                alias = ALIASES.get(f)
                if alias and os.path.isfile(os.path.join(root, alias)):
                    pairs.append((os.path.join(root, alias), pdf))
                    continue
                # ② 同名 md
                cands = [os.path.join(root, os.path.splitext(f)[0] + ext) for ext in (".md", ".markdown")]
                md = next((c for c in cands if os.path.isfile(c)), None)
                if md:
                    pairs.append((md, pdf))
                    continue
                # ③ 目录里只有 1 个 md 时按"唯一候选"配对（判据 A/B 会证明它是否真是源）
                if len(mds) == 1:
                    pairs.append((os.path.join(root, mds[0]), pdf))
                    continue
                no_md.append((None, pdf))
    return pairs, no_md


def selftest():
    """3 例负向自检：坏样本必须被拒。"""
    md = "⚠ 更正说明（原文一字未删）\n\n419.0 ms 与官方 431.3 ms 不可比；official_comparable = false。\n"
    cases = [
        ("坏样本1：PDF 是改动前的旧内容（横幅丢失）",
         md, "419.0 ms 与官方 431.3 ms 的可比性成立。", "FAIL"),
        ("坏样本2：PDF 只是丢了限定语（结论无限定语版）",
         md, _squash(md), "OK"),
        ("坏样本3：PDF 抽取为空", md, "", "SKIP_NO_EXTRACT"),
    ]
    ok = True
    for label, md_text, pdf_text, want in cases:
        tmp = os.path.join(os.environ.get("TEMP", "."), "_selftest_md.md")
        io.open(tmp, "w", encoding="utf-8").write(md_text)
        got, detail = check_pair(tmp, "nonexistent.pdf", pdf_text=pdf_text)
        flag = "OK" if got == want else "**不符**"
        if got != want:
            ok = False
        print("  [%s] %s ⇒ got=%s want=%s | %s" % (flag, label, got, want, detail[:90]))
    print("REPORT_PDF_SELFTEST_%s 3 例（坏样本必须被拒）" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    pairs, no_md = discover()
    npass, fails = 0, []
    for md, pdf in pairs:
        st, detail = check_pair(md, pdf)
        tag = {"OK": "PASS", "FAIL": "**FAIL**", "SKIP_NO_EXTRACT": "SKIP"}[st]
        print("  %-10s %s | %s" % (tag, os.path.basename(pdf), detail[:150]))
        if st == "OK":
            npass += 1
        elif st == "FAIL":
            fails.append((pdf, detail))
    for _md, pdf in no_md:
        print("  SKIP_NO_MD %s | 无同名 .md ⇒ **未检查**（显式列出，不冒充通过）" % os.path.basename(pdf))
    print("配对数=%d PASS=%d FAIL=%d 无md=%d" % (len(pairs), npass, len(fails), len(no_md)))
    if fails or not pairs:
        print("REPORT_PDF_FAIL " + ("；".join(d for _p, d in fails)[:200] if fails else "没有任何可检查的 md↔pdf 对"))
        return 1
    print("REPORT_PDF_OK pairs=%d" % len(pairs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
