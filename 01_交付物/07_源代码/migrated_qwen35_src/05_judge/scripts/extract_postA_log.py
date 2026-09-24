#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_postA_log.py — R2-SK04 · 从 hiascend 官方帖 JSON 还原「基准 A」训练日志原文

用途
----
官方基准 A（docs/raw/reference_log_iterations.csv 的出处）是 hiascend 交流圈官方账号
"赛事小助手"2026-06-27 帖《精度测试日志》（运行日期 2026-06-08）。该帖 JSON 抓取件
（docs/raw/hiascend_post_精度测试日志.json）是单行 JSON，正文以 HTML 内嵌在
data.result.content 里，无行号、不便审计引用。

本脚本把帖正文里的日志还原成行文本（仅做 HTML 结构解码，不改任何日志内容），
使 officialA.yaml 的"逐维出处注释"可以引用**带行号的证据文件**：

    evidence/officialA_configuration_details.txt   ← 本脚本默认输出

该证据文件里的行号即 officialA.yaml fingerprint.*.evidence[].line 的取值依据
（行号以本证据文件为基准，1 起；行内容与帖子正文逐字符一致，仅去掉 <p> 等标记）。

诚实标注
--------
- 纯 CPU、仅 Python 标准库；不依赖 pyyaml/torch/A2。
- 行号是对"提取产物"的行号（帖子 JSON 本身是单行，无原生行号）；提取产物已随 skill
  入库（evidence/ 下），可用本脚本 --check 复核其与帖子原文的一致性（防证据漂移）。
- 不对日志内容做任何解释/增删；解释见 officialA.yaml 的 note 字段与 README。

用法
----
  python3 scripts/extract_postA_log.py \
      --post docs/raw/hiascend_post_精度测试日志.json \
      [--out evidence/officialA_configuration_details.txt]   # 默认如上（相对 skill 根）
  python3 scripts/extract_postA_log.py --check [--out <同前>]  # 复核已入库产物，不改写

退出码：0=成功/复核一致；2=输入/IO 错误；3=复核不一致（--check 时）。
"""

import argparse
import json
import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path


def find_repo_root(start):
    """安全向上查找复赛根（含 snapshots/ 或 docs/official/）。

    ★ 坑 30 修复：原实现用 `skill_root.parents[3]` 硬编码深度，在本 Skill 被打包分发到
    任意路径（如 /root/qwen35-ascend-migrator）时直接 IndexError 崩溃。
    现改为：env `SK04_REPO_ROOT` → 逐级向上找标记目录 → 找不到返回 None（绝不抛异常）。
    """
    env = os.environ.get("SK04_REPO_ROOT")
    if env and Path(env).is_dir():
        return Path(env)
    p = Path(start).resolve()
    while True:
        if (p / "snapshots").is_dir() or (p / "docs" / "official").is_dir():
            return p
        if p.parent == p:
            return None
        p = p.parent

# 帖子 JSON 结构中的正文字段定位（内容含 "BaseArguments Configuration Manager Summary"；
# 帖子 HTML 用 U+00A0 不间断空格分隔单词，判定时先把 \xa0/&nbsp; 归一到普通空格）
_CONTENT_RE = re.compile(r"BaseArguments\s+Configuration\s+Manager\s+Summary")


def _normalized(s: str) -> str:
    return s.replace("\xa0", " ").replace("\u2007", " ").replace("&nbsp;", " ")


def find_content_text(raw: str):
    """从单行 JSON 中取出正文 HTML 字符串（不依赖字段名硬编码顺序，双重兜底）。"""
    # 优先按已知结构取 data.result.content
    try:
        doc = json.loads(raw)
    except Exception:
        doc = None
    if doc is not None:
        res = (doc.get("data") or {}).get("result") or {}
        for key in ("content", "contentHtml", "summary", "text"):
            v = res.get(key)
            if isinstance(v, str) and _CONTENT_RE.search(_normalized(v)):
                return v
    # 兜底：正则抓取含关键标记的最大字符串值
    m = re.search(r'"content"\s*:\s*"(.*)"', raw, flags=re.S)
    if m and _CONTENT_RE.search(_normalized(m.group(1))):
        return m.group(1)
    raise ValueError("post JSON 中未找到含 Configuration 内容的正文（content 字段）")


class _TextExtractor(HTMLParser):
    """剥 HTML：<p>/<br> 换行，其余标签剥除，实体解码交给 HtmlParser 自动处理。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag in ("p", "br", "div", "li"):
            self.parts.append("\n")


def html_to_lines(html: str):
    """HTML → 行列表。先处理 JSON 字符串转义，再剥标签。"""
    s = html
    # JSON 字符串内的转义（\" \\ \/ \n 字面量等）
    s = s.replace('\\"', '"').replace("\\/", "/").replace("\\n", "\n")
    # 兼容 \u003c 等被双重转义的情形（正文若出现则还原为 < 等）
    for esc, ch in (("\\u003c", "<"), ("\\u003e", ">"), ("\\u0026", "&"),
                    ("\\u003d", "="), ("\\u0027", "'"), ("\\u00a0", " ")):
        s = s.replace(esc, ch)
    p = _TextExtractor()
    try:
        p.feed(s)
        p.close()
    except Exception:
        pass  # 个别畸形实体不阻断整篇提取
    text = "".join(p.parts)
    text = text.replace("\xa0", " ").replace("\u2007", " ")
    lines = [ln.rstrip() for ln in text.splitlines()]
    # 丢弃纯空行之前/之后的 JSON 尾巴（content 串之后可能粘着相邻字段）
    out, seen_nonempty = [], False
    for ln in lines:
        if ln.strip():
            seen_nonempty = True
        if seen_nonempty:
            out.append(ln)
    while out and not out[-1].strip():
        out.pop()
    return out


def run_extract(post_raw: str) -> list:
    html = find_content_text(post_raw)
    return html_to_lines(html)


def main():
    ap = argparse.ArgumentParser(description="R2-SK04 基准 A 官方帖日志还原")
    ap.add_argument("--post", help="hiascend_post_精度测试日志.json 路径")
    ap.add_argument("--out", help="输出行文本路径（默认 evidence/officialA_configuration_details.txt）")
    ap.add_argument("--check", action="store_true",
                    help="只复核已入库产物与帖子原文一致（不改写）")
    args = ap.parse_args()

    skill_root = Path(__file__).resolve().parents[1]
    repo_root = find_repo_root(skill_root)          # ★ 坑 30：安全查找，找不到则 None
    default_post = ((repo_root / "docs" / "raw" / "hiascend_post_精度测试日志.json")
                    if repo_root else None)          # <复赛根>/docs/raw/...
    default_out = skill_root / "evidence" / "officialA_configuration_details.txt"

    post = Path(args.post) if args.post else default_post
    out = Path(args.out) if args.out else default_out

    # ---------- --check：优先做包内自洽性复核（★ 坑 31：post 不在包内时不阻断） ----------
    if args.check:
        if not out.is_file():
            print("CHECK_MISMATCH missing_evidence_file=%s" % out)
            return 3
        cur = out.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        if post is None or not post.is_file():
            # 打包分发场景：官方帖 JSON 不在包内 → 只能做存在性/完整性复核，明确标注"非全量"
            if not cur:
                print("CHECK_MISMATCH empty_evidence_file=%s" % out)
                return 3
            print("CHECK_PARTIAL_NO_POST evidence_lines=%d file=%s" % (len(cur), out))
            print("  note: 官方帖 JSON 不在包内，**未做重导出一致性比对**（非全量校验）；"
                  "如需全量校验请用 --post 指定或设 SK04_REPO_ROOT 指向复赛根。")
            return 0
        try:
            raw = post.read_text(encoding="utf-8-sig", errors="replace")
            lines = run_extract(raw)
        except (OSError, ValueError) as e:
            print("CHECK_PARTIAL_READ_FAIL %s" % e)
            return 3
        if cur == lines:
            print("CHECK_OK evidence_lines=%d file=%s" % (len(lines), out))
            return 0
        print("CHECK_MISMATCH evidence_drift expected=%d actual=%d file=%s"
              % (len(lines), len(cur), out))
        return 3

    # ---------- 默认：从官方帖 JSON 重导出证据文件（需要 post） ----------
    if post is None or not post.is_file():
        print("FATAL post_json_not_found %s" % post, file=sys.stderr)
        print("  hint: 打包分发场景下官方帖 JSON 不在包内；可用 --post 指定，"
              "或设 SK04_REPO_ROOT 指向复赛根。仅做产物复核请用 --check。",
              file=sys.stderr)
        return 2
    try:
        # hiascend 抓取件带 UTF-8 BOM → utf-8-sig
        raw = post.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as e:
        print("FATAL cannot_read_post %s" % e, file=sys.stderr)
        return 2

    try:
        lines = run_extract(raw)
    except ValueError as e:
        print("FATAL %s" % e, file=sys.stderr)
        return 2

    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
    except OSError as e:
        print("FATAL cannot_write %s (%s)" % (out, e), file=sys.stderr)
        return 2
    print("EXTRACT_OK lines=%d out=%s" % (len(lines), out))
    print("首个日志行: %s" % lines[0][:120] if lines else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
