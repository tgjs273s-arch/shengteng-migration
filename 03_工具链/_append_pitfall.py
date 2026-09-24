# -*- coding: utf-8 -*-
r"""_append_pitfall.py —— 往坑表**安全追加一行**（编号自带校验，杜绝坑 142 复发）

★ 重建说明（2026-09-22）：本文件原先在 `04_核心资产\交付物生成脚本\`，已在坑 180 中被删除。
  这里按**逐字还原**（该文件全文在事故前的会话里被完整读过），并补一道闸门：
  **写盘前先对坑表做时间戳备份**（G-d：会改文件的东西必须先有备份点）。
  其余行为保持不变 —— 尤其那三条"断言不通过坚决不落盘"。

为什么单独写这个脚本
--------------------
坑 142 的教训：拿"上一行"当 `old_string` 做替换来实现"追加"，会把锚点行**覆盖掉**，
而文件表面仍然自洽（编号断号只有闸门看得出来）。所以"追加类编辑"必须：
  1. 锚定**表内最后一行**并在其后插入（不是在锚点处替换）；
  2. 写入前后都断言 **编号 1..N 连续且条数 == N**；
  3. 断言不通过**坚决不落盘**。

用法：
  python _append_pitfall.py --row _row143.md --expect-max 143
  退出码：0 = 已写入且复核通过；3 = 断言失败（未落盘）；2 = 用法/缺文件
"""
import argparse
import os
import re
import shutil
import sys
import time

from _project_paths import bootstrap_paths
PATHS = bootstrap_paths(parse_cli=__name__ == "__main__")

PKG = PATHS.root
# ★ 重建后活副本仍在原处；交给 Codex 的副本目录（10_交给Codex）已在坑 180 中丢失 ⇒ 存在即同步、不存在则跳过
DST = os.path.join(PATHS.skill, "docs", "PITFALLS_坑表.md")
CODEX_COPY = os.path.join(PATHS.root, "10_交给Codex", "07_坑表与契约", "PITFALLS_坑表.md")
BACKUP_DIR = os.path.join(PATHS.tools, "_backup")
ROW_RE = re.compile(r"(?m)^\| \*{0,2}(\d+)\*{0,2} \|.*$")


def nums_of(text):
    return sorted({int(m.group(1)) for m in ROW_RE.finditer(text)})


def continuity(nums):
    return [n for n in range(nums[0], nums[-1] + 1) if n not in nums] if nums else [-1]


def backup(path):
    """G-d：改文件前先留备份点（时间戳）。失败不致命，但必须**打印出来**（不许静默）。"""
    if not os.path.isfile(path):
        return None
    os.makedirs(BACKUP_DIR, exist_ok=True)
    dst = os.path.join(BACKUP_DIR, "%s.%s.bak" % (os.path.basename(path), time.strftime("%Y%m%d_%H%M%S")))
    try:
        shutil.copy2(path, dst)
        return dst
    except Exception as exc:
        print("WARN 备份失败（继续，但请知悉）：%s: %s" % (type(exc).__name__, exc))
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True, help="含单行表格行的文本文件")
    ap.add_argument("--expect-max", type=int, required=True, help="写入后期望的最大编号 N")
    a = ap.parse_args()

    if not os.path.isfile(a.row):
        print("FATAL 找不到 --row 文件：%s" % a.row)
        return 2
    # ★ 容忍 BOM：这个脚本我天天从 PowerShell 调用，而 `Set-Content -Encoding UTF8` 会写 BOM，
    #   导致行首变成 "\ufeff|"，`^\|` 直接不匹配 ⇒ 报"行首不是表格行格式"（今天实测踩到）。
    #   读取用 utf-8-sig（有 BOM 就去掉、没有也无害），比"要求调用者别写 BOM"更可靠。
    new_row = open(a.row, encoding="utf-8-sig").read().strip()
    if "\n" in new_row:
        print("FATAL --row 文件必须**只有一行**（当前 %d 行）" % len(new_row.splitlines()))
        return 2

    m = re.match(r"^\| \*{0,2}(\d+)\*{0,2} \|", new_row)
    if not m:
        print("FATAL 行首不是表格行格式 `| **N** |`：%s" % new_row[:60])
        return 2
    new_num = int(m.group(1))

    # ★ 2026-09-22 补（坑 200 的直接后果）：**写入端执行闸门的规则**。
    #   我连着两次把控制台原文（含真 token）抄进坑表：第一次被打包闸门拦下、第二次（坑 200 自己的
    #   证据列）又被拦下。规律很清楚：**"事后记得脱敏"不可靠，必须让写入动作本身不可写出凭证**。
    #   这里直接复用打包闸门那份 `TOKEN_RE`（单一来源），命中就脱敏并**显式打印**。
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from safe_pack_sync import TOKEN_RE                       # noqa: E402
    if TOKEN_RE.search(new_row):
        n_secret = len(TOKEN_RE.findall(new_row))
        new_row = TOKEN_RE.sub("jt_<REDACTED_ID>:<REDACTED_SECRET>", new_row)
        print("REDACT 该行含 %d 处疑似真 token ⇒ 已按闸门同一份定义脱敏（jt_<REDACTED_ID>:<REDACTED_SECRET>）"
              % n_secret)

    # ★ 2026-09-22 补：**行尾与列数**也必须校验。
    #   实测事故（坑 196）：我把"现场证据"列里嵌了一个换行，于是写入端只看到**第一行**——
    #   行首格式合法 ⇒ 照样写进坑表，但内容被截断、列数少一截（后续那半截还被当成新行去追加 197 ⇒ 报错）。
    #   教训：**格式判据只管一头 = 半截内容照样通过**（与坑 33/34「静默当通过」同族，只是发生在写入端）。
    if not new_row.rstrip().endswith("|"):
        print("FATAL 行尾不是 `|`（列被截断？）：…%s" % new_row[-60:])
        return 2
    if new_row.count("|") < 6:
        print("FATAL 列数不足：找到 %d 个 `|`（表格应为 5 列 = 6 个 `|`）——拒绝写入半截行"
              % new_row.count("|"))
        return 2

    text = open(DST, encoding="utf-8").read()
    nums = nums_of(text)
    print("写入前: 编号 %d..%d 共 %d 条，缺号=%s" % (nums[0], nums[-1], len(nums), continuity(nums) or "无"))

    if new_num in nums:
        print("SKIP 编号 %d 已存在，避免重复追加" % new_num)
        return 0
    if new_num != nums[-1] + 1:
        print("FATAL 新编号 %d 不是 %d 的下一号（拒绝跳号写入）" % (new_num, nums[-1]))
        return 3
    if new_num != a.expect_max:
        print("FATAL 新编号 %d ≠ --expect-max %d" % (new_num, a.expect_max))
        return 3

    rows = list(ROW_RE.finditer(text))
    last = rows[-1]
    if int(last.group(1)) != nums[-1]:
        print("FATAL 表内最后一行编号 %s ≠ 最大编号 %d" % (last.group(1), nums[-1]))
        return 3
    # ★ 在**最后一行之后**插入：绝不覆盖锚点行（坑 142 的错就出在这里）
    text = text[:last.end()] + "\n" + new_row + text[last.end():]

    nums2 = nums_of(text)
    expect = list(range(1, a.expect_max + 1))
    miss = continuity(nums2)
    if miss or nums2 != expect:
        print("FATAL 写出前断言失败：编号 %d..%d 共 %d 条 缺号=%s（期望 1..%d 共 %d 条）—— 拒绝落盘"
              % (nums2[0], nums2[-1], len(nums2), miss or "无", a.expect_max, a.expect_max))
        return 3

    bk = backup(DST)
    print("备份点: %s" % (bk or "（无，文件此前不存在）"))
    with open(DST, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    back = open(DST, encoding="utf-8").read()
    nums3 = nums_of(back)
    ok = (nums3 == expect)
    print("写盘后复核: 编号 %d..%d 共 %d 条，缺号=%s → %s"
          % (nums3[0], nums3[-1], len(nums3), continuity(nums3) or "无", "OK" if ok else "FAIL"))

    if os.path.isfile(CODEX_COPY):
        with open(CODEX_COPY, "w", encoding="utf-8", newline="") as fh:
            fh.write(back)
        same = open(CODEX_COPY, encoding="utf-8").read() == back
        print("Codex 副本同步: %s" % ("一致" if same else "**不一致**"))
        ok = ok and same
    else:
        print("Codex 副本: 跳过（%s 不存在 —— 该目录在坑 180 中丢失）" % CODEX_COPY)

    print("PITFALL_APPEND_%s 新增=%d 条数=%d 编号=1..%d"
          % ("OK" if ok else "FAIL", new_num, len(nums3), nums3[-1]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
