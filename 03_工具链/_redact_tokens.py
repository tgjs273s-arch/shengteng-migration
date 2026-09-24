# -*- coding: utf-8 -*-
r"""_redact_tokens.py —— 把写进交付文档的**真 token** 脱敏（内容闸门拦住了出包，这是它的正确行为）

事故：坑 198 的"现场证据"列里我贴了控制台给的连接命令原文，其中含
`jt_<ID>:<长HEX>` —— **真凭证**。`safe_pack_sync.py --apply` 的内容闸门报
`! docs\PITFALLS_坑表.md 含疑似真 token（jt_<ID>:<HEX>）` ⇒ **拒绝出包**（zip 未重建 ⇒ G22 也 FAIL）。
⇒ 判据链在这里是**正确**的：凭证进了交付物 = 必须停下来，而不是"先出了再说"。

本脚本：扫描 Skill 活副本下所有文本文件，把 `jt_<ID>:<HEX>` 全量替换为脱敏形式
`jt_<REDACTED_ID>:<REDACTED_SECRET>`（保留前缀 `jt_` 以便读者知道那是什么字段），
写盘前备份、写盘后回读校验，并打印每处替换的文件与次数。
"""
import io
import os
import shutil
import sys
import time

# ★ 直接复用**打包闸门自己的**定义（单一来源）。第一版我在这里另写了一个更严的正则，
#   结果"脱敏 0 处命中、闸门却仍报命中" —— 判定与脱敏用了两套定义就必然分叉。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_pack_sync import TOKEN_RE as PAT      # noqa: E402

SKILL = r"C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\qwen35-ascend-migrator_整合版"
SKIP_DIRS = ("__pycache__", ".git", "out")
EXTS = (".md", ".py", ".json", ".yaml", ".yml", ".txt", ".sh", ".cfg", ".lock")


def main():
    hits, changed = [], []
    for root, dirs, files in os.walk(SKILL):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if not f.endswith(EXTS):
                continue
            p = os.path.join(root, f)
            try:
                src = io.open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            n = len(PAT.findall(src))
            if not n:
                continue
            hits.append((os.path.relpath(p, SKILL), n))
            out = PAT.sub("jt_<REDACTED_ID>:<REDACTED_SECRET>", src)
            if out != src:
                # ★ 2026-09-22 修（坑 201 根因）：备份**必须写在 Skill 树之外**。
                #   第一版写成 `p + ".bak_redact_<ts>"` ⇒ 备份落在 `docs/` 里 ⇒ **被打进交付 zip**，
                #   而它是"脱敏前"的副本 ⇒ **真 token 进了交付物**（且因无扩展名而躲过按扩展名过滤的扫描）。
                bdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_backup", "skill_baks")
                os.makedirs(bdir, exist_ok=True)
                bak = os.path.join(bdir, "%s.bak_redact_%s"
                                   % (os.path.basename(p), time.strftime("%Y%m%d_%H%M%S")))
                shutil.copy2(p, bak)
                io.open(p, "w", encoding="utf-8").write(out)
                back = io.open(p, encoding="utf-8").read()
                assert not PAT.search(back), "回读仍有未脱敏 token：%s" % p
                changed.append((os.path.relpath(p, SKILL), n))
                print("   备份（树外）: %s" % bak)
    for rel, n in hits:
        print("  命中 %-46s %d 处" % (rel, n))
    for rel, n in changed:
        print("REDACTED %-44s %d 处已脱敏" % (rel, n))
    left = sum(1 for rel, _n in hits if rel not in [c[0] for c in changed])
    print("REDACT_OK 命中文件=%d 已改=%d 未改=%d" % (len(hits), len(changed), left))
    return 0 if not left else 1


if __name__ == "__main__":
    sys.exit(main())
