#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evidence_manifest.py —— 证据逐文件哈希表（闸门 G12 的实现；只读+可写清单）

为什么需要：坑 180 把本地证据副本删过一次，事后靠"远端重拉 + 逐个哈希对照"才证明拉回的是同一份。
若当初有一张清单，**当时就能一眼看出缺了哪些文件**。本脚本做两件事：
  --write   扫描证据目录，生成 `_manifest.json`（相对路径 → sha256 + 大小）
  --verify  按清单逐文件核对：**缺失/多出/哈希不符**任一命中即 FAIL（fail-closed）
清单本身不存在时，`--verify` 判 FAIL —— 因为"没有清单"等于"无从证明证据完整"。

用法：
  python evidence_manifest.py --write --dir <证据目录>
  python evidence_manifest.py --verify --dir <证据目录>
输出：EVIDENCE_MANIFEST_OK items=N / EVIDENCE_MANIFEST_FAIL …
"""
import argparse
import hashlib
import json
import os
import sys

from _project_paths import bootstrap_paths
PATHS = bootstrap_paths(parse_cli=__name__ == "__main__")

DEFAULT_DIR = PATHS.evidence
MANIFEST = "_manifest.json"


def sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def scan(d):
    out = {}
    for root, _dirs, files in os.walk(d):
        for f in files:
            if f == MANIFEST:
                continue
            p = os.path.join(root, f)
            rel = os.path.relpath(p, d).replace("\\", "/")
            out[rel] = {"sha256": sha(p), "size": os.path.getsize(p)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if not os.path.isdir(a.dir):
        print("EVIDENCE_MANIFEST_FAIL 目录不存在：%s" % a.dir)
        return 2
    mp = os.path.join(a.dir, MANIFEST)

    if a.write:
        cur = scan(a.dir)
        if not cur:
            print("EVIDENCE_MANIFEST_FAIL 目录里没有文件：%s" % a.dir)
            return 1
        with open(mp, "w", encoding="utf-8") as fh:
            json.dump(cur, fh, ensure_ascii=False, indent=1, sort_keys=True)
        print("EVIDENCE_MANIFEST_WRITTEN %s items=%d" % (mp, len(cur)))
        print("EVIDENCE_MANIFEST_OK items=%d" % len(cur))
        return 0

    # 默认行为 = verify（宁可要求显式清单，也不假装"没清单也算过"）
    if not os.path.isfile(mp):
        print("EVIDENCE_MANIFEST_FAIL 缺清单 %s ⇒ 无从证明证据完整（先跑 --write）" % mp)
        return 1
    want = json.load(open(mp, encoding="utf-8"))
    cur = scan(a.dir)
    missing = sorted(set(want) - set(cur))
    extra = sorted(set(cur) - set(want))
    changed = sorted(k for k in (set(want) & set(cur)) if want[k]["sha256"] != cur[k]["sha256"])
    print("清单=%d 现有=%d 缺失=%d 多出=%d 哈希不符=%d"
          % (len(want), len(cur), len(missing), len(extra), len(changed)))
    for k in (missing + changed)[:8]:
        print("   ! %s" % k)
    if missing or changed:
        print("EVIDENCE_MANIFEST_FAIL 证据与清单不一致 ⇒ **不要声称证据完整**")
        return 1
    if extra:
        print("   注：多出 %d 个未入清单的文件（不判失败，但清单应更新）：%s" % (len(extra), extra[:5]))
    print("EVIDENCE_MANIFEST_OK items=%d" % len(cur))
    return 0


if __name__ == "__main__":
    sys.exit(main())
