#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""P2 候选原型（方案 1）：把 chunk_o.py 里**每次调用**的主机→设备常量搬运改为设备上直接构造。

三处改动（**数值完全不变**：同样的值、同样的 dtype，只是构造位置从主机改到设备）
--------------------------------------------------------------------------------
  A. torch.tril(torch.ones(BT, BT), diagonal=0).to(g.device)
     -> torch.tril(torch.ones(BT, BT, device=g.device), diagonal=0)          [2 处]
  B. torch.zeros(B, H, diff).to(g.device)
     -> torch.zeros(B, H, diff, device=g.device)                             [1 处]
  C. torch.arange(0, BT).to(g.device)
     -> torch.arange(0, BT, device=g.device)                                 [1 处]

★ **不引入任何全局副作用**：第一版草稿曾在模块顶层 monkey-patch `torch.ones/zeros/arange`
  做 env 门控 —— 那会污染整个进程且带未定义名，**已废弃**。现在改用最朴素的
  「备份 → 改 → 跑 → 还原」，基线臂用**还原后的原文件**跑，语义清晰、可逐字节核对。

★ 诚实边界：归因显示 `chunk_o.py:443` 的 `aten::copy_` **设备耗时为 0**（纯主机阻塞）
  ⇒ 本改动**设备侧增量为 0**，**不得**据此宣称加速；收益只在主机侧，须实测。
"""
import hashlib
import os
import shutil
import sys

TARGET = "/root/MindSpeed-MM/mindspeed_mm/fsdp/ops/gdn/triton/chunk_o.py"
BK = "/root/ops/fwbackup_p2"
MARK = "device=g.device"

EDITS = [
    ("torch.tril(torch.ones(BT, BT), diagonal=0).to(g.device)",
     "torch.tril(torch.ones(BT, BT, device=g.device), diagonal=0)", 2),
    ("torch.zeros(B, H, diff).to(g.device)",
     "torch.zeros(B, H, diff, device=g.device)", 1),
    ("torch.arange(0, BT).to(g.device)",
     "torch.arange(0, BT, device=g.device)", 1),
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "apply"
    if not os.path.isfile(TARGET):
        print("P2_PATCH_FAIL 找不到 %s" % TARGET)
        return 1

    if mode == "restore":
        bkp = os.path.join(BK, "chunk_o.py.orig")
        if not os.path.isfile(bkp):
            print("P2_RESTORE_FAIL 缺备份 %s" % bkp)
            return 1
        shutil.copy2(bkp, TARGET)
        rc = os.system("%s -m py_compile %s" % (sys.executable, TARGET))
        print("P2_RESTORE_OK sha256=%s py_compile_rc=%d" % (sha256(TARGET)[:16], rc))
        return 0 if rc == 0 else 1

    text = open(TARGET, encoding="utf-8").read()
    bkp = os.path.join(BK, "chunk_o.py.orig")
    # ★★ 2026-09-22 事故修复：**不能因为"备份已存在"就 SKIP**。
    #   本轮 P2 A/B 就是这样废掉的：备份在（数值复核时建的）⇒ apply 直接 SKIP 返回 0
    #   ⇒ 候选臂设置了个寂寞，跑的还是**原文件** ⇒ "候选 vs 基线"退化成"基线 vs 基线"。
    #   正确判据是"**目标文件当前是否已打过补丁**"，与备份在不在**无关**。
    if all(new in text for _old, new, _w in EDITS):
        print("P2_PATCH_SKIP 目标文件已处于打补丁状态（幂等）")
        return 0

    problems = []
    for old, _new, want in EDITS:
        got = text.count(old)
        if got != want:
            problems.append("锚点出现 %d 次（期望 %d）：%s" % (got, want, old[:70]))
    if problems:
        for p in problems:
            print("   P2_PATCH_FAIL %s" % p)
        print("P2_PATCH_FAIL 锚点校验未过 ⇒ 拒绝写入")
        return 1

    os.makedirs(BK, exist_ok=True)
    if not os.path.exists(bkp):
        shutil.copy2(TARGET, bkp)
        print("P2_PATCH backup=%s orig_sha256=%s" % (bkp, sha256(bkp)[:16]))
    else:
        # ★ 备份必须保持**原始未打补丁**的版本；若目标已被改过，绝不能拿它覆盖备份
        print("P2_PATCH 备份已存在，保留不动：%s（sha=%s）" % (bkp, sha256(bkp)[:16]))

    new_text = text
    for old, new, _want in EDITS:
        new_text = new_text.replace(old, new)
    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(new_text)
    rc = os.system("%s -m py_compile %s" % (sys.executable, TARGET))
    if rc != 0:
        print("P2_PATCH_FAIL py_compile 失败 ⇒ 立刻还原")
        shutil.copy2(bkp, TARGET)
        return 1
    print("P2_PATCH_OK new_sha256=%s py_compile=OK" % sha256(TARGET)[:16])
    print("   去重后剩余 \".to(g.device)\" 次数=%d（余下的不是常量构造，未动）"
          % new_text.count(".to(g.device)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
