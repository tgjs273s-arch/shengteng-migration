# -*- coding: utf-8 -*-
r"""_diff_inputs.py —— 本地取证：同配置两次运行 / 确定性配置两次运行的**输入侧**是否一致

对应复核的关键问法：「**找到第一处不同的输入或状态，而不是找到一段仍然相同的后缀**」。
本脚本只做一件事：把每份 train.log 里**启动阶段的输入 dump**（input_ids / label_ids / labels /
pixel / image_grid 等行）抽出来做哈希与逐行对比。

判读规则（写在报告里，避免事后解释）：
  · 同配置两次运行 **输入 dump 相同** 而 loss 第 3 步分叉 ⇒ 分歧**不在数据侧**（至少在打印精度内）
  · 输入 dump **不同** ⇒ 第一处不同在**前向之前**（sampler / worker / 预处理）
  · 确定性配置两次运行的输入 dump 必须相同（正对照）
"""
import hashlib
import io
import os
import re
import sys

LOG = r"C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\远端证据_20260922\logs"
PAT = re.compile(r"input_ids|label_ids|\blabels\b|pixel|image_grid|image_sizes|num_images", re.I)

FILES = [
    ("00A  (deter=false)", "AB_AVSB_20260921__00_A__train.log"),
    ("03A  (deter=false)", "AB_AVSB_20260921__03_A__train.log"),
    ("P2r1 (deter=true)", "phase1b_p2_20260921_171002__run1__train.log"),
    ("P2r2 (deter=true)", "phase1b_p2_20260921_171002__run2__train.log"),
]


def dump_lines(path):
    out = []
    for ln in io.open(path, encoding="utf-8", errors="replace"):
        if PAT.search(ln):
            out.append(ln.rstrip("\n"))
    return out


def main():
    data = {}
    for label, fn in FILES:
        p = os.path.join(LOG, fn)
        if not os.path.isfile(p):
            print("DIFF_INPUTS_MISSING %s" % p)
            return 2
        ls = dump_lines(p)
        h = hashlib.sha256("\n".join(ls).encode("utf-8")).hexdigest()[:16]
        data[label] = ls
        print("  %-20s 输入相关行=%-5d 大文件=%-6d sha256=%s" % (label, len(ls), os.path.getsize(p), h))

    pairs = [("00A  (deter=false)", "03A  (deter=false)", "同配置两次（deter=false）"),
             ("P2r1 (deter=true)", "P2r2 (deter=true)", "确定性配置两次（正对照）")]
    print("\n== 逐行对比 ==")
    for x, y, title in pairs:
        lx, ly = data[x], data[y]
        same = lx == ly
        print("  %-34s 行数 %d vs %d ⇒ %s" % (title, len(lx), len(ly), "**完全相同**" if same else "**不同**"))
        if not same:
            n = min(len(lx), len(ly))
            for i in range(n):
                if lx[i] != ly[i]:
                    print("      首个不同行 #%d：" % i)
                    print("        A: %s" % lx[i][:150])
                    print("        B: %s" % ly[i][:150])
                    break
            else:
                print("      前缀相同，仅长度不同（多出的行：%s）"
                      % (lx[n:] or ly[n:])[:120])
    print("\n判读提示（不得事后改写）：输入 dump 相同而 loss 第 3 步分叉 ⇒ 分歧不在数据侧（打印精度内）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
