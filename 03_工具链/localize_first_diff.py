#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""localize_first_diff.py —— 用**已有日志**定位"第一处不同"出现在前向还是反向

思路（不猜接口、不加 instrumentation，只用日志里已有的两个可观测量）：
  · 每步日志同时打印 `loss` 与 `grad norm`；
  · 若多份**同配置**运行的 **step1 loss 相同**而 **step1 grad_norm 不同** ⇒ 分歧**不来自前向输入**，
    而是先进**反向/梯度**；
  · 再逐指标找"第一次出现差异的步号"，并**分别**报告 loss 与 grad_norm ——
    这正是复核要求的口径区分（打印值不同 / 相对差超阈 / 张量比特不同，本工具只能给前两者，
    且明确受**打印精度**限制：loss 为 8 位有效、grad_norm 为 3 位小数）。

边界（写进输出，防止过度解读）：打印精度以下的差异**本方法看不见**；因此"step1 loss 相同"
不等于"step1 计算逐位一致"。
"""
import json
import os
import sys

EV = r"C:\Users\HUAWEI\Desktop\转交给codex的内容\04_核心资产\远端证据_20260922\logs"
SETS = [
    ("deter=false（出厂配置，3 次运行）", os.path.join(EV, "ab6.json"),
     {"00A": "A", "03A": "A", "04A": "A"}),
    ("deter=true（诊断路径，2 次运行）", os.path.join(EV, "p2mix.json"),
     {"P2r1": "A", "P2r2": "A"}),
]


def load(path, want):
    runs = json.load(open(path, encoding="utf-8"))
    out = {}
    for r in runs:
        if r.get("tag") in want:
            out[r["tag"]] = r
    missing = set(want) - set(out)
    if missing:
        print("  LOCALIZE_WARN 缺运行：%s" % sorted(missing))
    return out


def first_diff_steps(series_map, field, max_step=10):
    """返回 {相对第 1 份运行，首次出现差异的步号}；比较基准是**打印值**。"""
    tags = sorted(series_map)
    base = series_map[tags[0]].get(field) or []
    res = {}
    for t in tags[1:]:
        s = series_map[t].get(field) or []
        fd = None
        for i in range(min(len(base), len(s), max_step)):
            if base[i] != s[i]:
                fd = i + 1
                break
        res[t] = fd
    return tags[0], res


def main():
    for title, path, want in SETS:
        if not os.path.isfile(path):
            print("LOCALIZE_MISSING %s" % path)
            continue
        print("\n== %s ==" % title)
        m = load(path, want)
        for field in ("loss", "grad_norm"):
            print("  -- %s（前 3 步打印值）--" % field)
            for t in sorted(m):
                v = (m[t].get(field) or [])[:3]
                print("     %-6s %s" % (t, v))
            base, fd = first_diff_steps(m, field, max_step=10)
            print("     以 %s 为基准，首次出现**打印值不同**的步号：%s" % (base, fd))

    # 跨 deter 设置的关键对照：两边各自的 step1
    print("\n== 关键对照：step1 的 loss 与 grad_norm ==")
    try:
        a = load(os.path.join(EV, "ab6.json"), {"00A", "03A", "04A"})
        p = load(os.path.join(EV, "p2mix.json"), {"P2r1", "P2r2"})
    except Exception as exc:
        print("LOCALIZE_FAIL %s" % exc)
        return 2
    print("  deter=false：loss1=%s  grad1=%s"
          % ([ (a[t].get('loss') or [None])[0] for t in sorted(a) ],
             [ (a[t].get('grad_norm') or [None])[0] for t in sorted(a) ]))
    print("  deter=true ：loss1=%s  grad1=%s"
          % ([ (p[t].get('loss') or [None])[0] for t in sorted(p) ],
             [ (p[t].get('grad_norm') or [None])[0] for t in sorted(p) ]))
    print("\n判读（写在输出里，防止事后改写）：")
    print("  · 若同一 deter 设置内 loss1 **相同**而 grad1 **不同** ⇒ 分歧先进**反向/梯度**，不是前向输入；")
    print("  · 若 loss1 跨 deter 设置就不同 ⇒ 该开关改变的是**前向计算本身**（与前面已记录的 0.1154926 vs 0.1169518 一致）；")
    print("  · 打印精度：loss 8 位有效数字、grad_norm 3 位小数 ⇒ **精度以下的差异本方法看不见**。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
