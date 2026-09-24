#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""gate_selftest.py —— 闸门自身的坏例（Codex 复核 P1-b / P1-c 的判据）

为什么必须有：Codex 用**纯内存坏例**证明了两条闸门会误放行：
  · pyflakes 无法启动（`No module named pyflakes`）时，G23 仍报 `G23_OK undefined=0`；
  · 子进程 rc=1、先打印 `A_CASE_OK` 后打印 `VERIFY_FAIL case_ok=12/13` 时，G5 仍判 PASS。
本脚本把这些坏例**固化成可重复运行的测试**：注入上面两种情形，断言闸门必须报 **ERROR/FAIL**。
（"闸门在工具失效时不能报通过"——这正是 fail-closed 的定义。）

用法：python gate_selftest.py    输出 GATE_SELFTEST_OK cases=N failed=0
"""
import importlib.util
import os
import subprocess
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))


def load_prepush():
    spec = importlib.util.spec_from_file_location("pp_sbx", os.path.join(HERE, "prepush_check.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class FakeCompleted:
    def __init__(self, rc, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def main():
    fails = []
    m = load_prepush()

    def ck(name, cond, extra=""):
        print("  [%s] %-58s %s" % ("PASS" if cond else "FAIL", name, extra))
        if not cond:
            fails.append(name)

    real_run = m._run

    # ---- 坏例 1：pyflakes 无法启动 ⇒ G23 必须 ERROR（不得 PASS） ----
    m._run = lambda *a, **k: FakeCompleted(1, "", "No module named pyflakes\n")
    passed, got = m.gate_pyflakes(m.SKILL, 60)
    ck("pyflakes 无法启动 ⇒ G23 必须不通过", passed is False and "ERROR" in got, "got=%s" % got)

    # ---- 坏例 2：扫描到 0 个文件 ⇒ G23 必须 ERROR（零样本 ≠ 零问题） ----
    real_walk = os.walk
    m.os.walk = lambda *a, **k: iter([])
    m._run = lambda *a, **k: FakeCompleted(0, "", "")
    passed, got = m.gate_pyflakes(m.SKILL, 60)
    m.os.walk = real_walk
    ck("零样本（没扫到任何 .py）⇒ G23 必须不通过", passed is False and "ERROR" in got, "got=%s" % got)

    # ---- 坏例 3：rc=1 但输出里出现期望标记 ⇒ run_gate 必须 FAIL（局部过 ≠ 整体过） ----
    m._run = lambda *a, **k: FakeCompleted(1, "A_CASE_OK\nVERIFY_FAIL case_ok=12/13\n", "")
    ok = m.run_gate("G5", "模拟：局部 OK + 整体 FAIL + rc=1",
                    [sys.executable, "-c", "pass"], HERE, "FINGERPRINT_OK", timeout=30)
    ck("rc=1 且出现标记 ⇒ 必须 FAIL（不得被 _OK 子串救回）", ok is False, "run_gate=%s" % ok)

    # ---- 坏例 4：工具不存在 ⇒ 必须 ERROR（后端缺失） ----
    ok = m.run_gate("GX", "模拟：后端文件不存在",
                    [sys.executable, os.path.join(HERE, "_no_such_tool.py")], HERE, "ANY_OK", timeout=30)
    ck("后端缺失 ⇒ 必须 ERROR", ok is False, "run_gate=%s" % ok)

    # ---- 正对照：正常工具 + 正确标记 ⇒ PASS ----
    m._run = real_run          # ★ 必须先还原真实实现：否则正对照仍在桩下运行（测试自伤）
    ok = m.run_gate("GY", "正对照：python -c 打印标记",
                    [sys.executable, "-c", "print('PROBE_MARKER_OK')"], HERE, "PROBE_MARKER_OK", timeout=30)
    ck("正对照：rc=0 且标记命中 ⇒ PASS", ok is True, "run_gate=%s" % ok)

    m._run = real_run
    print("GATE_SELFTEST_%s cases=5 failed=%d" % ("OK" if not fails else "FAIL", len(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
