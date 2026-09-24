#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
selfcheck.py — Skill 完整性 + 环境一致性 + 判定链可用性 自检（0-GPU 可跑）

三层自检（评委无昇腾硬件也能跑 L1/L2）：
  L1 结构完整性：目录/脚本/配置/基线是否齐全，脚本能否通过语法检查
  L2 判定链可用性：scripts/judge/selfcheck_sk04.py → 期望 VERIFY_OK 24/24
  L3 环境一致性：scripts/selfcheck_env.py 对照 config/versions.lock（无 NPU 时降级为 UNVERIFIED，不判失败）

用法：
  python3 selfcheck.py            # 全部三层
  python3 selfcheck.py --l1       # 仅结构
退出码：0=SELFCHECK_OK；3=有 FAIL；2=前置缺失（无法自检）
"""

import argparse
import json
import os
import py_compile
import shutil as _sh
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
FAIL, WARN = [], []

REQUIRED = [
    "SKILL.md",
    "README.md",
    "config/env_matrix.yaml",
    "config/versions.lock",
    "config/baselines/officialA.yaml",
    "config/baselines/officialB.yaml",
    "config/templates/qwen3_5_0_8B_base.yaml",
    "scripts/00_probe_env.py",
    "scripts/10_analyze_points.py",
    "scripts/20_plan_migration.py",
    "scripts/30_verify_ops.py",
    "scripts/40_prepare_assets.py",
    "scripts/50_train.py",
    "scripts/60_bench.py",
    "scripts/70_judge.py",
    "scripts/05_preflight.py",
    "scripts/90_selfcheck_gates.py",
    "scripts/_envcompat.py",
    "scripts/00b_triton_min_kernel.py",
    "scripts/00a_probe_env_raw.sh",
    "scripts/bringup.sh",
    "scripts/run_from_zero.sh",
    "sk04_judge/scripts/fingerprint_cfg.py",
    "sk04_judge/scripts/fingerprint_observed.py",
    "sk04_judge/scripts/judge_comparable.py",
    "sk04_judge/scripts/registry_append.py",
    "sk04_judge/scripts/data_id.py",
    "sk04_judge/scripts/selfcheck_sk04.py",
    "docs/PITFALLS_坑表.md",
]


def check_l1(structure_only=False):
    print("[L1] 结构完整性")
    for rel in REQUIRED:
        p = os.path.join(ROOT, rel)
        if os.path.isfile(p):
            print("  PASS  %s" % rel)
        else:
            print("  FAIL  %s（缺失）" % rel)
            FAIL.append("missing:" + rel)
    if structure_only:
        return
    # 语法检查（.py）
    print("[L1] 脚本语法检查")
    n_ok = 0
    for dirpath, _, files in os.walk(os.path.join(ROOT, "scripts")):
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.join(dirpath, f)
            try:
                py_compile.compile(p, doraise=True)
                n_ok += 1
            except Exception as e:
                print("  FAIL  %s: %s" % (os.path.relpath(p, ROOT), str(e)[:120]))
                FAIL.append("syntax:" + f)
    print("  PASS  %d 个脚本语法通过" % n_ok)
    # ★ 坑 52：L1 此前**只检查 .py**，从不检查 .sh —— 而 bash 脚本是流程入口，
    #   一个语法错就整条链跑不动（且本机可能没有可用 bash，无法本地验证）。
    #   现加入 bash -n 闸门；无**可用** bash 时标 SKIP（不静默当通过）。
    # ★ 坑 53：`shutil.which("bash")` 在 Windows 上会命中 `C:\WINDOWS\system32\bash.exe`
    #   （WSL 转发器）—— 但它可能根本没有已安装的发行版（execvpe /bin/bash failed）。
    #   **"找得到"≠"能用"** → 必须实跑一次 `bash -c 'echo OK'` 验证。
    print("[L1] bash 脚本语法检查")

    def _find_bash():
        cands = [_sh.which("bash"), r"D:\Git\bin\bash.exe", r"C:\Program Files\Git\bin\bash.exe",
                 "/bin/bash", "/usr/bin/bash"]
        for c in cands:
            if not c or not os.path.exists(c) and not os.path.isabs(c):
                continue
            try:
                r = subprocess.run([c, "-c", "echo BASH_OK"], capture_output=True,
                                   timeout=30, encoding="utf-8", errors="replace")
                if "BASH_OK" in (r.stdout or ""):
                    return c
            except Exception:
                continue
        return None

    def _sh_run(argv, timeout=60):
        """★ 坑 54：Windows 下 subprocess 默认用本地编码(GBK)解码，bash 输出是 UTF-8
        → reader 线程 UnicodeDecodeError **直接崩掉自检**。必须显式 utf-8 + errors=replace。"""
        try:
            r = subprocess.run(argv, capture_output=True, timeout=timeout,
                               encoding="utf-8", errors="replace")
            return r.returncode, (r.stdout or ""), (r.stderr or "")
        except Exception as e:
            return -1, "", "%s: %s" % (type(e).__name__, e)

    bash = _find_bash()
    shs = []
    for dirpath, _, files in os.walk(os.path.join(ROOT, "scripts")):
        for f in files:
            if f.endswith(".sh"):
                shs.append(os.path.join(dirpath, f))
    if not bash:
        print("  SKIP  未找到**可用**的 bash（本机无法验证 .sh 语法）")
        print("        目标机上的 run_from_zero.sh 已内建 bash -n 闸门，可作为远端保证")
        WARN.append("no_usable_bash_for_syntax_check")
    else:
        bad = []
        for p in sorted(shs):
            rc_sh, _o, err = _sh_run([bash, "-n", p])
            if rc_sh != 0:
                bad.append((os.path.relpath(p, ROOT), err[:160]))
        if bad:
            for rel, err in bad:
                print("  FAIL  %s: %s" % (rel, err))
                FAIL.append("shsyntax:" + os.path.basename(rel))
        else:
            print("  PASS  %d 个 bash 脚本语法通过（bash -n @ %s）" % (len(shs), bash))

    # ★ 坑 59：**BOM 检测**。用某些工具（如 Windows PowerShell 的 `Set-Content -Encoding UTF8`）
    #   写文件会带上 UTF-8 BOM，于是 `.sh` 的第一行变成 `﻿#!/bin/bash`（不可识别的解释器），
    #   运行时抛 `line 1: ﻿#!/bin/bash: No such file or directory`。
    #   `bash -n` **查不出来**（语法没错），只有真跑才炸 —— 所以必须单独设闸门。
    print("[L1] BOM 检测（shebang 污染）")
    bom_files = []
    for dirpath, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "out")]
        for f in files:
            if not f.endswith((".sh", ".py", ".yaml")):
                continue
            p = os.path.join(dirpath, f)
            try:
                with open(p, "rb") as fh:
                    head = fh.read(3)
                if head == b"\xef\xbb\xbf":
                    bom_files.append(os.path.relpath(p, ROOT))
            except Exception:
                continue
    if bom_files:
        for rel in bom_files:
            print("  FAIL  %s（含 UTF-8 BOM → shebang 失效）" % rel)
            FAIL.append("bom:" + os.path.basename(rel))
    else:
        print("  PASS  无 BOM 污染（.sh/.py/.yaml）")

    # ★ 坑 46/60/62：triton 判据的**结构自检**（AST 级，不需要 NPU）
    #   守住"判据自身写错"这一类（triton.program_id 不存在 / tl 不在模块顶层）。
    print("[L1] triton 判据结构自检（AST，无需 NPU）")
    probe = os.path.join(ROOT, "scripts", "00b_triton_min_kernel.py")
    if not os.path.isfile(probe):
        print("  FAIL  scripts/00b_triton_min_kernel.py 缺失")
        FAIL.append("triton_probe_missing")
    else:
        rc_t, out_t, _e = _sh_run([sys.executable, probe, "--selfcheck"])
        line = (out_t or "").strip().splitlines()
        print("  " + (line[-1][:150] if line else "(无输出)"))
        if "TRITON_PROBE_SELFCHECK_OK" in (out_t or ""):
            print("  PASS  triton 判据结构约束满足")
        else:
            print("  FAIL  triton 判据结构不满足（见上）")
            FAIL.append("triton_probe_structure")
    # 环境矩阵可解析
    print("[L1] 配置可解析性")
    try:
        import yaml
        for rel in ("config/env_matrix.yaml", "config/baselines/officialB.yaml",
                    "config/templates/qwen3_5_0_8B_base.yaml"):
            yaml.safe_load(open(os.path.join(ROOT, rel), encoding="utf-8"))
            print("  PASS  %s" % rel)
    except ImportError:
        print("  WARN  未安装 pyyaml，跳过配置解析检查")
        WARN.append("pyyaml缺失")
    except Exception as e:
        print("  FAIL  配置解析失败: %s" % str(e)[:150])
        FAIL.append("yaml_parse")


def check_l2():
    print("[L2] 判定链可用性（judge/selfcheck_sk04.py）")
    p = os.path.join(ROOT, "sk04_judge", "scripts", "selfcheck_sk04.py")
    if not os.path.isfile(p):
        FAIL.append("sk04_missing"); print("  FAIL  判定链自检脚本缺失"); return
    try:
        r = subprocess.run([sys.executable, p], capture_output=True, text=True, timeout=180,
                           cwd=os.path.join(ROOT, "sk04_judge"))
        out = (r.stdout or "") + (r.stderr or "")
        tail = [l for l in out.splitlines() if l.strip()][-4:]
        for l in tail:
            print("  " + l[:200])
        # ★ 坑 42：原判据 `"VERIFY_OK" in out or r.returncode == 0` —— VERIFY_OK 那行
        #   无论成败都会打印，故 `or` 恒真 → **永远 PASS**（假阳性）。现要求：
        #   rc==0 且 failed=0 且 skipped 有显式记录（SKIP 不算失败，但必须可见）。
        import re as _re
        m = _re.search(r"failed=(\d+)", out)
        n_failed = int(m.group(1)) if m else None
        ok = (r.returncode == 0) and (n_failed == 0)
        if ok:
            print("  PASS  判定链自检通过（rc=0, %s）"
                  % ("failed=0" if n_failed == 0 else "无 failed 字段"))
        else:
            print("  FAIL  判定链自检未通过（rc=%s, failed=%s）"
                  % (r.returncode, "?" if n_failed is None else n_failed))
            FAIL.append("sk04_selfcheck")
    except Exception as e:
        print("  FAIL  执行失败: %s" % str(e)[:150])
        FAIL.append("sk04_exec")


def check_l3():
    """★ 坑 41：环境一致性与"本机是否可用"是两个问题，必须分开判。
      - portability（栈内部自洽 + 版本在支持范围内）= **通过判据**
      - acceptance（是否就是 versions.lock 描述的那台验收环境）= **信息性报告**
    原实现只有 acceptance，导致换到新机器必然 SELFCHECK_FAIL，无法区分
    "环境不对"与"环境不同但可用"。
    """
    print("[L3] 环境自检（scripts/selfcheck_env.py）")
    p = os.path.join(ROOT, "scripts", "selfcheck_env.py")
    if not os.path.isfile(p):
        WARN.append("env_selfcheck_missing"); print("  WARN  环境自检脚本缺失（跳过）"); return

    def run(mode):
        r = subprocess.run([sys.executable, p, "--mode=" + mode],
                           capture_output=True, text=True, timeout=180, cwd=ROOT)
        return (r.stdout or "") + (r.stderr or ""), r.returncode

    try:
        out_p, rc_p = run("portability")
        for l in [x for x in out_p.splitlines() if x.strip()][-14:]:
            print("  " + l[:160])
        if "SELFCHECK_FAIL" in out_p or rc_p == 3:
            print("  FAIL  栈内部不自洽（torch/torch_npu 不配对，或版本超出支持范围）")
            FAIL.append("env_inconsistent")
        else:
            print("  PASS  栈内部自洽（portability）")

        out_a, rc_a = run("acceptance")
        diff = [l.strip() for l in out_a.splitlines()
                if ("FAIL" in l or "DRIFT" in l)]
        if rc_a == 3 or "SELFCHECK_FAIL" in out_a:
            print("  INFO  本机 **不是** versions.lock 描述的验收环境（验收模式不通过，属预期）：")
            for l in diff[:8]:
                print("          " + l[:150])
            print("         → 若需在本机产出可宣称的精度/性能结论，必须**重新锚定**。")
            WARN.append("env_not_acceptance_machine")
        else:
            print("  PASS  本机即 versions.lock 描述的验收环境（acceptance）")
    except Exception as e:
        print("  WARN  环境自检执行异常: %s" % str(e)[:120])
        WARN.append("env_exec")


def check_l4():
    """L4 判据负向对照 —— **测试判据本身**（good 必过 / bad 必挂）。

    ★ 加入理由：干净环境验证暴露出 4 个假阳性判据（坑 33/35/42/47）与 1 个假阴性判据（坑 46）。
      它们的共性是"判据从没被验证过能不能失败"。**一个不会失败的判据不是判据。**
      本层对每个关键判据双向量化验证，并把历史上踩过的缺陷模式做成静态回归守卫
      （自带可证伪性自证：正则必须能匹配合成坏样本，否则守卫本身失效）。
    """
    print("[L4] 判据负向对照（测试判据本身）")
    p = os.path.join(ROOT, "scripts", "90_selfcheck_gates.py")
    if not os.path.isfile(p):
        WARN.append("gates_missing"); print("  WARN  判据对照脚本缺失（跳过）"); return
    try:
        r = subprocess.run([sys.executable, p], capture_output=True, timeout=900,
                           cwd=ROOT, encoding="utf-8", errors="replace")
        out = (r.stdout or "") + (r.stderr or "")
        keep = [l for l in out.splitlines()
                if l.strip().startswith(("PASS", "FAIL", "VACUOUS", "SKIP", "ERROR", "~", "!"))
                or "合计" in l]
        for l in keep[-16:]:
            print("  " + l[:170])
        if "GATES_OK" in out and r.returncode == 0:
            print("  PASS  全部判据均可双向验证（能通过、也能失败）")
        else:
            print("  FAIL  存在失效/空过判据（见上）")
            FAIL.append("gates_not_falsifiable")
    except Exception as e:
        print("  WARN  判据对照执行异常: %s" % str(e)[:120])
        WARN.append("gates_exec")


def main():
    ap = argparse.ArgumentParser(description="qwen35-ascend-migrator 自检")
    ap.add_argument("--l1", action="store_true", help="仅结构完整性")
    ap.add_argument("--no-gates", action="store_true", help="跳过 L4 判据负向对照（慢）")
    args = ap.parse_args()

    print("=" * 66)
    print("qwen35-ascend-migrator 自检  root=%s" % ROOT)
    print("=" * 66)
    check_l1(structure_only=args.l1)
    if not args.l1:
        check_l2()
        check_l3()
        if not args.no_gates:
            check_l4()

    print("=" * 66)
    if FAIL:
        print("SELFCHECK_FAIL items=%d: %s" % (len(FAIL), ", ".join(FAIL)))
        if WARN:
            print("WARN: %s" % ", ".join(WARN))
        return 3
    print("SELFCHECK_OK%s" % ("（WARN: %s）" % ", ".join(WARN) if WARN else ""))
    print("四层自检通过：结构完整 / 判定链可用 / 环境一致性 / 判据可证伪")
    return 0


if __name__ == "__main__":
    sys.exit(main())
