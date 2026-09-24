#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
selfcheck_env.py — 环境一致性自检（0-GPU，评委复现入场券）

★ v2 修复（坑 41）：原实现把"换了机器"与"装错版本"一律判 FAIL → 在新机器上必然
  SELFCHECK_FAIL，信号失真（无法区分"环境不对"与"环境不同但可用"）。现区分两种用途：

  --mode=acceptance （默认）  验收复现：本机必须是 versions.lock 描述的那台环境，逐项严格相等
  --mode=portability          新机可移植性：报告与 lock 的 drift，但**只对"内部不自洽"判失败**
                              （如 torch 与 torch_npu 版本不配对、Python 版本不受支持）

用法:
  python3 selfcheck_env.py                          # 验收模式
  python3 selfcheck_env.py --mode=portability       # 新机器上用这个
  python3 selfcheck_env.py --strict                 # UNVERIFIED 也计失败

退出码: 0 通过 | 2 strict 下有 UNVERIFIED | 3 失败
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# ---------------- drift 分类 ----------------
# HARD: 直接影响数值/可复现性的项（换了就必须重新锚定精度）
HARD_KEYS = {"python", "torch", "torch_npu", "transformers", "triton", "cann",
             "hardware.chip_count", "mindspeed"}
# SOFT: 补丁级/工具级，通常不改变数值语义
SOFT_KEYS = {"hardware.driver", "pip"}

# 本 Skill 声明支持的范围（超出则该组合未验证，走降级路径）
SUPPORTED_PY = {"3.9", "3.10", "3.11", "3.12"}
SUPPORTED_CANN_MAJOR = {"9.0", "9.1"}


def _find_lock():
    """向上查找 config/versions.lock（兼容 scripts/ 或项目根目录布局）。"""
    here = Path(__file__).resolve().parent
    for base in (here, here.parent, here.parent.parent):
        cand = base / "config" / "versions.lock"
        if cand.is_file():
            return cand
    return here / "config" / "versions.lock"


LOCK = _find_lock()
FAIL_KEYS = []      # 硬失败
DRIFT_KEYS = []     # 与 lock 不同（可移植模式下不判失败）
UNVERIFIED = []     # 本机不可测
ACTUAL = {}         # 记录实测值供一致性校验


def sh(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return (r.stdout + r.stderr).strip()
    except Exception:
        return ""


def parse_lock(path):
    """versions.lock: name | value | source | note（# 注释行跳过）。返回 {name: value}。"""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        out[parts[0]] = parts[1] if len(parts) > 1 else ""
    return out


def _strip_local(v):
    """★ 坑 50：剥离 PEP440 local version 后缀（`2.7.1+cpu` → `2.7.1`）。
    aarch64 上 pypi 的 torch 就是 `+cpu` 构建，torch_npu 负责接 NPU —— 这是**正常预期**，
    不应被判为 drift。"""
    return str(v).split("+")[0] if v else v


def check(name, lock_val, actual, mode, ok_if=None):
    """actual: str | None。lock_val 取首 token（容忍括号注释）。"""
    if actual is None:
        UNVERIFIED.append(name)
        print("  %-22s UNVERIFIED (本机不可测)   lock=%s" % (name, lock_val))
        return
    ACTUAL[name] = actual
    lock_head = _strip_local(lock_val.split()[0] if lock_val else "")
    actual_cmp = _strip_local(actual)
    ok = actual_cmp == lock_head
    if ok_if and ok_if(actual):
        ok = True
    if ok:
        note = "" if actual_cmp == actual else "  (+local 后缀已忽略)"
        print("  %-22s %-5s lock=%-28s actual=%s%s" % (name, "PASS", lock_head, actual[:60], note))
        return
    kind = "HARD" if name in HARD_KEYS else ("SOFT" if name in SOFT_KEYS else "HARD")
    if mode == "acceptance":
        FAIL_KEYS.append(name)
        print("  %-22s %-5s lock=%-28s actual=%s  [%s]" % (name, "FAIL", lock_head, actual[:60], kind))
    else:
        DRIFT_KEYS.append((name, kind, lock_head, actual))
        print("  %-22s %-5s lock=%-28s actual=%s  [DRIFT/%s]" % (name, "DRIFT", lock_head, actual[:60], kind))


def cross_consistency():
    """★ 真正决定"这台新机器能不能用"的判据：栈内部是否自洽 + 是否在本 Skill 支持范围内。"""
    print("[consistency]  栈内部自洽性与支持范围（不依赖 lock）")
    problems = []

    py = ACTUAL.get("python", "")
    if py:
        mm = ".".join(py.split(".")[:2])
        if mm in SUPPORTED_PY:
            print("  %-34s PASS  python %s 在支持范围 %s" % ("python_supported", py, sorted(SUPPORTED_PY)))
        else:
            problems.append("python %s 不在支持范围 %s" % (py, sorted(SUPPORTED_PY)))
            print("  %-34s FAIL  python %s 未验证" % ("python_supported", py))

    t, tn = ACTUAL.get("torch", ""), ACTUAL.get("torch_npu", "")
    if t and tn:
        tm = ".".join(t.split(".")[:2])
        if tn.startswith(tm):
            print("  %-34s PASS  torch=%s torch_npu=%s 主次版本一致" % ("torch_pairing", t, tn))
        else:
            problems.append("torch=%s 与 torch_npu=%s 主次版本不一致" % (t, tn))
            print("  %-34s FAIL  torch=%s 与 torch_npu=%s 不配对" % ("torch_pairing", t, tn))
    else:
        print("  %-34s SKIP  torch/torch_npu 未装（先跑 bringup.sh）" % "torch_pairing")

    cann = ACTUAL.get("cann", "")
    if cann:
        cm = ".".join(cann.split("-")[0].split(".")[:2])
        if cm in SUPPORTED_CANN_MAJOR:
            print("  %-34s PASS  CANN %s 在支持范围 %s" % ("cann_supported", cann, sorted(SUPPORTED_CANN_MAJOR)))
        else:
            problems.append("CANN %s 未验证" % cann)
            print("  %-34s FAIL  CANN %s 未验证" % ("cann_supported", cann))
        if "-beta" in cann or "-rc" in cann.lower():
            print("  %-34s NOTE  预发布版 CANN（%s）—— torch_npu 配套可能有别于 GA，需实测" % ("cann_prerelease", cann))
    return problems


def main():
    ap = argparse.ArgumentParser(description="环境一致性自检（对照 versions.lock）")
    ap.add_argument("--lock", default=str(LOCK))
    ap.add_argument("--strict", action="store_true", help="UNVERIFIED 也计失败")
    ap.add_argument("--mode", choices=["acceptance", "portability"], default="acceptance",
                    help="acceptance=必须是 lock 描述的那台环境; portability=新机器上报告 drift")
    args = ap.parse_args()
    mode = args.mode

    lock = parse_lock(Path(args.lock))
    print("== ENV SELFCHECK (versions.lock: %d 项, mode=%s) ==" % (len(lock), mode))

    # ---- 硬件 ----
    print("[hardware]")
    npu_l = sh("npu-smi info -l 2>/dev/null | head -8")
    m = re.search(r"Chip Count\s*:\s*(\d+)", npu_l)
    check("hardware.chip_count", lock.get("hardware.chip_count"), m.group(1) if m else None, mode)
    ver = sh("cat /usr/local/Ascend/driver/version.info 2>/dev/null | head -1")
    m = re.search(r"Version=([\d.]+)", ver)
    check("hardware.driver", lock.get("hardware.driver"), m.group(1) if m else None, mode)

    # ---- Python ----
    print("[python]")
    py = sh("python3 --version")
    m = re.search(r"Python\s+([\d.]+)", py)
    check("python", lock.get("python"), m.group(1) if m else None, mode)
    pip = sh("python3 -m pip --version 2>/dev/null | head -1")
    m = re.search(r"pip\s+([\d.]+)", pip)
    if m is None:
        # ★ 坑 38：python3 可能没有 pip（pip3 属于另一个解释器）→ 给出可操作的提示
        alt = sh("command -v pip3 2>/dev/null")
        print("  %-22s UNVERIFIED (python3 无 pip)" % "pip"
              + (" — pip3 在 %s，注意两者可能是**不同解释器**（坑38）" % alt if alt else ""))
        UNVERIFIED.append("pip")
    else:
        check("pip", lock.get("pip"), m.group(1), mode)

    # ---- 框架（导入不可得 → UNVERIFIED）----
    print("[framework]")
    for mod, lockkey in [("torch", "torch"), ("torch_npu", "torch_npu"),
                         ("transformers", "transformers"), ("triton", "triton"),
                         ("mindspeed", "mindspeed")]:
        try:
            modobj = __import__(mod)
            v = getattr(modobj, "__version__", "?")
            check(lockkey, lock.get(lockkey), str(v), mode)
        except ImportError:
            print("  %-22s UNVERIFIED (import %s 失败 — 需 CANN env? 见 versions.lock env.ld_library_path)" % (lockkey, mod))
            UNVERIFIED.append(lockkey)

    print("[triton backend]")
    tb = sh("python3 -c 'import triton; from triton.backends import ascend; print(\"ascend-ok\")' 2>&1 | tail -1")
    check("triton backend ascend", "ascend-ok", tb if "ascend-ok" in tb else None, mode)

    print("[cann]")
    # ★ 坑 40：CANN 目录名可能是 cann-9.1.0-beta.3（含后缀），原正则 [\d.]+ 会截掉 -beta.3
    cann_dir = sh("ls -d /usr/local/Ascend/cann* /usr/local/Ascend/ascend-toolkit/latest 2>/dev/null | head -3")
    m = re.search(r"cann-(\S+)", cann_dir)
    if not m:
        m = re.search(r"cann(?:-|/)(\S+)", sh("readlink -f /usr/local/Ascend/ascend-toolkit/latest 2>/dev/null"))
    if m:
        check("cann", lock.get("cann"), m.group(1).strip("/"), mode)
    else:
        UNVERIFIED.append("cann")
        print("  cann UNVERIFIED（未找到 cann-* 目录或版本文件）")

    # ---- 内部自洽性 ----
    problems = cross_consistency()

    # ---- 汇总 ----
    print("=" * 60)
    if mode == "portability":
        if DRIFT_KEYS:
            print("SELFCHECK_DRIFT 与 versions.lock 不同 %d 项（非失败，仅说明本机非验收环境）：" % len(DRIFT_KEYS))
            for name, kind, lv, av in DRIFT_KEYS:
                print("    - %-22s [%s] lock=%s  actual=%s" % (name, kind, lv, av[:40]))
        if problems:
            print("SELFCHECK_FAIL 栈内部不自洽 %d 项：%s" % (len(problems), "; ".join(problems)))
            return 3
        if args.strict and UNVERIFIED:
            print("SELFCHECK_WARN strict 模式: UNVERIFIED=%s" % ",".join(UNVERIFIED))
            return 2
        print("SELFCHECK_PORTABILITY_OK  栈内部自洽，未在 lock 描述的那台机器上（属预期）；"
              "UNVERIFIED=%d" % len(UNVERIFIED))
        print("  注意：本模式下**不得**用它宣称「与验收环境一致」；精度/性能结论需在本机重新锚定。")
        return 0

    # acceptance 模式
    if FAIL_KEYS:
        print("SELFCHECK_FAIL items=%d: %s" % (len(FAIL_KEYS), ",".join(FAIL_KEYS)))
        print("  提示：若这是新机器而非验收环境，改用 --mode=portability")
        return 3
    if args.strict and UNVERIFIED:
        print("SELFCHECK_WARN strict 模式: UNVERIFIED=%s" % ",".join(UNVERIFIED))
        return 2
    print("SELFCHECK_OK  (UNVERIFIED 不计失败; --strict 可收紧)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
