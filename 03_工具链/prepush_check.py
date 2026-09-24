#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""prepush_check.py —— 推送前闸门**驱动器**（重建版 v2；2026-09-22）

v2 修掉了 v1 实测暴露的三类判据错误（都是"判据措辞没对样本"，与坑 176/177/178 同族）：
  ① **不再用全局通过/失败标记清单**：每条闸门**各声明自己的期望标记**；标记缺失时明确报
     `标记不匹配`，而不是泛泛 FAIL（v1 因此把 rc=0、输出 `GATES_OK` 的 G2 误判成失败）。
  ② **G23 按类别判，不按退出码**：pyflakes 因 `unused import` 返回 1，但本闸门要抓的是
     **`undefined name`**（坑 147/153 的 `steps_complete` / `pip_install`）⇒ 只看那一类；
     用退出码当判据会让闸门**永远红**，而永远红最后会训练人忽略红。
  ③ **G22 使用 `--verify-only`**：检查与发布分离，不在检查过程中修改交付物。
另：G1（编译 + 加载冒烟）改为**驱动器内自实现**（v1 用的 `--compile-smoke` 是我猜的参数），
    并沿用坑 173 的修法：**显式 `cwd=SKILL`**（旧版不传 cwd，导致脚本写相对路径 `out/` 时
    落到调用者目录 ⇒ 同一份检查在不同目录下结论不同）。

设计原则（"不要漏了"）
--------------------
1. 闸门清单以**现存入口**为准；入口不存在 → **FAIL**（不是 SKIP、不是省略）。
2. `LOST_GATES` 里**已知丢失的检查显式列出**并计入 FAIL —— 宁可总判决红，也不让检查悄悄消失。
3. 任何异常/超时 → FAIL。
4. 不重复实现：三方一致性直接用 `safe_pack_sync.py`。

用法：
    python prepush_check.py --list         # 清单与现存性
    python prepush_check.py                # 全跑
    python prepush_check.py --only G9,G28  # 只跑指定
"""
import argparse
import io
import json
import os
import py_compile
import re
import subprocess
import sys
import time

from _project_paths import bootstrap_paths
PATHS = bootstrap_paths(parse_cli=__name__ == "__main__")

PKG = PATHS.root
SKILL = PATHS.skill
TOOLS = PATHS.tools
# ★ 交付目录常量：我第一版在 G30 里直接写了 `DELIV` 而**它根本不存在**（NameError 在运行到那一行时才炸）。
#   教训：`py_compile` 只查语法、查不出未定义名 ⇒ 所以本轮同时把 G23（pyflakes undefined name）
#   的扫描范围从"只扫 Skill"扩到"也扫工具目录"（工具是产出证据的仪器，坏在这里最难发现）。
DELIV = PATHS.deliverables
PY = sys.executable

# id, 名称, argv(空=驱动器内实现), cwd, **期望标记**
GATES = [
    ("G1", "编译 + 加载冒烟（--help 真正执行模块顶层代码；cwd=SKILL，坑 173）", [], SKILL, "_DRIVER_"),
    ("G2", "Skill 自带闸门集合（90_selfcheck_gates.py）",
     [PY, os.path.join(SKILL, "scripts", "90_selfcheck_gates.py")], SKILL, "GATES_OK"),
    ("G3", "技能自检（selfcheck.py）", [PY, os.path.join(SKILL, "selfcheck.py")], SKILL, "四层自检通过"),
    ("G9", "兼容层自检（_envcompat.py --selftest）",
     [PY, os.path.join(SKILL, "scripts", "_envcompat.py"), "--selftest"], SKILL, "SELFCHECK_OK"),
    ("G13", "环境探测自检（00_probe_env.py --selftest）",
     [PY, os.path.join(SKILL, "scripts", "00_probe_env.py"), "--selftest"], SKILL, "PROFILE_SELFTEST_OK"),
    ("G14", "训练驱动诊断自检（50_train.py --diag-selftest）",
     [PY, os.path.join(SKILL, "scripts", "50_train.py"), "--diag-selftest"], SKILL, "DIAG_SELFTEST_OK"),
    ("G19", "A/B 判据纯函数自检（59 --selftest）",
     [PY, os.path.join(SKILL, "scripts", "59_config_ab.py"), "--selftest"], SKILL, "P59_SELFTEST_OK"),
    ("G24", "编排路径自检（59 --selftest-orchestration）",
     [PY, os.path.join(SKILL, "scripts", "59_config_ab.py"), "--selftest-orchestration"], SKILL,
     "P59_ORCH_SELFTEST"),
    ("G25", "双配置配对器自检（63 --selftest）",
     [PY, os.path.join(SKILL, "scripts", "63_two_config_ab.py"), "--selftest"], SKILL, "T63_SELFTEST"),
    # ★ 阶段 0：闸门**只读**。原实现调 `--apply` ⇒ 闸门自己会写文件（验证与发布没分离）。
    ("G22", "三方一致性（zip / 两处解包副本 / 活副本，逐文件哈希；**只读**）",
     [PY, os.path.join(TOOLS, "safe_pack_sync.py"), "--verify-only"], TOOLS, "SAFE_PACK_VERIFY_OK"),
    ("G23", "未定义名静态扫描（pyflakes；**只看 undefined name**）",
     [PY, "-m", "pyflakes"], SKILL, "_PYFLAKES_UNDEFINED_"),
    ("G26", "对齐审计：推荐配置 A（官方 dump 逐键）",
     [PY, os.path.join(TOOLS, "align_audit.py"), "--run",
      os.path.join(SKILL, "config", "templates", "qwen3_5_0_8B_recommended_A.yaml")], TOOLS,
     "ALIGN_AUDIT_OK"),
    ("G27", "对齐审计：保底配置 B（官方 dump 逐键）",
     [PY, os.path.join(TOOLS, "align_audit.py"), "--run",
      os.path.join(SKILL, "config", "templates", "qwen3_5_0_8B_fallback_B.yaml")], TOOLS,
     "ALIGN_AUDIT_OK"),
    ("G28", "坑表连续性（1..N 无缺号无重复）",
     [PY, os.path.join(TOOLS, "check_pitfalls.py")], TOOLS, "PITFALL_CHECK_OK"),
    # ---- 2026-09-22 补：三条原本"丢失"的技术闸门，按现存入口重建 ----
    ("G5", "SK04 判定链自检（sk04_judge **本体健在**，此处只补调用封装）",
     [PY, os.path.join(SKILL, "sk04_judge", "scripts", "selfcheck_sk04.py"),
      "--repo-root", SKILL], SKILL, "FINGERPRINT_OK"),
    ("G6", "判据负向对照（四项坏样本必须被拒绝）",
     [PY, os.path.join(TOOLS, "negative_controls.py")], TOOLS, "NEGATIVE_CONTROLS_OK"),
    ("G12", "证据逐文件哈希表（远端证据 15 文件）",
     [PY, os.path.join(TOOLS, "evidence_manifest.py"), "--verify", "--dir",
      PATHS.evidence], TOOLS, "EVIDENCE_MANIFEST_OK"),
    # ★ 2026-09-22：G7 从**人工项升为自动闸门**。起因是我自己踩的坑 188/189：
    #   交付 PDF 印的是"插入更正横幅**之前**"的内容（md 有横幅、PDF 没有），而当时 19 个
    #   自动闸门全 PASS —— 因为 G7 是 MANUAL（"pdf 与 md 内容一致"），没人拦。
    ("G7", "报告 PDF ↔ md 一致性（首部落没 + 口径限定语同现）",
     [PY, os.path.join(TOOLS, "check_report_pdf.py")], TOOLS, "REPORT_PDF_OK"),
    # 判据本身的负向自检也跟着进套件（G6/G23/G28 同规矩）：坏样本必须被拒，否则"PASS"没有含义
    ("G7S", "报告 PDF 判据负向自检（3 例：横幅丢失/限定语丢失/抽取为空）",
     [PY, os.path.join(TOOLS, "check_report_pdf.py"), "--selftest"], TOOLS, "REPORT_PDF_SELFTEST_OK"),
    # ★ 2026-09-22：G31 —— 协议 JSON 必须能解析（起因：我把 Python 相邻字符串拼接写进 JSON，
    #   文件根本不可解析，却连过两次打包；23 个闸门无一会报错 ⇒ 补上"能读得动"这条最基础的检查）
    ("G31", "协议 JSON 可解析性 + schema 键（机器要读的文件必须读得动）", [], "", "_DRIVER_"),
    ("G8", "交付目录结构（必需目录存在且非空）", [], "", "_DRIVER_"),
    # ★ 2026-09-22：G30 —— 文档里的"状态数字"必须与驱动器实际数量一致（坑 195）。
    #   自指：本闸门也计入总数 ⇒ 加它之后文档数字必须同步（这就是它要防的过期）。
    ("G30", "文档状态数字一致性（文档写的闸门数/人工项数 vs 驱动器实际）", [], "", "_DRIVER_"),
    # ★ 2026-09-22：官方可比性判据自检（外部复核指出"仍按卡数推导 ⇒ mock 在双卡下可能被误标 true"）
    ("G29", "profile 官方可比性判据自检（4 例：mock 未声明/空白声明/单 die/正对照）",
     [PY, os.path.join(SKILL, "scripts", "56_profile_run.py"), "--selftest-official"], SKILL,
     "PROF_OFFICIAL_SELFTEST_OK"),
    # ★ 2026-09-22：G18 从人工项升为自动闸门（交付源码包 vs Skill，同名文件内容必须一致）
    ("G18", "源代码包一致性（交付 07_源代码 vs Skill，**同名不同内容即失败**）",
     [PY, os.path.join(TOOLS, "check_source_consistency.py")], TOOLS, "SOURCE_CONSISTENCY_OK"),
    # ★ 2026-09-22（§4-4 产品化）：G32 —— "先量噪声"从**纪律**变成**闸门**。
    #   起因：`59 --null-test` 能量底噪、`59/63` 能算效应，但**两者之间没有任何机器判据** ——
    #   底噪的后果只写在 `resolution_note` 的一句话里（"差异若小于它则不可判定"），
    #   "测了噪声"与"按噪声下结论"之间**靠人脑连接**。这正是本项目最反复的失败模式：
    #   **一个声明存在、但没有闸门**（坑 153/192 同族："自检通过"本身成了假安全）。
    #   G32 跑 `62_reportability.py --selftest`（42 例，含把交付主结论 19.606% 复算出来）。
    ("G32", "性能达标判据自检（62 --selftest v2：★区间下界硬门 + 三类结果拆开 + 角色制）",
     [PY, os.path.join(SKILL, "scripts", "62_reportability.py"), "--selftest"], SKILL,
     "P62_SELFTEST_OK"),
    # ★ 2026-09-22：G33 —— 复核要求「远端运行代码 / 本地解析代码**分别记录哈希**，
    #   不能只靠操作者记得」。起因是坑 216：编排脚本在远端复用 59 的判据，而远端那份是旧版
    #   （无 win_ms_of）⇒ 判据语义随副本漂移。G33 只跑**判据逻辑的自检**（离线、确定性）；
    #   真正连远端取哈希的那次运行是**联网**的，不能当闸门（会话断了会造成假失败），
    #   因此按流程要求"跑远端实验前先跑一次 hash_consistency.py"。
    ("G33", "哈希一致性判据自检（hash_consistency --selftest：判据偏斜必须被识别）",
     [PY, os.path.join(TOOLS, "hash_consistency.py"), "--selftest"], TOOLS,
     "HASH_CONSISTENCY_SELFTEST_OK"),
]

# 只能**人工确认**的项（逐条打印 + 见 docs/交付前人工清单.md）；不再计入自动判决的 FAIL，
# 但总判决会输出专有措辞 PREPUSH_OK_WITH_MANUAL —— 防止被读成"全绿"。
MANUAL_GATES = [
    ("G4", "配置模板字段一致性", "旧驱动器内联；两套模板已由 G26/G27 逐键审计覆盖",
     "人工确认：config/templates/*.yaml 与 config/versions.lock 的 geometry/dataloader 键一致"),
    ("G11", "目标机依赖覆盖（torch/torch_npu 真机）", "旧驱动器内联",
     "人工确认：真机 05_preflight 输出（本轮已由 env.json + triton 修复记录覆盖）"),
    ("G17", "创意书字段完整性", "check_proposal.py / fill_proposal_fields.py 被删",
     "人工确认：01_项目创意书 是否仍有【待填/待确认】字样（用户负责）"),
]
# 注：G18（源代码包一致性）**已升为自动闸门**（check_source_consistency.py）⇒ 从人工清单移除，
#     否则同一次输出里会既 PASS 又 MANUAL（自相矛盾的状态比缺项更坏）。
# 注：G7（报告/PDF 一致性）**已于 2026-09-22 升为自动闸门**（check_report_pdf.py + 负向自检 G7S），
#     原因见 GATES 里的注释（坑 188/189 就是在这个"人工缺口"里发生的）。
#     顺带查出：交付 PDF 与其源 md **不同名**（`性能测试报告.pdf` ← `性能优化报告.md`），
#     所以"同名配对"写法在这里配不上对 ⇒ 已在 check_report_pdf.py 里显式登记这层映射。

LOST_GATES = []   # 已无"既没重建、也没写进人工清单"的项（2026-09-22 清空）

ENV_DEPS = ("torch", "torch_npu", "triton", "yaml", "numpy", "transformers", "mindspeed_mm",
            "mindspeed", "deepspeed")


def _run(argv, cwd, timeout):
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def gate_compile_smoke(cwd, timeout):
    """G1：编译全部 .py + 逐个 `--help` 加载冒烟（判据同旧 G3；cwd 显式钉死，坑 173）。"""
    pys = []
    for root, dirs, files in os.walk(SKILL):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "out", "refs", "docs", "examples")]
        pys += [os.path.join(root, f) for f in files if f.endswith(".py")]
    bad = []
    for p in sorted(pys):
        try:
            py_compile.compile(p, doraise=True)
        except Exception as exc:
            bad.append("%s 编译失败：%s" % (os.path.basename(p), exc))
    smoke_fail, smoke_skip = [], []
    for p in sorted(pys):
        try:
            r = _run([PY, p, "--help"], SKILL, 180)
        except subprocess.TimeoutExpired:
            smoke_fail.append("%s: --help 超时" % os.path.basename(p))
            continue
        blob = (r.stdout or "") + (r.stderr or "")
        if "Traceback (most recent call last)" not in blob:
            continue
        if "NameError" in blob:
            smoke_fail.append("%s: NameError（漏写 import）" % os.path.basename(p))
        elif "ModuleNotFoundError" in blob or "ImportError" in blob:
            # ★ 实测暴露：原写法用 `"'%s'" % d` 做**精确带引号匹配**，于是 `mindspeed` 匹配不到
            #   `'mindspeed_mm'` ⇒ 把"本机缺框架（应 SKIP）"误报成**冒烟失败**。
            #   这是"判据过紧 ⇒ 假失败"的一类（假失败会稀释真信号，和假通过一样有害）。
            #   改为"引号后的**前缀**匹配"，并把子模块名一并登记。
            dep = next((d for d in ENV_DEPS if re.search(r"['\"]%s" % re.escape(d), blob)), None)
            (smoke_skip if dep else smoke_fail).append(
                "%s(缺 %s)" % (os.path.basename(p), dep) if dep else "%s: %s" % (
                    os.path.basename(p), blob.strip().splitlines()[-1][:80]))
        else:
            smoke_fail.append("%s: %s" % (os.path.basename(p), blob.strip().splitlines()[-1][:80]))
    print("        %d 个 .py 编译；冒烟失败 %d、跳过 %d（本机缺环境依赖）"
          % (len(pys), len(smoke_fail), len(smoke_skip)))
    for x in (bad + smoke_fail)[:5]:
        print("        ! %s" % x)
    return (not bad and not smoke_fail), "G1_OK" if (not bad and not smoke_fail) else "G1_FAIL"


def gate_pyflakes(cwd, timeout):
    """G23（三层化）：工具是否正常运行 / 是否覆盖目标 / 是否满足要求。

    ★ Codex 复核 P1-b（我实测确认）：原实现只 grep `undefined name` 字面量 ⇒
      **pyflakes 根本没装**（输出 "No module named pyflakes"）时 undef 为空 ⇒ 报 `G23_OK`（误放行）。
      现在三层分开：rc≠0 或 stderr 含 No module named / 扫描 0 个文件 ⇒ **ERROR**（不是 PASS）。
    """
    files = []
    # ★ 2026-09-22：扫描范围从"只扫 Skill"扩到"**也扫工具目录**"。
    #   起因：我在 G30 里引用了不存在的常量 `DELIV`，`py_compile` 只查语法 ⇒ 查不出来，
    #   直到跑 prepush 时才 NameError 崩在那一行。工具是**产出证据的仪器**，坏在这里最难发现
    #   （它不会让交付物变红，只会让"检查"本身失真）。G23 本就该覆盖它。
    for base in (SKILL, TOOLS):
        for root, dirs, files_ in os.walk(base):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", "out", "refs", "_backup",
                                                    "_reanalysis", "protocols")]
            files += [os.path.join(root, f) for f in files_ if f.endswith(".py")]
    r = _run([PY, "-m", "pyflakes"] + files, SKILL, timeout)
    out = (r.stdout or "") + (r.stderr or "")
    if "No module named" in out or "not found" in out:
        print("        ERROR 工具无法运行：%s" % out.strip().splitlines()[-1][:90])
        return False, "ERROR(工具缺失)"
    if len(files) == 0:
        print("        ERROR 覆盖为空：没有扫到任何 .py ⇒ 不能当零问题")
        return False, "ERROR(零样本)"
    undef = [l for l in out.splitlines() if "undefined name" in l or "unable to detect undefined" in l]
    others = [l for l in out.splitlines() if l.strip() and l not in undef]
    print("        扫描 %d 个 .py；undefined name=%d；其他类别（不计入失败）=%d；rc=%d"
          % (len(files), len(undef), len(others), r.returncode))
    for l in undef[:5]:
        print("        ! %s" % l)
    if undef:
        return False, "G23_FAIL undefined=%d" % len(undef)
    return True, "G23_OK undefined=0 scanned=%d rc=%d" % (len(files), r.returncode)


def check_backend(argv, cwd):
    for a in argv[1:] if len(argv) > 1 else []:
        if a.startswith("-"):
            continue
        if os.path.isabs(a) and not os.path.exists(a):
            return False, "缺 %s" % os.path.relpath(a, PKG)
    return (os.path.isdir(cwd), "ok" if os.path.isdir(cwd) else "cwd 不存在 %s" % cwd)


AUTO_CLAIM = re.compile(r"自动化闸门\s*\**\s*(\d+)\s*/\s*\**\s*(\d+)")
MANUAL_CLAIM = re.compile(r"人工\s*\**\s*(\d+)\s*\**\s*项")
# 含这些词的句子是**在讲历史/在描述缺陷**，不是在声称当前状态 ⇒ 不当作状态声明
HISTORY_HINTS = ("历史", "更正", "曾", "旧值", "先前", "过去", "过期", "被证伪")
DOC_SCAN_SKIP = ("PITFALLS_坑表.md",)      # 坑表天然会引用旧数字（它记录的就是过期这件事）


def gate_doc_gate_counts():
    """G30：文档里的"状态数字"必须与**驱动器实际数量**一致（坑 195 的可自动发现化）。

    为什么需要：坑 195 的形态是"同一份交付包里，驱动器输出 21/21 + 人工 3 项，
    而文档仍写 18/18 + 人工 5 项"——**同一个事实两个副本，没有任何东西强制它们一致**。
    本闸门只做一件小事：把 `自动化闸门 N/N`、`人工 N 项` 这类**派生量**抓出来，与
    `len(GATES)`、`len(MANUAL_GATES)` 比对。

    ★ 自指说明：**本闸门自身也计入 `len(GATES)`** ⇒ 每次增减闸门后，文档里的数字必须同步
      （这正是它要防的那种过期）。若本闸门因"文档数字过期"而 FAIL，那是**真失败**，
      不是它算错了——正确动作是改文档，而不是改本闸门。
    """
    want_auto, want_manual = len(GATES), len(MANUAL_GATES)
    roots = [(SKILL, ("docs",)), (DELIV, ("02_README", "04_性能测试报告"))]
    claims, problems = 0, []
    for base, subs in roots:
        for sub in subs:
            d = os.path.join(base, sub)
            if not os.path.isdir(d):
                continue
            for r, _dd, fs in os.walk(d):
                for f in fs:
                    if not f.endswith(".md") or f in DOC_SCAN_SKIP:
                        continue
                    p = os.path.join(r, f)
                    try:
                        lines = io.open(p, encoding="utf-8-sig", errors="replace").read().splitlines()
                    except OSError as exc:
                        problems.append("读不了 %s：%s" % (f, exc))
                        continue
                    for i, ln in enumerate(lines, 1):
                        if any(h in ln for h in HISTORY_HINTS):
                            continue
                        m = AUTO_CLAIM.search(ln)
                        if m:
                            claims += 1
                            if (int(m.group(1)), int(m.group(2))) != (want_auto, want_auto):
                                problems.append("%s L%d：文档写『自动化闸门 %s/%s』，实际 %d/%d"
                                                % (f, i, m.group(1), m.group(2), want_auto, want_auto))
                        m2 = MANUAL_CLAIM.search(ln)
                        if m2:
                            claims += 1
                            if int(m2.group(1)) != want_manual:
                                problems.append("%s L%d：文档写『人工 %s 项』，实际 %d 项"
                                                % (f, i, m2.group(1), want_manual))
    print("        扫描状态数字声明 %d 处（期望 自动化=%d 人工=%d）" % (claims, want_auto, want_manual))
    if problems:
        for x in problems[:6]:
            print("        ! %s" % x)
        return False, "G30_FAIL 不一致=%d" % len(problems)
    return True, "G30_OK claims=%d 自动=%d 人工=%d" % (claims, want_auto, want_manual)


def gate_protocol_json():
    """G31：Skill 的 `protocols/*.json` 必须**能解析**且含 `schema` 键。

    起因（pitfall 210）：我把 Python 的**相邻字符串拼接**写法写进了 JSON（两个字符串挨着、没逗号），
    于是 `prefetch_depth_20260922.json` **根本无法解析** —— 而当时 23 个闸门**没有一个**会因此报错，
    它就这样被打了两次包（直到我要用脚本读它、`json.load` 抛错才发现）。
    ⇒ 交付物里凡是"机器要读的文件"，都必须有一条"能读得动"的闸门。
    """
    d = os.path.join(SKILL, "protocols")
    if not os.path.isdir(d):
        return False, "G31_FAIL protocols 目录不存在"
    bad, n = [], 0
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json"):
            continue
        n += 1
        p = os.path.join(d, f)
        try:
            obj = json.load(io.open(p, encoding="utf-8"))
        except Exception as exc:
            bad.append("%s 解析失败：%s" % (f, str(exc)[:70]))
            continue
        if not isinstance(obj, dict) or "schema" not in obj:
            bad.append("%s 缺 schema 键（协议必须自报 schema）" % f)
    print("        protocols/*.json 共 %d 个" % n)
    for x in bad[:5]:
        print("        ! %s" % x)
    if n == 0:
        return False, "G31_FAIL 零样本（没有任何协议文件 ⇒ 不能当通过）"
    if bad:
        return False, "G31_FAIL files=%d bad=%d" % (n, len(bad))
    return True, "G31_OK protocols=%d" % n


def gate_delivery_tree():
    """G8：交付目录结构体检（驱动器内实现；不碰文件，只数数与判存在）。"""
    root = DELIV
    need = ["00_数据与结论", "01_项目创意书", "02_README", "03_精度分析报告", "04_性能测试报告",
            "05_原始日志", "06_Skill", "07_源代码", "08_复现视频", "99_官方模板与要求"]
    bad = []
    for d in need:
        p = os.path.join(root, d)
        n = sum(len(fs) for _r, _dd, fs in os.walk(p)) if os.path.isdir(p) else -1
        if n <= 0:
            bad.append("%s(文件=%s)" % (d, n))
    zips = [f for f in os.listdir(os.path.join(root, "06_Skill"))
            if f.endswith(".zip")] if os.path.isdir(os.path.join(root, "06_Skill")) else []
    if len(zips) != 1:
        bad.append("06_Skill 下 zip 数=%d（应为 1 ⇒ 多版本会让评委困惑）" % len(zips))
    print("        交付目录 %d 个必需目录；zip 数=%d" % (len(need), len(zips)))
    for b in bad[:5]:
        print("        ! %s" % b)
    if bad:
        return False, "G8_FAIL"
    return True, "G8_OK dirs=%d zip=%s" % (len(need), zips[0])


def run_gate(gid, name, argv, cwd, marker, timeout=1800):
    """三层状态：先问"工具是否正常运行"，再问"是否覆盖目标"，最后问"是否满足要求"。"""
    if not argv:
        ok, why = (True, "ok") if os.path.isdir(SKILL) else (False, "SKILL 不存在")
    else:
        ok, why = check_backend(argv, cwd)
    if not ok:
        print("  ERROR %-4s %-50s 后端缺失：%s" % (gid, name[:50], why))
        return False
    t0 = time.time()
    try:
        if gid == "G1":
            passed, got = gate_compile_smoke(cwd, timeout)
        elif gid == "G30":
            passed, got = gate_doc_gate_counts()
        elif gid == "G31":
            passed, got = gate_protocol_json()
        elif gid == "G23":
            passed, got = gate_pyflakes(cwd, timeout)
        elif gid == "G8":
            passed, got = gate_delivery_tree()
        else:
            r = _run(argv, cwd, timeout)
            out = (r.stdout or "") + (r.stderr or "")
            blob_tail = [l for l in out.strip().splitlines() if l.strip()][-1:] or [""]
            if "No module named" in out or "command not found" in out:
                print("  ERROR %-4s %-50s 工具无法运行：%s" % (gid, name[:50], blob_tail[0][:80]))
                return False
            if r.returncode != 0 and marker in out:
                # rc≠0 却出现期望标记 ⇒ 局部用例过、整体失败（Codex 的坏例）
                print("  FAIL  %-4s %-50s rc=%s 但出现 %r ⇒ 判失败（局部通过不等于整体通过）"
                      % (gid, name[:50], r.returncode, marker))
                return False
            passed = (marker in out) and (r.returncode == 0) and ("Traceback" not in out)
            got = marker if marker in out else "<标记不匹配>"
    except subprocess.TimeoutExpired:
        print("  ERROR %-4s %-50s 超时 >%ds ⇒ ERROR（不是 SKIP）" % (gid, name[:50], timeout))
        return False
    dt = time.time() - t0
    print("  %-5s %-4s %-50s %.1fs | 判据=%s"
          % ("PASS" if passed else "FAIL", gid, name[:50], dt, got))
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = [x.strip().upper() for x in a.only.split(",") if x.strip()]

    print("== prepush_check v2（重建版）==")
    print("SKILL = %s (exists=%s)" % (SKILL, os.path.isdir(SKILL)))
    print("现存闸门 %d 条；已知丢失 %d 条（丢失项**计入 FAIL**，不静默省略）"
          % (len(GATES), len(LOST_GATES)))
    if a.list:
        for gid, name, argv, cwd, marker in GATES:
            ok, why = check_backend(argv, cwd)
            print("  %-4s %-8s %-48s 标记=%-22s %s"
                  % (gid, "OK" if ok else "MISSING", name[:48], marker, why))
        print("\n-- 已知丢失 --")
        for gid, name, why, alt in LOST_GATES:
            print("  %-4s %-46s %s | 替代=%s" % (gid, name[:46], why, alt))
        return 0

    results = {}
    for gid, name, argv, cwd, marker in GATES:
        if only and gid not in only:
            continue
        results[gid] = run_gate(gid, name, argv, cwd, marker)

    if LOST_GATES:
        print("\n-- 已知丢失（计入 FAIL）--")
        for gid, name, why, alt in LOST_GATES:
            print("  MISSING %-4s %-44s %s" % (gid, name[:44], why))
    print("\n-- 需人工确认（见 docs/交付前人工清单.md；不计入自动判决）--")
    for gid, name, why, how in MANUAL_GATES:
        print("  MANUAL %-4s %-40s %s" % (gid, name[:40], how))

    npass = sum(1 for v in results.values() if v)
    print("\n合计：现存 %d 条（PASS=%d FAIL=%d）+ 丢失 %d 条 = 应有 %d 条"
          % (len(results), npass, len(results) - npass, len(LOST_GATES),
             len(GATES) + len(LOST_GATES)))
    if npass != len(results) or LOST_GATES:
        print("RELEASE_BLOCKED —— 自动化闸门未全通过（或有丢失项）⇒ **阻断发布**；"
              "仍可继续只读分析工作")
        return 1
    print("AUTOMATED_OK_MANUAL_OPEN —— 自动化闸门 %d/%d 通过、无丢失项；"
          "**人工 %d 项未闭合 ⇒ 发布仍受人工门约束**（清单见 docs/交付前人工清单.md）"
          % (npass, len(results), len(MANUAL_GATES)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
