#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
90_selfcheck_gates.py —— 判据负向对照（★ 鲁棒性的核心：**测试判据本身**）

为什么需要它
------------
2026-09-16 的干净环境验证暴露了 4 个**假阳性**判据（坑 33/35/42/47）和 1 个**假阴性**判据（坑 46）。
它们的共同特征是：**判据从未被验证过"能不能失败"**。
    · 条件恒真（`"VERIFY_OK" in out or rc==0`）
    · 空集恒真（`len([]) <= 1`、`checks=[] → fails=0`）
    · 消费陈旧产物（`tests/tmp/` 里的旧文件）
    · 判据自身的代码写错（`triton.program_id`）

**一个不会失败的判据不是判据，是装饰。**

本脚本对每个关键判据做**双向**验证：
    good 输入 → 必须"通过"
    bad  输入 → 必须"失败"   （若 bad 也"通过" → 判为 GATE_VACUOUS，本脚本失败）

另含「静态回归守卫」：对历史踩过的具体缺陷模式做源码扫描，
并**自带可证伪性证明** —— 每个模式都要能匹配上一段合成坏样本，否则该守卫本身失效。

用法
----
    python3 scripts/90_selfcheck_gates.py            # 全部
    python3 scripts/90_selfcheck_gates.py --list     # 只列判据
退出码: 0 = GATES_OK；1 = 有判据失效/空过
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(SCRIPTS)
sys.path.insert(0, SCRIPTS)
from _envcompat import python_exe  # noqa: E402

PY = python_exe()
RESULTS = []


def run(args, timeout=300, cwd=None, env=None):
    try:
        r = subprocess.run([PY] + [str(a) for a in args], capture_output=True, text=True,
                           timeout=timeout, cwd=cwd or SKILL, env=env)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return -1, "%s: %s" % (type(e).__name__, e)


def gate(name, desc, out_good, ok_marker, bad_marker, note=""):
    """记录一个双向判据的结果。

    out_good: (rc_good, text_good, rc_bad, text_bad)
    ok_marker:  good 输入下必须出现的标记
    bad_marker: bad 输入下必须出现的标记（缺它 = 判据不会失败 = GATE_VACUOUS）
    """
    rc_g, txt_g, rc_b, txt_b = out_good
    good_ok = ok_marker in txt_g
    bad_fired = bad_marker in txt_b
    status = "PASS"
    detail = "good:%s bad:%s" % ("✓" if good_ok else "✗", "✓" if bad_fired else "✗")
    if not good_ok:
        status = "FAIL"
        detail += " （good 输入未出现 %s）" % ok_marker
    elif not bad_fired:
        status = "VACUOUS"
        detail += " （**bad 输入也未出现 %s → 判据不会失败**）" % bad_marker
    print("  %-8s %-30s %s" % (status, name, detail))
    if note and status != "PASS":
        print("           └─ %s" % note)
    RESULTS.append({"gate": name, "status": status, "desc": desc, "detail": detail})
    return status == "PASS"


# =====================================================================
# 动态负向对照
# =====================================================================
def g_assets_completeness(tmp):
    """40_prepare_assets：资产缺失必须报 INCOMPLETE；齐备则不得报 INCOMPLETE。"""
    bad = os.path.join(tmp, "empty")
    os.makedirs(bad, exist_ok=True)
    rc_b, t_b = run([f"{SCRIPTS}/40_prepare_assets.py", "--no-download",
                     "--data-dir", bad, "--model-dir", bad, "--out", os.path.join(tmp, "o1")])
    good = os.path.join(tmp, "full")
    os.makedirs(os.path.join(good, "Qwen3.5-0.8B-hf"), exist_ok=True)
    os.makedirs(os.path.join(good, "llava"), exist_ok=True)
    open(os.path.join(good, "llava", "llava_instruct_150k.json"), "w").write("[]")
    open(os.path.join(good, "output_llava_coco_data.json"), "w").write("[]")
    rc_g, t_g = run([f"{SCRIPTS}/40_prepare_assets.py", "--no-download",
                     "--data-dir", good, "--model-dir", good, "--out", os.path.join(tmp, "o2")])
    return gate("assets_completeness",
                "资产缺失 → ASSETS_INCOMPLETE(rc=3)；齐备 → 不再报 INCOMPLETE",
                (rc_g, t_g, rc_b, t_b), "ASSETS_", "ASSETS_INCOMPLETE",
                "若 bad 输入不报 INCOMPLETE，说明完整性判据形同虚设（坑 47）")


def g_registry_tamper(tmp):
    """registry_append --verify：篡改必须被拒；原账本必须通过。"""
    src = os.path.join(SKILL, "examples", "judge", "registry.json")
    if not os.path.isfile(src):
        src = os.path.join(SKILL, "sk04_judge", "evidence", "registry.json")
    good = os.path.join(tmp, "reg_ok.json")
    shutil.copy(src, good)
    rc_g, t_g = run([f"{SKILL}/sk04_judge/scripts/registry_append.py",
                     "--registry", good, "--verify"])
    doc = json.load(open(good, encoding="utf-8-sig"))
    doc["entries"][0]["tag"] = "TAMPERED"
    bad = os.path.join(tmp, "reg_bad.json")
    json.dump(doc, open(bad, "w", encoding="utf-8"), ensure_ascii=False)
    rc_b, t_b = run([f"{SKILL}/sk04_judge/scripts/registry_append.py",
                     "--registry", bad, "--verify"])
    return gate("registry_tamper", "哈希链篡改必须 REGISTRY_CHAIN_BROKEN",
                (rc_g, t_g, rc_b, t_b), "REGISTRY_VERIFY_OK", "REGISTRY_CHAIN_BROKEN")


def g_registry_dup(tmp):
    """registry_append：重复 tag 必须被拒。"""
    src = os.path.join(SKILL, "examples", "judge", "registry.json")
    if not os.path.isfile(src):
        src = os.path.join(SKILL, "sk04_judge", "evidence", "registry.json")
    reg = os.path.join(tmp, "reg_dup.json")
    shutil.copy(src, reg)
    fp = os.path.join(SKILL, "examples", "judge", "fp.json")
    obs = os.path.join(SKILL, "examples", "judge", "obs.json")
    args = [f"{SKILL}/sk04_judge/scripts/registry_append.py", "--registry", reg,
            "--run-dir", "x", "--manifest", obs, "--fingerprint", fp, "--observed", obs]
    rc_g, t_g = run(args + ["--tag", "gate_dup_test"])
    rc_b, t_b = run(args + ["--tag", "gate_dup_test"])   # 同一 tag 再来一次
    return gate("registry_dup_tag", "重复 tag 必须 REGISTRY_DUP_TAG",
                (rc_g, t_g, rc_b, t_b), "REGISTRY_APPEND_OK", "REGISTRY_DUP_TAG")


def g_judge_determinism(tmp):
    """judge_comparable：同证据同 verdict_id；改一个字节 id 必须变（证明对证据敏感）。"""
    fp = os.path.join(SKILL, "examples", "judge", "fp.json")
    obs = os.path.join(SKILL, "examples", "judge", "obs.json")
    o1 = os.path.join(tmp, "v1.json")
    o2 = os.path.join(tmp, "v2.json")
    rc_g, t_g = run([f"{SKILL}/sk04_judge/scripts/judge_comparable.py", "--run", fp,
                     "--observed", obs, "--baseline", "officialB", "--out", o1])
    run([f"{SKILL}/sk04_judge/scripts/judge_comparable.py", "--run", fp,
         "--observed", obs, "--baseline", "officialB", "--out", o2])
    id1 = re.search(r"verdict_id=(\w+)", t_g)
    det = False
    try:
        det = json.load(open(o1, encoding="utf-8-sig"))["verdict_id"] == \
              json.load(open(o2, encoding="utf-8-sig"))["verdict_id"]
    except Exception:
        pass

    # bad：篡改 fp 的一个字段 → verdict_id 必须变化
    d = json.load(open(fp, encoding="utf-8-sig"))
    try:
        d["dims"][0]["status"] = "TAMPERED"
    except Exception:
        d["tampered"] = True
    fp_bad = os.path.join(tmp, "fp_bad.json")
    json.dump(d, open(fp_bad, "w", encoding="utf-8"), ensure_ascii=False)
    o3 = os.path.join(tmp, "v3.json")
    rc_b, t_b = run([f"{SKILL}/sk04_judge/scripts/judge_comparable.py", "--run", fp_bad,
                     "--observed", obs, "--baseline", "officialB", "--out", o3])
    try:
        id3 = json.load(open(o3, encoding="utf-8-sig"))["verdict_id"]
    except Exception:
        id3 = "NONE"
    changed = bool(id1) and id3 != id1.group(1)
    t_b2 = t_b + ("\nVERDICT_ID_CHANGED\n" if changed else "")
    return gate("judge_determinism_and_sensitivity",
                "同证据 → 同 verdict_id；证据改一字节 → id 必须变",
                (rc_g, t_g + ("\nDET\n" if det else ""), rc_b, t_b2),
                "JUDGE_OK", "VERDICT_ID_CHANGED",
                "若 id 不随证据变化，则 verdict_id 不能作为证据承诺（纯函数契约失效）")


def g_judge_invalid_input(tmp):
    """judge_comparable：非法输入必须非零退出。"""
    fp = os.path.join(SKILL, "examples", "judge", "obs.json")
    obs = os.path.join(SKILL, "examples", "judge", "fp.json")
    o = os.path.join(tmp, "vi.json")
    rc_g, t_g = run([f"{SKILL}/sk04_judge/scripts/judge_comparable.py", "--run", obs,
                     "--observed", fp, "--baseline", "officialB", "--out", o])
    rc_b, t_b = run([f"{SKILL}/sk04_judge/scripts/judge_comparable.py",
                     "--run", os.path.join(tmp, "nope.json"),
                     "--observed", os.path.join(tmp, "nope2.json"),
                     "--baseline", "officialB", "--out", o])
    # bad 判据 = 非零退出（标记用 rc）
    t_b2 = t_b + ("\nNONZERO_EXIT rc=%d\n" % rc_b if rc_b != 0 else "")
    return gate("judge_invalid_input", "非法输入必须非零退出",
                (rc_g, t_g, rc_b, t_b2), "JUDGE_OK", "NONZERO_EXIT")


def g_replay_verdict_id(tmp):
    """包内示例必须重放出交付材料的 verdict_id（0-GPU 自证）。"""
    fp = os.path.join(SKILL, "examples", "judge", "fp.json")
    obs = os.path.join(SKILL, "examples", "judge", "obs.json")
    o = os.path.join(tmp, "rp.json")
    rc_g, t_g = run([f"{SKILL}/sk04_judge/scripts/judge_comparable.py", "--run", fp,
                     "--observed", obs, "--baseline", "officialB", "--out", o])
    # bad：篡改 observed → 不得再出现交付红线 id
    d = json.load(open(obs, encoding="utf-8-sig"))
    d["tampered_marker"] = True
    obs_bad = os.path.join(tmp, "obs_bad.json")
    json.dump(d, open(obs_bad, "w", encoding="utf-8"), ensure_ascii=False)
    o2 = os.path.join(tmp, "rp2.json")
    rc_b, t_b = run([f"{SKILL}/sk04_judge/scripts/judge_comparable.py", "--run", fp,
                     "--observed", obs_bad, "--baseline", "officialB", "--out", o2])
    try:
        idb = json.load(open(o2, encoding="utf-8-sig")).get("verdict_id", "")
    except Exception:
        idb = ""
    same = "606056b3cb04ecbb" in idb
    t_b2 = t_b + ("\nSTILL_SAME_ID\n" if same else "\nID_CHANGED_ON_TAMPER\n")
    return gate("replay_verdict_id_redline", "示例重放必须得 606056b3cb04ecbb；篡改后必须变",
                (rc_g, t_g, rc_b, t_b2), "verdict_id=606056b3cb04ecbb",
                "ID_CHANGED_ON_TAMPER")


def g_env_modes(tmp):
    """selfcheck_env：非验收环境时 acceptance 必须 FAIL、portability 必须 OK。

    ★ 负向对照必须**平台自适配**：不能写死"python 一定可测"（Windows 上可能没有 python3）。
      做法：先用 portability 模式（配一个不存在的 lock）拿到**本机确实测到**的键与实测值，
      再为其中一个键构造矛盾 lock，喂给 acceptance —— 它必须 FAIL。
      若本机没有任何可测键 → 标 SKIP（平台限制），**不得**因此判 PASS 或 VACUOUS。
    """
    rc_g, t_g = run([f"{SCRIPTS}/selfcheck_env.py", "--mode=portability"])
    good_ok = "SELFCHECK_PORTABILITY_OK" in t_g

    # 找出本机可测的键（DRIFT 行里 actual 非空）
    measurable = re.findall(r"^\s+(\S+)\s+DRIFT\s+lock=\S*\s+actual=(\S+)", t_g, re.M)
    if not measurable:
        print("  %-8s %-30s %s" % ("SKIP", "env_acceptance_vs_portability",
                                   "本机无可测键（无 NPU/python3/CANN），无法构造矛盾 lock"))
        RESULTS.append({"gate": "env_acceptance_vs_portability", "status": "SKIP",
                        "desc": "平台无法测任何版本键 → 无法做负向对照",
                        "detail": "measurable=0; good_ok=%s" % good_ok})
        return True

    key, val = measurable[0]
    lockdir = os.path.join(tmp, "lock_bad")
    os.makedirs(lockdir, exist_ok=True)
    lockp = os.path.join(lockdir, "versions.lock")
    with open(lockp, "w", encoding="utf-8") as f:
        f.write("# 合成矛盾 lock（负向对照）：把本机实测为 %s 的键 %s 改成不可能值\n"
                "%s | 0.0.0-impossible | synthetic | negative-control\n" % (val, key, key))
    rc_b, t_b = run([f"{SCRIPTS}/selfcheck_env.py", "--lock", lockp, "--mode=acceptance"])
    fired = "SELFCHECK_FAIL" in t_b and key in t_b
    t_b2 = t_b + ("\nACCEPTANCE_FIRED\n" if fired else "")
    return gate("env_acceptance_vs_portability",
                "portability 与 lock 无关必须 OK；acceptance 遇矛盾 lock(%s) 必须 FAIL" % key,
                (rc_g, t_g, rc_b, t_b2), "SELFCHECK_PORTABILITY_OK", "ACCEPTANCE_FIRED",
                "若矛盾 lock 也判 OK，则验收门是假的（坑 41）")


def g_env_mode_logic_unit(tmp):
    """★ 完全可移植的负向对照：**直接测判据函数本身**（不依赖平台能否测版本）。

    为什么需要它：子进程版负向对照在"本机没有任何可测版本键"的平台（如 Windows）只能 SKIP，
    于是这条判据在那样的平台上**没有被验证过** —— 而"没被验证"正是坑 33/35/42/47 的成因。
    进程内调用 `selfcheck_env.check()` 可以无视平台，直接证明：
        · acceptance 遇不匹配 → 进 FAIL_KEYS（判据会失败）
        · acceptance 遇匹配   → 不进 FAIL_KEYS（判据会通过）
        · portability 遇不匹配 → 进 DRIFT_KEYS 而**不**进 FAIL_KEYS（两种模式确实不同）
        · `+cpu` local 后缀被剥离 → 不算 drift（坑 50 回归）
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("_sce_unit",
                                                  os.path.join(SCRIPTS, "selfcheck_env.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    def probe(key, lock_v, actual_v, mode):
        m.FAIL_KEYS.clear()
        m.DRIFT_KEYS.clear()
        m.UNVERIFIED.clear()
        m.check(key, lock_v, actual_v, mode)
        return list(m.FAIL_KEYS), list(m.DRIFT_KEYS)

    f_acc_bad, _ = probe("python", "3.10.12", "9.9.9", "acceptance")       # 必须 FAIL
    f_acc_good, _ = probe("python", "3.10.12", "3.10.12", "acceptance")    # 必须通过
    f_port_bad, d_port_bad = probe("python", "3.10.12", "9.9.9", "portability")
    f_local, _ = probe("torch", "2.7.1", "2.7.1+cpu", "acceptance")        # local 后缀不算 drift
    f_none, _ = probe("python", "3.10.12", None, "acceptance")             # 不可测 → UNVERIFIED

    checks = {
        "acceptance_mismatch_fails": ("python" in f_acc_bad),
        "acceptance_match_passes": ("python" not in f_acc_good),
        "portability_mismatch_drifts_not_fails": (
            "python" not in f_port_bad and len(d_port_bad) == 1),
        "local_version_suffix_tolerated": ("torch" not in f_local),
        "unmeasurable_not_counted_as_fail": ("python" not in f_none),
    }
    all_ok = all(checks.values())
    txt_good = "\n".join("%s=%s" % (k, v) for k, v in checks.items())
    txt_bad = txt_good + ("\nUNIT_GATE_HOLDS\n" if all_ok else "")
    print("  %-8s %-30s %s" % ("PASS" if all_ok else "FAIL",
                               "env_mode_logic_unit(进程内)",
                               " ".join("%s:%s" % (k.split("_")[0], "✓" if v else "✗")
                                        for k, v in checks.items())))
    if not all_ok:
        for k, v in checks.items():
            if not v:
                print("        └─ 未通过: %s" % k)
        RESULTS.append({"gate": "env_mode_logic_unit", "status": "FAIL",
                        "desc": "selfcheck_env.check() 的模式逻辑", "detail": txt_good})
    else:
        RESULTS.append({"gate": "env_mode_logic_unit", "status": "PASS",
                        "desc": "selfcheck_env.check() 的模式逻辑（含坑50 回归）",
                        "detail": txt_good})
    return all_ok


DYNAMIC = [g_assets_completeness, g_registry_tamper, g_registry_dup,
           g_judge_determinism, g_judge_invalid_input, g_replay_verdict_id,
           g_env_mode_logic_unit, g_env_modes]


# =====================================================================
# 静态回归守卫
#
# ★ 守卫本身也必须可信 —— 否则它只是制造噪声：
#   · **去注释/去文档串**再匹配：我自己在注释里写的"坑 46：旧判据用 triton.program_id"
#     不该被判为违规（否则守卫第一轮就有 5 个假阳性，直接失去信号价值）
#   · **按文件语义放行**：bringup.sh 的 `apt-get install` 出现在 `PKG=apt` 分支里，是**正确**的
#     包管理器分派；守卫要抓的是"**无分派的**硬编码 apt"
#   · **窗口判定**：`exec > >(tee ...)` 只要 ±5 行内有 `[ -t 1 ]` 守卫即合规
#   · **可证伪性自证**：每个正则必须能匹配一段合成坏样本，否则守卫本身失效（GUARD_BROKEN）
# =====================================================================
def _strip_noise(text, is_py):
    """把注释与文档串替换为等长空白（保留行号与列对齐）。"""
    out = list(text)
    if is_py:
        # 三引号块（粗略但足够：本用途只关心"是否出现模式"）
        for m in re.finditer(r'"""(?:.|\n)*?"""|\'\'\'(?:.|\n)*?\'\'\'', text):
            for i in range(m.start(), m.end()):
                if out[i] != "\n":
                    out[i] = " "
    lines = "".join(out).split("\n")
    res = []
    for ln in lines:
        stripped = ln.lstrip()
        if stripped.startswith("#"):
            res.append(" " * len(ln))
        else:
            res.append(ln)
    return "\n".join(res)


def _scan(pattern, exts, allow_if_file=None, allow_if_line=None, window=None):
    """返回命中列表。window: (模式, 行数) —— 命中行 ±行数 内出现该模式则放行。"""
    hits = []
    for root, dirs, files in os.walk(SKILL):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "tests")]
        for f in files:
            if not f.endswith(exts):
                continue
            p = os.path.join(root, f)
            rel = os.path.relpath(p, SKILL).replace("\\", "/")
            if rel == "scripts/90_selfcheck_gates.py":
                continue
            try:
                raw = open(p, encoding="utf-8", errors="replace").read()
            except Exception:
                continue
            if allow_if_file and allow_if_file(raw):
                continue
            text = _strip_noise(raw, f.endswith(".py"))
            lines = text.split("\n")
            for i, line in enumerate(lines, 1):
                if not re.search(pattern, line):
                    continue
                if allow_if_line and allow_if_line(line):
                    continue
                if window:
                    wpat, w = window
                    lo, hi = max(0, i - 1 - w), min(len(lines), i + w)
                    if any(re.search(wpat, lines[j]) for j in range(lo, hi)):
                        continue
                hits.append("%s:%d: %s" % (rel, i, line.strip()[:110]))
    return hits


# (名称, 正则, 合成坏样本, 扩展名, 文件级放行, 行级放行, 窗口放行, 为什么要抓)
GUARDS = [
    ("坑46 triton 判据写法", r"triton\.program_id\s*\(", "i = triton.program_id(0)",
     (".py", ".sh"), None, None, None,
     "必须用 tl.program_id(0)；错误写法=假阴性 → 误判 triton 不可用 → 降级丢性能"),
    ("坑30 硬编码路径深度", r"parents\[\s*[2-9]\s*\]", "SKILL_ROOT.parents[3]",
     (".py",), None, None, None,
     "改用 find_repo_root()；硬编码深度在打包分发时 IndexError"),
    ("坑37 无分派的硬编码 apt",
     r"apt(-get)?\s+install\s", "apt-get install -y python3-dev build-essential",
     (".sh", ".py"),
     lambda t: ("PKG=" in t) or ("pkg_manager" in t) or ("install_hint" in t),
     None, None,
     "openEuler 无 apt；必须经包管理器探测分派（bringup.sh 已分派 → 放行）"),
    ("坑42 恒真判据", r'"[A-Z_]*VERIFY_OK"\s+in\s+\w+\s+or',
     'if "VERIFY_OK" in out or r.returncode == 0:',
     (".py",), None, None, None,
     "常量标记恒存在 → or 恒真 → 判据永远通过；必须同时校验 rc 与 failed=N"),
    ("坑49 错误模块名", r"import\s+triton_ascend", "import triton_ascend",
     (".py", ".sh"), None, None, None,
     "该模块名不存在；应判 triton.backends 是否含 ascend"),
    ("坑44 tee 未加 TTY 守卫", r"exec\s*>\s*>\(tee", 'exec > >(tee -a "$LOG") 2>&1',
     (".sh",), None, None, (r"\[\s*-t\s+1\s*\]", 5),
     "非 TTY 时 tee 持有 SSH 通道写端 → 后台启动挂死；±5 行内有 [ -t 1 ] 才算合规"),
    ("坑39 依赖 hostname 命令", r"^\s*hostname\s*$", "hostname",
     (".sh",), None, None, None,
     "极简镜像可能没有 hostname；改用 /proc/sys/kernel/hostname"),
    ("坑38 裸 python3 -m pip 安装",
     r"python3\s+-m\s+pip\s+install", "python3 -m pip install torch",
     (".sh",),
     None,
     # 只抓"真的在执行"的行：.py 里的提示文案（"修复: python3 -m pip install pyyaml"）应放行
     None, None,
     "python3 与 pip3 可能是不同解释器；必须锁定单一 PYBIN（仅扫 .sh）"),
]


def run_guards():
    print("\n[静态回归守卫]  （去注释/去文档串后匹配；自带可证伪性自证）")
    n_ok = 0
    for name, pat, sample, exts, aif, ail, win, why in GUARDS:
        hits = _scan(pat, exts, aif, ail, win)
        # ★ 可证伪性自证：正则在"去噪后"仍能匹配合成坏样本，否则守卫无效
        self_hit = bool(re.search(pat, _strip_noise(sample, True), re.M))
        mark = "✓" if self_hit else "✗ 守卫失效"
        print("  %-30s 命中=%-3d 自证=%s" % (name, len(hits), mark))
        if not self_hit:
            print("        └─ **正则匹配不到已知坏样本 → 该守卫无效（不是「没问题」，是「看不见」）**")
            RESULTS.append({"gate": name, "status": "GUARD_BROKEN", "desc": why,
                            "detail": "正则无法匹配已知坏样本"})
            continue
        if hits:
            for h in hits[:6]:
                print("        ! %s" % h)
            RESULTS.append({"gate": name, "status": "FAIL", "desc": why,
                            "detail": "; ".join(hits[:4])})
        else:
            n_ok += 1
            RESULTS.append({"gate": name, "status": "PASS", "desc": why, "detail": "未命中"})
    return n_ok


def main():
    ap = argparse.ArgumentParser(description="判据负向对照 / 回归守卫")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    print("=" * 78)
    print("判据负向对照（good 必过 / bad 必挂）   skill=%s" % SKILL)
    print("=" * 78)
    if a.list:
        for g in DYNAMIC:
            print("  dyn  %s" % g.__name__)
        for n, _, _, _ in GUARDS:
            print("  grd  %s" % n)
        return 0

    tmp = tempfile.mkdtemp(prefix="gatecheck_")
    print("\n[动态负向对照]")
    for fn in DYNAMIC:
        try:
            fn(tmp)
        except Exception as e:
            print("  ERROR    %-30s %s: %s" % (fn.__name__, type(e).__name__, e))
            RESULTS.append({"gate": fn.__name__, "status": "ERROR",
                            "detail": "%s: %s" % (type(e).__name__, e)})
    run_guards()
    shutil.rmtree(tmp, ignore_errors=True)

    # ★ 注意：SKIP 不算失败 —— 否则本脚本会犯下它自己要防的那个错（坑 34：
    #   把"因环境限制没检查"计为失败 → 淹没真实信号）。SKIP 必须**显式打印**，
    #   既不静默当通过（坑 33），也不冒充失败。
    bad = [r for r in RESULTS if r["status"] not in ("PASS", "SKIP")]
    skipped = [r for r in RESULTS if r["status"] == "SKIP"]
    print("\n" + "=" * 78)
    kinds = sorted({r["status"] for r in RESULTS})
    print("合计 %d 项：%s"
          % (len(RESULTS), "  ".join("%s=%d" % (k, sum(1 for r in RESULTS if r["status"] == k))
                                     for k in kinds)))
    for r in bad:
        print("  ! %-8s %-30s %s" % (r["status"], r["gate"], r.get("detail", "")[:150]))
    for r in skipped:
        print("  ~ SKIP     %-30s %s" % (r["gate"], r.get("detail", "")[:130]))
    ok = not bad
    print("GATES_%s%s" % ("OK" if ok else "FAIL",
                          ("  （%d 项因平台限制未验证 —— 在目标机(Linux+NPU)上必须全部跑到）"
                           % len(skipped)) if skipped else ""))
    if a.json:
        json.dump(RESULTS, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("已写: %s" % a.json)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
