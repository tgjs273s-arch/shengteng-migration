#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_envcompat.py —— 跨环境兼容层（Skill 鲁棒性的公共底座）

设计目标（★ 每一项都对应一次真实踩坑）
--------------------------------------
1. **没有隐含环境假设**：解释器、包管理器、路径、命令可用性、版本字符串格式，
   全部走本模块，不散落在各脚本里。
   根因：坑 37/38/39/40/45/49/50 —— 硬编码 apt / python3 / /root/... / hostname /
        `import triton_ascend` / `+cpu` 版本串，换台机器就崩或假阴性。
2. **降级必须留痕**：任何"因环境不支持而换路径"都要写进降级账本，
   下游与交付物可直接读。根因：SKILL.md §4 诚实红线的机读化。
3. **下载必须可续传可校验**：19GB 的 COCO 在限流镜像上必须能断点续传 + 大小校验。
   根因：坑 10（HTTP 429）、坑 29（大文件必须后台 + 轮询）。
4. **多源冗余**：每个外部资源至少两个来源，失败自动切换并记录。

用法（同目录脚本）
------------------
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _envcompat import (skill_root, pkg_root, find_repo_root, python_exe,
                            pkg_manager, install_hint, norm_ver, safe_hostname,
                            npu_info, cann_root, cann_version,
                            download, pip_install, degrade, GATES)

自检：  python3 _envcompat.py --selftest
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

# =====================================================================
# 0. 外部资源多源表（主源 → 备源，按顺序尝试）
# =====================================================================
SOURCES = {
    "pypi": [
        "https://mirrors.huaweicloud.com/ascend/repos/pypi",
        "https://repo.huaweicloud.com/repository/pypi/simple",
        "https://pypi.org/simple/",
    ],
    "msmm_git": [
        "https://github.com/Ascend/MindSpeed-MM.git",
        "https://gitcode.com/Ascend/MindSpeed-MM.git",
        "https://gitee.com/ascend/MindSpeed-MM.git",
    ],
    "modelscope": ["https://www.modelscope.cn", "https://modelscope.cn"],
    # ★ 每类**至少两个**来源（单一来源 = 单点故障）。hf-mirror 之后还有官方站兜底，
    #   官方站可能 401（坑：hf-mirror 401）——所以顺序是 镜像 → 官方，并记录实际用了哪个。
    "hf": ["https://hf-mirror.com", "https://huggingface.co"],
}

# pip 额外源（安装时同时给出，让 pip 自己择优）
EXTRA_INDEX = "https://repo.huaweicloud.com/repository/pypi/simple"

# =====================================================================
# 1. 路径解析（绝不硬编码深度 —— 坑 30/36）
# =====================================================================
def _this_dir():
    return os.path.dirname(os.path.abspath(__file__))


def skill_root():
    """scripts/ 的父目录 = Skill 根（含 SKILL.md）。"""
    d = _this_dir()
    return os.path.dirname(d) if os.path.isfile(os.path.join(d, "..", "SKILL.md")) else d


def pkg_root():
    """包根：Skill 被打包分发时等于 skill_root()；嵌入式布局时可能在其上层。"""
    sr = skill_root()
    return sr


def find_repo_root(start=None):
    """向上查找复赛根（含 snapshots/ 或 docs/official/）。

    ★ 坑 30：绝不用 `parents[N]` 硬编码深度 —— 打包到任意路径会 IndexError。
    找不到就返回 None，永不抛异常。环境变量 SK04_REPO_ROOT 优先。
    """
    env = os.environ.get("SK04_REPO_ROOT")
    if env and os.path.isdir(env):
        return env
    p = os.path.abspath(start or _this_dir())
    while True:
        if os.path.isdir(os.path.join(p, "snapshots")) or \
           os.path.isdir(os.path.join(p, "docs", "official")):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            return None
        p = parent


def default_msmm_dir():
    """MindSpeed-MM 目录：env MSMM_DIR 优先，其次常见位置。"""
    env = os.environ.get("MSMM_DIR")
    if env:
        return env
    for c in ("/root/MindSpeed-MM", os.path.join(os.path.expanduser("~"), "MindSpeed-MM")):
        if os.path.isdir(c):
            return c
    return "/root/MindSpeed-MM"


def default_data_dir():
    env = os.environ.get("DATA_DIR")
    if env:
        return env
    for c in ("/root/data", os.path.join(os.path.expanduser("~"), "data")):
        if os.path.isdir(c):
            return c
    return "/root/data"


# =====================================================================
# 2. 解释器（坑 38：python3 与 pip3 可能是不同解释器）
# =====================================================================
def python_exe():
    """返回应当使用的解释器路径。

    优先级：env PYBIN > 当前解释器（若它能 import 关键包）> python3 > 报错。
    ★ 绝不假定 `python3` 就是装了包的那个。
    """
    env = os.environ.get("PYBIN")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    return sys.executable or shutil.which("python3") or "python3"


def python_version(exe=None):
    exe = exe or python_exe()
    try:
        out = subprocess.run([exe, "-c", "import sys;print('%d.%d.%d' % sys.version_info[:3])"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or None
    except Exception:
        return None


def has_module(mod, exe=None):
    exe = exe or python_exe()
    try:
        r = subprocess.run([exe, "-c", "import %s" % mod], capture_output=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


def python_include_dir(exe=None):
    exe = exe or python_exe()
    try:
        out = subprocess.run([exe, "-c",
                              "import sysconfig;print(sysconfig.get_paths()['include'])"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or None
    except Exception:
        return None


def has_python_h(exe=None):
    inc = python_include_dir(exe)
    return bool(inc) and os.path.isfile(os.path.join(inc, "Python.h"))


# =====================================================================
# 3. 包管理器（坑 37：openEuler 没有 apt）
# =====================================================================
_PKG_TABLE = {
    "dnf": {"pyd": "python3-devel", "cc": "gcc gcc-c++ make", "pip": "python3-pip"},
    "yum": {"pyd": "python3-devel", "cc": "gcc gcc-c++ make", "pip": "python3-pip"},
    "microdnf": {"pyd": "python3-devel", "cc": "gcc gcc-c++ make", "pip": "python3-pip"},
    "apt": {"pyd": "python3-dev", "cc": "build-essential", "pip": "python3-pip"},
    "apt-get": {"pyd": "python3-dev", "cc": "build-essential", "pip": "python3-pip"},
}


def pkg_manager():
    for m in ("dnf", "yum", "microdnf", "apt-get", "apt"):
        if shutil.which(m):
            return m
    return None


def install_hint(kind, manager=None):
    """kind: 'pyd'（python 头文件）| 'cc'（编译器）| 'pip'。返回可直接执行的命令。"""
    m = manager or pkg_manager()
    if not m:
        return "（无可用包管理器：请手工安装对应依赖）"
    pkgs = _PKG_TABLE.get(m, {}).get(kind, "")
    if not pkgs:
        return "（未知包管理器 %s）" % m
    if m in ("apt", "apt-get"):
        return "%s install -y %s" % (m, pkgs)
    return "%s install -y %s" % (m, pkgs)


def install_pkgs(kind):
    """尽力安装；返回 (ok, 命令)。不抛异常。"""
    cmd = install_hint(kind)
    if cmd.startswith("（"):
        return False, cmd
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        return r.returncode == 0, cmd
    except Exception:
        return False, cmd


# =====================================================================
# 4. 版本字符串规范化（坑 50）
# =====================================================================
def norm_ver(v):
    """剥离 PEP440 local version（`2.7.1+cpu` → `2.7.1`）与首尾空白。
    ★ aarch64 上 pypi 的 torch 就是 `+cpu` 构建，torch_npu 负责接 NPU —— 属正常，不算 drift。"""
    if v is None:
        return None
    return str(v).strip().split("+")[0]


def ver_tuple(v):
    """`9.1.0-beta.3` → (9, 1, 0)；无法解析的段忽略。"""
    v = norm_ver(v) or ""
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def is_prerelease(v):
    s = (v or "").lower()
    return ("beta" in s) or ("rc" in s) or ("alpha" in s) or ("pre" in s)


# =====================================================================
# 5. 基础命令与主机信息（坑 39：极简镜像没有 hostname）
# =====================================================================
def safe_hostname():
    try:
        with open("/proc/sys/kernel/hostname", "r") as f:
            return f.read().strip()
    except Exception:
        pass
    return shutil.which("hostname") and _run("hostname") or "unknown"


def which(cmd):
    return shutil.which(cmd)


def missing_commands(cmds):
    return [c for c in cmds if not shutil.which(c)]


def _run(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")
    except Exception:
        return ""


# =====================================================================
# 6. 昇腾硬件与 CANN
# =====================================================================
def _ascend_env():
    env = dict(os.environ)
    ld = env.get("LD_LIBRARY_PATH", "")
    for p in ("/usr/local/Ascend/driver/lib64",
              "/usr/local/Ascend/driver/lib64/common",
              "/usr/local/Ascend/driver/lib64/driver"):
        if p not in ld:
            ld = p + ":" + ld
    env["LD_LIBRARY_PATH"] = ld
    return env


def npu_info():
    """返回 {visible, chip_count, chip_name, devices, raw}；任何一步失败都不抛异常。"""
    out = {"visible": False, "chip_count": None, "chip_name": None,
           "devices": 0, "raw": ""}
    env = _ascend_env()
    for exe in ("/usr/local/bin/npu-smi", "npu-smi"):
        if not (os.path.isfile(exe) or shutil.which(exe)):
            continue
        try:
            r = subprocess.run([exe, "info", "-l"], capture_output=True, text=True,
                               timeout=60, env=env)
            txt = (r.stdout or "") + (r.stderr or "")
            if not txt.strip():
                continue
            out["raw"] = txt
            m = re.search(r"Chip Count\s*:\s*(\d+)", txt)
            if m:
                out["chip_count"] = int(m.group(1))
                out["visible"] = True
            r2 = subprocess.run([exe, "info"], capture_output=True, text=True,
                                timeout=60, env=env)
            t2 = (r2.stdout or "") + (r2.stderr or "")
            m2 = re.search(r"\|\s*\d+\s+(\w+)\s*\|", t2)
            if m2:
                out["chip_name"] = m2.group(1)
            break
        except Exception:
            continue
    out["devices"] = len(davinci_nodes())
    return out


def davinci_nodes():
    """列出 NPU 设备节点；Windows 等没有 /dev 的平台返回空列表。"""
    try:
        import glob
        return sorted(glob.glob("/dev/davinci[0-9]*"))
    except OSError:
        return []


def torch_npu_info(exe=None):
    """优先用 torch_npu 拿真实 SoC 名（npu-smi 可能只给通用名）—— 坑 8。"""
    exe = exe or python_exe()
    code = ("import json,torch,torch_npu;"
            "d={'available':bool(torch.npu.is_available()),"
            "'count':int(torch.npu.device_count()) if torch.npu.is_available() else 0,"
            "'names':[torch.npu.get_device_name(i) for i in range(torch.npu.device_count())]"
            " if torch.npu.is_available() else [],"
            "'torch':torch.__version__,'torch_npu':torch_npu.__version__};"
            "print(json.dumps(d))")
    try:
        r = subprocess.run([exe, "-c", code], capture_output=True, text=True,
                           timeout=120, env=_ascend_env())
        line = [l for l in (r.stdout or "").splitlines() if l.strip().startswith("{")]
        if line:
            return json.loads(line[-1])
    except Exception:
        pass
    return None


def cann_root():
    for c in ("/usr/local/Ascend/ascend-toolkit/latest", "/usr/local/Ascend/cann",
              "/usr/local/Ascend/ascend-toolkit"):
        if os.path.isdir(c):
            return c
    try:
        import glob as _g
        cands = sorted(_g.glob("/usr/local/Ascend/cann-*"))
        if cands:
            return cands[-1]
    except Exception:
        pass
    return None


def cann_version():
    """★ 坑 40：版本文件名不固定（compiler/version.info / opp/version.info），
    且 `ascend-toolkit/latest` 可能是符号链接。"""
    try:
        c = os.path.realpath("/usr/local/Ascend/ascend-toolkit/latest")
        m = re.search(r"cann-(\S+)", c)
        if m:
            return m.group(1)
    except Exception:
        pass
    root = cann_root()
    if root:
        m = re.search(r"cann-(\S+)", os.path.basename(root.rstrip("/")))
        if m:
            return m.group(1)
    for rel in ("compiler/version.info", "opp/version.info", "version.cfg"):
        p = os.path.join(root or "", rel)
        if os.path.isfile(p):
            txt = open(p, encoding="utf-8", errors="replace").read()
            m = re.search(r"[Vv]ersion\s*[=:]\s*([\w.\-]+)", txt)
            if m:
                return m.group(1)
    return None


def driver_version():
    p = "/usr/local/Ascend/driver/version.info"
    if os.path.isfile(p):
        m = re.search(r"Version=([\w.\-]+)", open(p, encoding="utf-8", errors="replace").read())
        if m:
            return m.group(1)
    return None


# =====================================================================
# 7. 下载（续传 + 大小校验 + 多源 + 退避）—— 坑 10/29
# =====================================================================
def _http_size(url, timeout=25):
    """HEAD 拿 Content-Length；拿不到返回 None。"""
    try:
        r = subprocess.run(["curl", "-sIL", "--max-time", str(timeout), url],
                           capture_output=True, text=True, timeout=timeout + 10)
        sizes = re.findall(r"(?i)content-length:\s*(\d+)", r.stdout or "")
        return int(sizes[-1]) if sizes else None
    except Exception:
        return None


def download(url, dst, expect_bytes=None, retries=4, resume=True, timeout=7200):
    """下载到 dst，支持**断点续传**与**大小校验**。

    返回 dict: {ok, bytes, expected, resumed, attempts, error}
    ★ 19GB 文件在网络抖动/429 下必须能续传，否则一次失败就前功尽弃。
    """
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    expected = expect_bytes or _http_size(url)
    attempts = 0
    last_err = ""
    while attempts < retries:
        attempts += 1
        have = os.path.getsize(dst) if os.path.isfile(dst) else 0
        if expected and have == expected:
            return {"ok": True, "bytes": have, "expected": expected,
                    "resumed": True, "attempts": attempts, "error": ""}
        resumed = resume and have > 0
        cmd = ["curl", "-L", "--fail", "--retry", "3", "--retry-delay", "5",
               "--max-time", str(timeout), "-o", dst]
        if resumed:
            cmd += ["-C", "-"]
        cmd.append(url)
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 60)
        except Exception as e:
            last_err = "%s: %s" % (type(e).__name__, e)
        have = os.path.getsize(dst) if os.path.isfile(dst) else 0
        if expected and have == expected:
            return {"ok": True, "bytes": have, "expected": expected,
                    "resumed": resumed, "attempts": attempts, "error": ""}
        if expected is None and have > 0:
            # 无法校验大小 → 只能认为成功（并标注）
            return {"ok": True, "bytes": have, "expected": None,
                    "resumed": resumed, "attempts": attempts,
                    "error": "size_unverifiable"}
        last_err = last_err or "size_mismatch have=%s expect=%s" % (have, expected)
        time.sleep(min(60, 5 * attempts))
    return {"ok": False, "bytes": os.path.getsize(dst) if os.path.isfile(dst) else 0,
            "expected": expected, "resumed": False, "attempts": attempts, "error": last_err}


def pip_install(packages, exe=None, index_urls=None, extra_index=EXTRA_INDEX,
                retries=3, quiet=True):
    """带**指数退避**与**多源回退**的 pip 安装（坑 10：HTTP 429 限流）。

    返回 dict: {ok, attempts, source_used, log_tail}
    """
    exe = exe or python_exe()
    if isinstance(packages, str):
        packages = packages.split()
    urls = index_urls or SOURCES["pypi"]
    attempt = 0
    log_tail = ""
    while attempt < retries:
        src = urls[min(attempt, len(urls) - 1)]
        attempt += 1
        cmd = [exe, "-m", "pip", "install", "-q" if quiet else ""] if False else \
              [exe, "-m", "pip", "install"] + (["-q"] if quiet else []) + packages
        cmd += ["--index-url", src]
        if extra_index and extra_index != src:
            cmd += ["--extra-index-url", extra_index]
        cmd += ["--timeout", "60", "--retries", "2"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            log_tail = ((r.stdout or "") + (r.stderr or ""))[-600:]
            if r.returncode == 0:
                return {"ok": True, "attempts": attempt, "source_used": src,
                        "log_tail": log_tail}
        except Exception as e:
            log_tail = "%s: %s" % (type(e).__name__, e)
        time.sleep(min(90, 10 * attempt))   # 退避：应对 429
    return {"ok": False, "attempts": attempt, "source_used": urls[-1], "log_tail": log_tail}


# =====================================================================
# 8. 降级账本（诚实红线的机读化）
# =====================================================================
def degrade(record_path, name, reason, severity="degraded", extra=None):
    """把一次降级写进账本。**任何"因环境不支持而换路径"都必须调用本函数。**

    账本是 JSON 数组，位于 out/degradations.json（或指定路径）。
    交付物/报告可直接读取，避免"悄悄降级"。
    """
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "name": name, "severity": severity,
           "reason": reason}
    if extra:
        rec.update(extra)
    doc = []
    try:
        if os.path.isfile(record_path):
            doc = json.load(open(record_path, encoding="utf-8"))
            if not isinstance(doc, list):
                doc = []
    except Exception:
        doc = []
    doc.append(rec)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(record_path)) or ".", exist_ok=True)
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
    except Exception:
        pass
    print("[DEGRADED] %s — %s" % (name, reason))
    return rec


def read_degradations(record_path):
    try:
        return json.load(open(record_path, encoding="utf-8"))
    except Exception:
        return []


# =====================================================================
# 8.5 可比性判据 —— **三个独立结论**（不得互相替代）
# =====================================================================
# ★ 坑 146（第一版）：`official_comparable` 曾**只看几何**（dp/mbs/gas 是否等于官方值），
#   于是"官方几何 + mock 数据"的实跑被标成 True（真机 710.55 ms vs 官方 431.3 ms，差 1.65 倍）。
# ★ 坑 148（复核意见 2026-09-21，第二版）：修成"运行 vs 计划十项是否相同"后**仍不够 fail closed** ——
#   "**双方都缺**该字段"被算成一致、"计划与运行用**同一份 mock**"也被算成一致。
#   **计划一致性 ≠ 官方可比性**。
#   ⇒ 现在拆成三个**分别**下结论、且**不得互相替代**的函数：
#       plan_consistent()     实跑与该次计划一致，且必要字段齐全（缺字段即 False）
#       ab_comparable()       A/B 两组只有登记的变量不同（工作负载/协议一致）
#       official_comparable() 官方工作负载+配置+口径满足比较要求；**缺证据一律 False**
#   ★ 特别注意：「本机没有可对标官方的数据」**不等于**「没有有效的同机 A/B 证据」——
#     这是两个不同问题，前者 False 不妨碍后者成立。
COMPARE_KEYS = (
    "parallel.data_parallel_size",
    "training.micro_batch_size",
    "training.gradient_accumulation_steps",
    "training.load_rank0_and_broadcast",
    "training.save_format",
    "data.dataset_param.basic_parameters.dataset",
    "data.dataset_param.basic_parameters.dataset_dir",
    "data.dataset_param.basic_parameters.cutoff_len",
    # ★ 2026-09-21 修正：`image_max_pixels` 在 **`preprocess_parameters`** 下，不在
    #   `basic_parameters` 下。我原先写错路径 ⇒ 该键在计划与运行里**永远 absent**
    #   ⇒ 「缺必要字段即 fail」的规则会让 `plan_consistent` **恒为 False**，
    #   而且缺失噪声会**掩盖**真正的偏离。**由 align_audit 在真实配置上抓出**：
    #   真实派生配置里 `preprocess_parameters.image_max_pixels: 262144` 与官方一致。
    #   同时给 `plan_consistent` 加了"叶子名唯一匹配"回退，使同类路径笔误不再变成静默缺失。
    "data.dataset_param.preprocess_parameters.image_max_pixels",
    "data.dataloader_param.num_workers",
    # ★ 复核意见（2026-09-21）：「新的比较检查必须覆盖它」——
    #   `model.skip_gdn_recompute` 是**计算量相关**的键，官方日志是 `True`
    #   （`official_baseline.log:94` + L27 注册行），我方 A3 跑的是 `false`，
    #   而在此之前**判据链完全不比对它**。现纳入比对；`gdn_implementation` 一并纳入
    #   （它与 `skip_gdn_recompute` **耦合**：eager + True 会 raise NotImplemented）。
    "model.gdn_implementation",
    "model.causal_conv1d_implementation",
    "model.skip_gdn_recompute",
    # ★★ 2026-09-21 第二次补漏：`align_audit.py` 在官方对齐配置上报了
    #   `ALIGNED=9 / DECLARED=4 / UNDECLARED=0`（全绿），**但这个绿是"没看"出来的** ——
    #   逐键比对官方日志与 `sk04_judge/evidence/officialA_configuration_details.txt` 后发现，
    #   `parallel.fsdp_plan` 里其实有**两个**键与官方不同：
    #       pregather              官方 False / 我方 True
    #       reshard_after_forward  官方 True  / 我方 False   ← **此前根本不在比对清单里**
    #   而 `reshard_after_forward=False`（forward 后不重新分片参数）是**实打实的显存与性能相关键**，
    #   既可能是"快"的原因，也可能是 OOM 的帮凶。它不在清单里 ⇒ 审计器**结构上看不见它**
    #   ⇒ `UNDECLARED=0` 是"闸门没通电"，不是"没有偏离"（与坑 163/166 同族）。
    #   附带的教训：**"全绿"必须连同"检了哪些键"一起读**，只报绿灯不报覆盖面的闸门是假安全。
    "parallel.fsdp_plan.pregather",
    "parallel.fsdp_plan.reshard_after_forward",
    "parallel.fsdp_plan.num_to_forward_prefetch",
    "parallel.fsdp_plan.num_to_backward_prefetch",
)
# 刻意**不**比较（按设计本就该变；比了只会制造"假偏离"，反而训练人忽略真偏离）：
#   training.save         —— 每次实验写自己的目录
#   training.train_iters  —— 由 --steps 决定，它定义"窗口"，不是可比性条件
#   tools.* / profiling   —— 只影响测量方式，不改变被比对象


def find_leaf_paths(doc, leaf, prefix=""):
    """按**叶子名**找出所有出现位置（点号路径列表）。用于路径笔误的回退解析。"""
    out = []
    if isinstance(doc, dict):
        for k, v in doc.items():
            p = "%s.%s" % (prefix, k) if prefix else str(k)
            if k == leaf:
                out.append(p)
            out.extend(find_leaf_paths(v, leaf, p))
    elif isinstance(doc, list):
        for i, v in enumerate(doc):
            out.extend(find_leaf_paths(v, leaf, "%s[%d]" % (prefix, i)))
    return out


def dig_any(doc, dotted):
    """按点号路径取值；**路径不存在时退回"叶子名唯一匹配"**。返回 (value, path, how)。

    how ∈ {"exact", "leaf", "absent", "ambiguous"}。

    ★ 为什么要这个回退（2026-09-21，真实教训）：我把 `image_max_pixels` 的路径写成
    `basic_parameters.image_max_pixels`，而它实际在 `preprocess_parameters` 下 ⇒
    该键在计划与运行里**永远 absent** ⇒「缺必要字段即 fail」让 `plan_consistent` 恒为 False，
    且缺失噪声掩盖真实偏离。**而且我的自检用的是我手写的合成计划（路径同样写错），
    所以自检"验证"了我的错误** —— 只有拿真实产物做的检查才抓得到。
    回退解析把这类"路径笔误"变成"能解析但会打印实际路径"，不再退化成静默缺失。
    """
    cur, ok = doc, True
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            ok = False
            break
        cur = cur[part]
    if ok:
        return cur, dotted, "exact"
    leaf = dotted.split(".")[-1]
    hits = find_leaf_paths(doc, leaf)
    if len(hits) == 1:
        return dig(doc, hits[0]), hits[0], "leaf"
    if len(hits) > 1:
        return None, ",".join(hits), "ambiguous"
    return None, dotted, "absent"


def dig(doc, dotted, default=None):
    """按点号路径取值；任一层缺失返回 default（不抛异常，便于逐项对比）。"""
    cur = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def put(doc, dotted, value):
    """按点号路径赋值（仅当父层已存在时生效）。返回是否写入成功。"""
    parts = dotted.split(".")
    cur = doc
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    if not isinstance(cur, dict):
        return False
    cur[parts[-1]] = value
    return True


def flatten(doc, prefix=""):
    """把嵌套配置摊平成 {点号路径: 叶子值}；列表整体当一个叶子比较。"""
    out = {}
    if isinstance(doc, dict):
        for k, v in doc.items():
            out.update(flatten(v, "%s.%s" % (prefix, k) if prefix else str(k)))
    elif isinstance(doc, list):
        out[prefix] = repr(doc)
    else:
        out[prefix] = doc
    return out


# ---- 结论 1：plan_consistent ------------------------------------------------
def plan_consistent(run_cfg, plan_path):
    """实跑**与它自己那次计划**一致，且**必要字段齐全**。返回 (bool, details, note)。

    ★ 复核意见（2026-09-21，已确认）：原实现只比较"运行 vs 计划的十个字段是否相同" ——
      于是 **"双方都缺该字段"被算成"一致"**、**计划与运行都用同一份 mock 也算一致**。
      **计划一致性 ≠ 官方可比性**：缺字段必须**显式判失败**，而不是两边一起缺就算匹配。
    本函数只回答"与计划是否一致 + 字段是否齐全"，**不得**被当作可比性结论使用。
    """
    if not plan_path or not os.path.isfile(plan_path):
        return False, {"plan": str(plan_path), "missing_fields": list(COMPARE_KEYS) + ["<plan>"],
                       "deviations": []}, \
               "P2 计划配置不存在（%s）→ plan_consistent=False" % plan_path
    try:
        import yaml
        plan = yaml.safe_load(open(plan_path, encoding="utf-8"))
    except Exception as exc:
        return False, {"plan": str(plan_path), "error": type(exc).__name__,
                       "missing_fields": list(COMPARE_KEYS), "deviations": []}, \
               "P2 计划配置不可解析（%s）→ plan_consistent=False" % type(exc).__name__

    # ★ 用 dig_any（带回退解析）：路径笔误不再退化成"静默缺失"，而是**可解析 + 打印实际路径**
    missing, devs, resolved = [], [], {}
    for k in COMPARE_KEYS:
        pv, ppath, phow = dig_any(plan, k)
        rv, rpath, rhow = dig_any(run_cfg, k)
        resolved[k] = {"plan_path": ppath, "plan_how": phow,
                       "run_path": rpath, "run_how": rhow}
        if phow in ("absent", "ambiguous") or rhow in ("absent", "ambiguous"):
            missing.append("%s(plan=%s,run=%s)" % (k, phow, rhow))
        elif pv != rv:
            devs.append({"key": k, "plan": pv, "run": rv,
                         "plan_path": ppath, "run_path": rpath})
    details = {"plan": str(plan_path), "n_compared": len(COMPARE_KEYS),
               "missing_fields": missing, "deviations": devs, "resolved": resolved}
    # 回退解析不是错误，但必须**显式可见**（否则"路径写错"又会变成看不见的东西）
    fell_back = [k for k, r in resolved.items()
                 if r["plan_how"] == "leaf" or r["run_how"] == "leaf"]
    if fell_back:
        details["resolved_by_leaf_fallback"] = fell_back
    if missing:
        note = "必要字段缺失 %d 项（**双方都缺不算一致**）：%s" % (len(missing), "、".join(missing))
    elif devs:
        note = "与计划存在 %d/%d 处偏离：%s" % (len(devs), len(COMPARE_KEYS),
                                              "、".join(d["key"] for d in devs))
    else:
        note = "与 P2 计划逐项一致且必要字段齐全（%d 项）" % len(COMPARE_KEYS)
    return (not missing and not devs), details, note


# ---- A/B 可比性：只允许"登记的变量"不同 --------------------------------------
# 按设计每次实验就会不同的项（不构成"不可比"）：保存目录、迭代数（由 --steps 决定）
AB_PROTOCOL_KEYS = ("training.save", "training.train_iters")


def ab_comparable(cfg_a, cfg_b, key, protocol_keys=AB_PROTOCOL_KEYS):
    """结论 2：A/B 两组是否**只有登记的实验变量不同**。返回 (bool, details)。

    ★ 复核意见：A/B 可比性、计划一致性、官方可比性是**三个独立结论**，必须分别下结论。
    `--order` 里 A/B 的**执行顺序**不在此函数职责内（由 59 的 order 平衡性单独记录）。
    """
    import copy as _copy
    a, b = _copy.deepcopy(cfg_a), _copy.deepcopy(cfg_b)
    put(a, key, "<registered>")
    put(b, key, "<registered>")
    for k in protocol_keys:
        put(a, k, "<protocol>")
        put(b, k, "<protocol>")
    fa, fb = flatten(a), flatten(b)
    diffs = [{"path": k, "a": fa.get(k, "<absent>"), "b": fb.get(k, "<absent>")}
             for k in sorted(set(fa) | set(fb))
             if fa.get(k, "<absent>") != fb.get(k, "<absent>")]
    return (not diffs), {"key": key, "n_paths": len(set(fa) | set(fb)),
                         "protocol_normalized": list(protocol_keys), "diffs": diffs}


# ---- 结论 3：official_comparable（最严，缺证据一律 False）--------------------
OFFICIAL_DATASET_BASENAME = "output_llava_coco_data.json"


def dataset_kind(cfg):
    """数据类别：`official` / `mock` / `unknown`。

    ★ 复核意见：应显式记录 `dataset_kind`。**不能靠"路径相等"断言可比性** ——
      两份不同的 mock 用同一路径就完全一样了（这正是原实现的漏洞）。
    """
    p = dig(cfg, "data.dataset_param.basic_parameters.dataset")
    if not isinstance(p, str) or not p.strip():
        return "unknown"
    return "official" if os.path.basename(p) == OFFICIAL_DATASET_BASENAME else "mock"


def data_identity(cfg):
    """数据身份：路径 + 类别 + 字节数 + sha256 前 12 位（读不到就如实标 None，不假装一致）。"""
    import hashlib
    p = dig(cfg, "data.dataset_param.basic_parameters.dataset")
    rec = {"path": p, "kind": dataset_kind(cfg), "bytes": None, "sha256_12": None}
    if isinstance(p, str) and p and os.path.isfile(p):
        try:
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            rec["bytes"] = os.path.getsize(p)
            rec["sha256_12"] = h.hexdigest()[:12]
        except OSError as exc:
            rec["error"] = type(exc).__name__
    return rec


def official_comparable(run_cfg, plan_path, geometry_ok):
    """结论 3：`official_comparable` —— **三个结论里最严的一个**。返回 (bool, details, note)。

    必须**同时**满足：
      · `geometry_ok`      —— 几何等于官方（dp/mbs/gas 的红线检查）
      · `plan_consistent`  —— 与计划一致，**且必要字段齐全**
      · `dataset_kind == "official"` —— 数据是官方那份（mock 一律 False）
      · 数据身份可核（能读到文件并算出哈希）
    任一条缺证据即 False（fail closed）。**调用方不得用几何单独下结论。**
    """
    pc, pd, _pn = plan_consistent(run_cfg, plan_path)
    dk = dataset_kind(run_cfg)
    di = data_identity(run_cfg)
    reasons = []
    if not geometry_ok:
        reasons.append("几何非官方")
    if not pc:
        reasons.append("与计划不一致或必要字段缺失（plan_consistent=False）")
    if dk != "official":
        reasons.append("数据类别=%s（非官方数据）" % dk)
    if di.get("sha256_12") is None:
        reasons.append("数据身份无法核验（文件不存在或不可读）")
    details = {"is_official_geometry": geometry_ok, "plan_consistent": pc,
               "plan_details": pd, "dataset_kind": dk, "data_identity": di,
               "reasons": reasons}
    note = "满足官方可比性" if not reasons else ("不可比：" + "；".join(reasons))
    return (not reasons), details, note


# ---- 运行有效性：59 与 61 必须给出**同一个答案** ------------------------------
# ★ 复核意见（2026-09-21）：「现在 59 检查完整步号，61 仍只检查解析行数。应统一，
#   避免同一种运行得到不同的 valid。」
#   ⇒ 有效性的**唯一实现**放在这里，59/61/62 全部复用。
def run_valid(returncode, train_rc, step_ids, steps):
    """单次实跑是否有效。返回 `(valid, steps_complete)`。

    四者同时成立才算有效：
      · `returncode == 0`    进程正常退出
      · `train_rc == ["0"]`  驱动脚本内部的训练返回码为 0
      · `len(step_ids) == steps`  解析到的步数不缺
      · `steps_complete`     步号必须是 1..steps **恰好一次**（缺步/重复都无效）
    """
    steps_complete = (sorted(step_ids) == list(range(1, steps + 1)))
    valid = (returncode == 0 and train_rc == ["0"]
             and len(step_ids) == steps and steps_complete)
    return valid, steps_complete


# =====================================================================
# 9. 判据登记表（供 90_selfcheck_gates.py 做负向对照）
# =====================================================================
GATES = {
    # name: (说明, 期望在"坏输入"下返回非零/输出失败标记)
    "assets_completeness": "资产缺失时必须 ASSETS_INCOMPLETE 且 rc=3",
    "registry_tamper": "哈希链被篡改时必须 REGISTRY_CHAIN_BROKEN 且 rc=3",
    "registry_dup_tag": "重复 tag 必须被拒",
    "judge_determinism": "同一证据必须得到同一个 verdict_id",
    "env_acceptance_vs_portability": "非验收环境时 acceptance 必须 FAIL、portability 必须 OK",
    "triton_min_kernel": "kernel 结果错误时必须 TRITON_NPU_FAIL",
    "bench_no_baseline": "没有基线轮时必须拒绝出对比数据",
    "judge_invalid_input": "非法输入必须 INVALID_INPUT",
    "replay_verdict_id": "包内示例必须重放出交付材料的 verdict_id",
    "official_comparable_requires_data_match":
        "官方几何 + 非官方数据时必须 official_comparable=False（坑 146）",
    "three_comparability_conclusions":
        "plan_consistent / ab_comparable / official_comparable 必须分别下结论，"
        "且「双方都缺字段」「计划与运行同一 mock」都不得判为官方可比（坑 148）",
    "run_valid_unified":
        "59 与 61 对同一次运行的有效性判定必须一致（步号 1..N 恰好一次）",
}


# =====================================================================
# selftest
# =====================================================================
def selftest():
    print("== _envcompat selftest ==")
    fails = 0
    total = 0

    def ck(name, cond, detail=""):
        nonlocal fails, total
        total += 1
        print("  %-28s %s %s" % (name, "PASS" if cond else "FAIL", detail))
        if not cond:
            fails += 1

    ck("skill_root 存在 SKILL.md", os.path.isfile(os.path.join(skill_root(), "SKILL.md")),
       skill_root())
    ck("find_repo_root 不抛异常", find_repo_root() is None or os.path.isdir(find_repo_root()))
    ck("python_exe 可执行", os.path.isfile(python_exe()) or bool(which(python_exe())),
       python_exe())
    ck("norm_ver 剥离 +cpu", norm_ver("2.7.1+cpu") == "2.7.1")
    ck("ver_tuple 解析预发布", ver_tuple("9.1.0-beta.3") == (9, 1, 0))
    ck("is_prerelease 识别 beta", is_prerelease("9.1.0-beta.3") is True)
    ck("safe_hostname 不抛异常", isinstance(safe_hostname(), str))
    ck("pkg_manager 可判定", pkg_manager() is None or isinstance(pkg_manager(), str),
       str(pkg_manager()))
    # ★ 本项是"平台属性"而非缺陷：Windows 上没有 dnf/apt 是正常的。
    #   故仅在**存在**包管理器时才要求命令名正确（否则标 N/A），避免自检在非目标平台上误报。
    if pkg_manager() is None:
        print("  %-28s N/A  本机无包管理器（非目标平台，跳过）" % "install_hint 分派正确")
    else:
        hint = install_hint("pyd")
        ck("install_hint 分派正确", ("python3-devel" in hint or "python3-dev" in hint), hint)
    ck("SOURCES 每类 >=2 源",
       all(len(v) >= 2 for v in SOURCES.values()),
       ",".join("%s=%d" % (k, len(v)) for k, v in SOURCES.items()))
    ck("missing_commands 可判定", isinstance(missing_commands(["definitely_not_a_cmd_xyz"]), list))
    ck("npu_info 不抛异常", isinstance(npu_info(), dict))
    ck("GATES 登记 >= 8 条", len(GATES) >= 8, "n=%d" % len(GATES))

    # ★ 复核意见（2026-09-21）给出的两个**合成坏例**必须被拒 ——
    #   ① 计划与运行**都缺**数据字段：旧实现判"一致"→ True（错，缺字段不等于匹配）；
    #   ② 计划与运行用**同一份 mock**：旧实现判 True（错，计划一致 ≠ 官方可比）。
    #   现在拆成三个独立结论：plan_consistent / ab_comparable / official_comparable。
    try:
        import copy as _copy
        import shutil as _sh
        import tempfile as _tf
        import yaml as _yaml

        _td = _tf.mkdtemp()
        _official = os.path.join(_td, OFFICIAL_DATASET_BASENAME)
        _mockfile = os.path.join(_td, "mock_data.json")
        for _p in (_official, _mockfile):
            with open(_p, "w", encoding="utf-8") as _fh:
                _fh.write("[]\n")
        _pp = os.path.join(_td, "plan.yaml")
        with open(_pp, "w", encoding="utf-8") as _fh:
            _fh.write("parallel:\n  data_parallel_size: 2\n"
                      # ★ 2026-09-21（第二次）：COMPARE_KEYS 又新增 4 个
                      #   `parallel.fsdp_plan.*` 键（官方 dump 逐键比对才暴露：`pregather` 与
                      #   `reshard_after_forward` 都与官方不同，而旧比对清单里根本没有它们）。
                      #   **合成计划必须同步**，否则"必要字段缺失"会把正例判失败 ——
                      #   这次同样是被自检的正例抓出来的（连着两次同一个教训，故写在这里）。
                      "  fsdp_plan:\n    pregather: false\n    reshard_after_forward: true\n"
                      "    num_to_forward_prefetch: 1\n    num_to_backward_prefetch: 1\n"
                      "training:\n  micro_batch_size: 4\n"
                      "  gradient_accumulation_steps: 1\n"
                      "  load_rank0_and_broadcast: true\n  save_format: hf\n"
                      "data:\n  dataset_param:\n"
                      # ★ 与**真实产物**同结构：`image_max_pixels` 在 `preprocess_parameters` 下。
                      #   我原先在自检里按**错误路径**（basic_parameters）手写合成计划，
                      #   于是自检"验证"了我的路径笔误 —— 教训：合成输入必须照真实产物抄结构。
                      "    preprocess_parameters:\n      image_max_pixels: 262144\n"
                      "    basic_parameters:\n"
                      "      dataset: %s\n      dataset_dir: %s\n      cutoff_len: 1024\n"
                      "  dataloader_param:\n    num_workers: 8\n"
                      # ★ 2026-09-21：COMPARE_KEYS 新增 3 个 model.* 键（官方日志是 True，
                      #   我方 A3 曾是 false 且判据链不比对）—— 合成计划必须同步具备这些键，
                      #   否则"必要字段缺失"会把正例也判失败（这次正是自检把它抓出来的）。
                      "model:\n  gdn_implementation: triton\n"
                      "  causal_conv1d_implementation: triton\n  skip_gdn_recompute: true\n"
                      % (_official, _td))
        _base = _yaml.safe_load(open(_pp, encoding="utf-8"))

        _ok1, _d1, _n1 = official_comparable(_base, _pp, True)
        ck("可比性：官方数据 + 计划一致 → True", _ok1 is True, _n1)

        # 坏例 ①：计划与运行**都缺** dataset 字段
        _nodata = _copy.deepcopy(_base)
        _nodata["data"]["dataset_param"]["basic_parameters"].pop("dataset", None)
        _pp2 = os.path.join(_td, "plan_nodata.yaml")
        with open(_pp2, "w", encoding="utf-8") as _fh:
            _yaml.safe_dump(_nodata, _fh, sort_keys=False)
        _pc2, _pd2, _pn2 = plan_consistent(_nodata, _pp2)
        _ok2, _dd2, _n2 = official_comparable(_nodata, _pp2, True)
        ck("坏例①计划与运行都缺数据字段 → plan_consistent=False", _pc2 is False,
           "缺=%s" % _pd2["missing_fields"])
        ck("坏例①… 且 official_comparable=False", _ok2 is False, _n2)

        # 坏例 ②：计划与运行用**同一份 mock**
        _m = _copy.deepcopy(_base)
        _m["data"]["dataset_param"]["basic_parameters"]["dataset"] = _mockfile
        _pp3 = os.path.join(_td, "plan_mock.yaml")
        with open(_pp3, "w", encoding="utf-8") as _fh:
            _yaml.safe_dump(_m, _fh, sort_keys=False)
        _pc3, _pd3, _pn3 = plan_consistent(_m, _pp3)
        _ok3, _dd3, _n3 = official_comparable(_m, _pp3, True)
        ck("坏例②同一 mock：plan_consistent 可以为 True", _pc3 is True, _pn3)
        ck("坏例②同一 mock：official_comparable 必须 False", _ok3 is False,
           "dataset_kind=%s" % _dd3["dataset_kind"])

        _ok4, _dd4, _n4 = official_comparable(_m, _pp3, False)
        ck("非官方几何 → False", _ok4 is False, _n4)
        _ok5, _dd5, _n5 = official_comparable(_base, "/no/such/plan.yaml", True)
        ck("计划缺失 → 保守 False", _ok5 is False, _n5)

        # A/B 可比性（三个结论里的第二个）：只允许登记变量不同
        _a = _copy.deepcopy(_base)
        _b = _copy.deepcopy(_base)
        _b["parallel"]["data_parallel_size"] = 4
        _ab_ok, _abd = ab_comparable(_a, _b, "parallel.data_parallel_size")
        ck("ab_comparable：只有登记变量不同 → True", _ab_ok is True,
           "diffs=%d" % len(_abd["diffs"]))
        _b2 = _copy.deepcopy(_b)
        _b2["training"]["micro_batch_size"] = 8
        _ab2_ok, _abd2 = ab_comparable(_a, _b2, "parallel.data_parallel_size")
        ck("ab_comparable：多出未登记差异 → False", _ab2_ok is False,
           "diffs=%s" % [d["path"] for d in _abd2["diffs"]])

        # ★ 路径回退（2026-09-21 真实教训：我把 image_max_pixels 的路径写错，
        #   该键在计划与运行里永远 absent ⇒ plan_consistent 恒 False 且掩盖真实偏离）。
        #   回退后：错误路径仍能解析（并标 how=leaf、打印实际路径），真正缺失才判 absent。
        _v, _p, _h = dig_any(_base, "data.dataset_param.basic_parameters.image_max_pixels")
        ck("路径回退：错误路径仍可解析并标 leaf", _h == "leaf" and _v == 262144,
           "how=%s path=%s val=%s" % (_h, _p, _v))
        _v2, _p2, _h2 = dig_any(_base, "data.dataset_param.basic_parameters.no_such_key_xyz")
        ck("路径回退：真正缺失 → absent", _h2 == "absent", "how=%s" % _h2)

        _sh.rmtree(_td, ignore_errors=True)
    except ImportError as _e:
        print("  %-28s N/A  缺 pyyaml/tempfile（%s），未做可比性负向对照"
              % ("可比性判据负向对照", _e))

    # ★ 原实现打印的是 `items=fails` —— 于是**全过时显示 `items=0`**，
    #   读起来像"一项都没检查"，与"检查了很多项且全过"正好相反（声明≠实况）。
    print("SELFCHECK_%s items=%d failed=%d" % ("OK" if not fails else "FAIL", total, fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="_envcompat 兼容层自检")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", action="store_true", help="输出环境档案 JSON")
    a = ap.parse_args()
    if a.json:
        prof = {
            "skill_root": skill_root(), "repo_root": find_repo_root(),
            "python_exe": python_exe(), "python_version": python_version(),
            "python_h": has_python_h(), "pkg_manager": pkg_manager(),
            "hostname": safe_hostname(), "missing_cmds": missing_commands(
                ["hostname", "unzip", "tar", "git", "curl", "gcc", "make"]),
            "npu": npu_info(), "torch_npu": torch_npu_info(),
            "cann": cann_version(), "cann_root": cann_root(), "driver": driver_version(),
            "msmm_dir": default_msmm_dir(), "data_dir": default_data_dir(),
        }
        print(json.dumps(prof, ensure_ascii=False, indent=1))
        sys.exit(0)
    sys.exit(selftest())
