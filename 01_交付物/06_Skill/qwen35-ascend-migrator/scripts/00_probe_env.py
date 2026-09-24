#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_probe_env.py — P0 环境探测（决定整条迁移路径）

探测维度：
  硬件：NPU 可见性 / die(chip) 数 / 芯片型号 / HBM 容量 / 设备节点
  软件：CANN / driver / torch / torch_npu / triton-ascend / transformers / MindSpeed-MM
  前置：python3-dev（triton 驱动编译必需）/ build-essential
  资源：磁盘可用空间 / 网络可达性（modelscope / obs / gitcode）
  并行：world_size / dp / 是否具备官方几何所需 die 数

输出：
  out/probe/env.json   — 机器可读档案（供后续阶段决策）
  控制台人读摘要

设计原则（冗余）：任一探测项失败不影响其它项，记为 unknown 并继续；
                 最终给出 path 判定与 degraded 原因列表，绝不因探测失败而中断。
退出码：0=探测完成（无论环境是否完备）；2=致命错误（无法写文件）
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------- 工具

def sh(cmd, timeout=30):
    """执行 shell 命令，返回 (stdout+stderr) 文本；失败返回空串。"""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")
    except Exception:
        return ""


# 明显的命令不可用/错误输出（跨平台健壮性：Windows/精简镜像下 shell 会回错误文本，
# 不能把它当成有效探测结果）
_ERR_MARKERS = (
    "cannot find the path", "is not recognized", "command not found", "No such file",
    "not found", "Permission denied", "拒绝访问", "系统找不到",
)


def is_valid_output(text):
    """判断命令输出是否为有效结果（而非错误提示）。"""
    if not text or not text.strip():
        return False
    low = text.strip().lower()
    return not any(m.lower() in low for m in _ERR_MARKERS)


def sh_valid(cmd, timeout=30):
    """执行命令，仅在输出有效时返回文本，否则返回空串。"""
    out = sh(cmd, timeout=timeout)
    return out if is_valid_output(out) else ""


def first_match(text, pattern, group=1):
    m = re.search(pattern, text)
    return m.group(group).strip() if m else None


def try_import_version(module):
    """尝试导入模块并取 __version__；失败返回 None（不抛异常）。"""
    try:
        mod = __import__(module)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return None


def file_exists(p):
    try:
        return os.path.exists(p)
    except Exception:
        return False


def du_free_gb(path="/"):
    try:
        st = os.statvfs(path)
        return round(st.f_bavail * st.f_frsize / 1e9, 1)
    except Exception:
        out = sh("df -BG %s 2>/dev/null | tail -1" % path)
        m = re.search(r"(\d+)G", out)
        return int(m.group(1)) if m else None


# ---------------------------------------------------------------- 探测项

def probe_hardware():
    hw = {}
    # npu-smi（需 CANN lib 路径，先补 LD）
    env_lib = ("export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:"
               "/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:"
               "/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH; ")
    out = sh(env_lib + "npu-smi info -l 2>&1 | head -20")
    hw["npu_smi_raw_ok"] = bool(re.search(r"(NPU ID|Chip Count|Total Count)", out))
    hw["total_count"] = first_match(out, r"Total Count\s*:\s*(\d+)")
    hw["npu_id"] = first_match(out, r"NPU ID\s*:\s*(\d+)")
    hw["chip_count"] = first_match(out, r"Chip Count\s*:\s*(\d+)")

    # 芯片型号（board 查询）
    chip_name = None
    npu_id = hw.get("npu_id")
    if npu_id is not None:
        for c in range(int(hw.get("chip_count") or 1)):
            b = sh(env_lib + "npu-smi info -t board -i %s -c %d 2>&1 | head -12" % (npu_id, c))
            nm = first_match(b, r"NPU Name\s*:\s*(\S+)")
            cn = first_match(b, r"Chip Name\s*:\s*(\S+)")
            if nm:
                chip_name = nm
            if cn and not hw.get("chip_hw_name"):
                hw["chip_hw_name"] = cn
            bid = first_match(b, r"Board ID\s*:\s*(\S+)")
            if bid and not hw.get("board_id"):
                hw["board_id"] = bid
    hw["chip_model"] = chip_name  # 如 9382 / 910B4

    # 设备节点（只保留真实 /dev/davinci* 路径）
    devs = sh("ls /dev/davinci[0-9]* 2>/dev/null | tr '\\n' ' '")
    hw["davinci_nodes"] = [d for d in devs.split() if d.startswith("/dev/davinci")]

    # 驱动
    drv = sh("cat /usr/local/Ascend/driver/version.info 2>/dev/null | head -3")
    hw["driver_version"] = first_match(drv, r"Version=([\d.]+)")

    # torch_npu 视角（需 set_env）
    tn = sh("source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null; "
            "python3 -c \"import torch,torch_npu;print('DEVCOUNT',torch.npu.device_count());"
            "print('DEVNAME',torch.npu.get_device_name(0))\" 2>&1 | tail -3")
    hw["torch_npu_device_count"] = first_match(tn, r"DEVCOUNT\s+(\d+)")
    hw["torch_npu_device_name"] = first_match(tn, r"DEVNAME\s+(\S+)")

    # HBM 容量（每 chip）
    # ★ 坑 103：原实现只试 `npu-smi info -t memory -i <id> -c 0` 一条路径 ——
    #   真机上它没能给出可解析输出 → `hbm_mb_per_chip = null` →
    #   **档位选择掉到最差档（910b4_low_mem, world_size=1）**，
    #   于是 P5 用了错误几何（GBS=4 而非 8）。**单一路径的探测等于把环境适配押在一个命令上。**
    #   现改为**三路回退**（可靠性递增）：
    #     ① npu-smi info -t memory（原路径，多模式匹配）
    #     ② **npu-smi info 主表的 `used / total` 列**（实测该表稳定输出 `3132 / 65536`）
    #     ③ **torch_npu `mem_get_info()`**（最可靠，完全不依赖文本解析）
    hbm_mb = None
    mem = sh(env_lib + "npu-smi info -t memory -i %s -c 0 2>&1 | head -12" % (npu_id or 0))
    m = re.search(r"(?:HBM\s*)?Total\s*:\s*(\d+)", mem) or re.search(r"(\d{4,6})\s*$", mem.strip())
    if m:
        hbm_mb = m.group(1)

    if not (hbm_mb and hbm_mb.isdigit()):
        # 路径②：主表里形如 "3132 / 65536" 的列，取所有 chip 的最大 total
        raw = hw.get("npu_smi_raw") or hw.get("npu_smi_info_raw") or ""
        if not raw:
            raw = sh(env_lib + "npu-smi info 2>&1")
        tots = [int(x) for x in re.findall(r"\d+\s*/\s*(\d{4,6})", raw)]
        if tots:
            hbm_mb = str(max(tots))
            hw["hbm_source"] = "npu_smi_table"

    if not (hbm_mb and str(hbm_mb).isdigit()):
        # 路径③：torch_npu（字节 → MB），最可靠
        tnj = sh("source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null; "
                 "python3 -c \"import torch,torch_npu;f,t=torch.npu.mem_get_info(0);"
                 "print('HBMB', int(t//1024//1024))\" 2>&1 | tail -1")
        mm = first_match(tnj, r"HBMB\s+(\d+)")
        if mm and mm.isdigit():
            hbm_mb = mm
            hw["hbm_source"] = "torch_npu_mem_get_info"

    hw["hbm_mb_per_chip"] = int(hbm_mb) if (hbm_mb and str(hbm_mb).isdigit()) else None
    return hw


def probe_software():
    sw = {}
    sw["python"] = first_match(sh_valid("python3 --version"), r"Python\s+([\d.]+)")
    sw["pip"] = first_match(sh_valid("python3 -m pip --version 2>/dev/null"), r"pip\s+([\d.]+)")
    # CANN
    cann = sh_valid("ls -d /usr/local/Ascend/cann-* /usr/local/Ascend/ascend-toolkit/latest 2>/dev/null | head -3")
    sw["cann_dirs"] = cann.split() if cann.strip() else []
    sw["cann_version"] = first_match(cann, r"cann-([\d.]+)")
    if not sw["cann_version"]:
        vcfg = sh_valid("cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null | head -5")
        sw["cann_version"] = first_match(vcfg, r"version\s*=\s*([\d.]+)")
    # 框架栈（需 CANN env 才能 import torch_npu）
    env = ("source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null; ")
    for name, mod in (("torch", "torch"), ("torch_npu", "torch_npu"),
                      ("transformers", "transformers"), ("triton", "triton"),
                      ("numpy", "numpy"), ("yaml", "yaml")):
        v = first_match(sh(env + "python3 -c \"import %s;print('V',getattr(%s,'__version__','unknown'))\" "
                               "2>&1 | tail -1" % (mod, mod)), r"V\s+(\S+)")
        sw[name] = v
    # triton 版本需要**三路**采集（★ 坑 123）：实测新机器上这三个数**互不相同** ——
    #   `import triton; triton.__version__` = 3.2.0
    #   dist `triton_ascend`               = 3.2.2（与 A3 官方配对一致 → 版本本身没问题）
    #   dist `triton`（上游）              = 3.5.0（★ 与 triton_ascend **共存**，属安装混装）
    #   旧实现只取 `triton.__version__` 且另有一路 `import triton_ascend`（模块名不存在 → 恒失败），
    #   于是 env.json 里 `triton_ascend: null` → **做性能归因时拿不到这个关键版本**。
    #   只报一个数会掩盖混装；因此**三个数都记**，并把不一致显式标出来。
    _tri = sh(env + "python3 -c \"import importlib.metadata as m\n"
                    "out=[]\n"
                    "for d in ('triton-ascend','triton','torch-npu','torch'):\n"
                    "    try: out.append(d+'='+m.version(d))\n"
                    "    except Exception: out.append(d+'=ABSENT')\n"
                    "import triton\n"
                    "out.append('triton.__version__='+str(getattr(triton,'__version__','?')))\n"
                    "print('TRIV|'+'|'.join(out))\" 2>&1 | tail -1")
    _tv = first_match(_tri, r"TRIV\|(.+)")
    sw["triton_versions_raw"] = _tv
    if _tv:
        for _part in _tv.split("|"):
            if "=" in _part:
                _k, _v = _part.split("=", 1)
                sw["ver_" + _k.replace("-", "_").replace(".", "_")] = _v
        _ok, _bad = triton_version_verdict(sw.get("ver_triton___version__"),
                                           sw.get("ver_triton_ascend"),
                                           sw.get("ver_triton"))
        sw["triton_version_consistent"] = _ok
        if _bad:
            sw["triton_version_note"] = (
                "★ triton 环境异常：" + "；".join(_bad) +
                " —— 性能对比时必须同时记录这三个数，并考虑干净重装（先记录当前版本以便回退）")
    # 兼容旧键（下游/文档引用过 sw["triton"]），保持向后一致
    if sw.get("ver_triton___version__"):
        sw["triton"] = sw["ver_triton___version__"]
    # triton Ascend backend 是否真正可用（关键！决定算子后端）
    tb = sh(env + "python3 -c \"import triton;from triton.backends import ascend;print('ASCEND_OK')\" "
                  "2>&1 | tail -2")
    sw["triton_ascend_backend"] = ("ok" if "ASCEND_OK" in tb
                                   else ("unavailable" if tb.strip() else "unknown"))
    sw["triton_backend_err"] = tb.strip()[-300:] if "ASCEND_OK" not in tb else ""
    # 最小 triton kernel 实测（可选，慢；仅当 backend 声称 ok 时）
    # ★ 坑 46：这里曾内嵌 `triton.program_id(0)` —— 该属性在 JIT 命名空间不存在，
    #   会抛 AttributeError → 判据**假阴性** → 误判 triton 不可用 → 降级丢性能。
    #   现统一改为调用唯一权威判据 scripts/00b_triton_min_kernel.py（规范写法 tl.program_id）。
    if sw["triton_ascend_backend"] == "ok":
        _probe = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "00b_triton_min_kernel.py")
        if os.path.isfile(_probe):
            r = sh(env + "python3 %s 2>&1 | tail -2" % _probe, timeout=300)
        else:
            _min = ("import torch, torch_npu, triton\n"
                    "import triton.language as tl\n"
                    "@triton.jit\n"
                    "def _k(X, Y, N, BLOCK: tl.constexpr):\n"
                    "    pid = tl.program_id(0)\n"
                    "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
                    "    m = offs < N\n"
                    "    tl.store(Y + offs, tl.load(X + offs, mask=m, other=0.0) * 2.0, mask=m)\n"
                    "n = 1024\n"
                    "x = torch.randn(n).npu(); y = torch.empty_like(x)\n"
                    "_k[(triton.cdiv(n, 256),)](x, y, n, BLOCK=256)\n"
                    "print('TRITON_NPU_OK' if torch.allclose(y.cpu(), x.cpu()*2, atol=1e-3)"
                    " else 'TRITON_MISMATCH')\n")
            with open("/tmp/_triton_min_probe.py", "w") as f:
                f.write(_min)
            r = sh(env + "python3 /tmp/_triton_min_probe.py 2>&1 | tail -2", timeout=300)
        sw["triton_npu_kernel"] = ("ok" if "TRITON_NPU_OK" in r else "failed")
        sw["triton_npu_kernel_err"] = "" if "TRITON_NPU_OK" in r else r.strip()[-300:]
    else:
        sw["triton_npu_kernel"] = "skipped"
    # MindSpeed-MM
    for p in ("/root/MindSpeed-MM", "./MindSpeed-MM"):
        if file_exists(p):
            sw["mindspeed_mm_dir"] = os.path.abspath(p)
            tag = sh("git -C %s describe --tags 2>/dev/null" % p).strip()
            sw["mindspeed_mm_tag"] = tag or None
            break
    sw["mindspeed"] = try_import_version("mindspeed")
    return sw


def probe_prereq():
    pre = {}
    # python3-dev（triton 驱动编译必需）：必须是真实头文件路径
    inc = sh("ls /usr/include/python3*/Python.h /usr/local/include/python3*/Python.h 2>/dev/null | head -1")
    inc = inc.strip() if inc.strip().endswith("Python.h") else ""
    pre["python_dev_header"] = inc or None
    pre["python3_dev_ok"] = bool(inc)
    # build-essential（gcc/g++ 真实路径）
    gcc = sh("which gcc g++ 2>/dev/null | head -2").strip()
    pre["build_essential_ok"] = bool(re.search(r"/g(cc|\+\+)$", gcc, re.M))
    # apt 可用
    pre["apt_ok"] = bool(re.search(r"/apt-get$", sh("which apt-get 2>/dev/null").strip(), re.M))
    return pre


def probe_resources(data_dir="/root"):
    res = {}
    res["disk_free_gb_root"] = du_free_gb("/")
    if file_exists(data_dir):
        res["disk_free_gb_data"] = du_free_gb(data_dir)
    # 网络（以是否返回 HTTP 响应行为准）
    def http_ok(url):
        out = sh("timeout 12 curl -sI %s 2>&1 | head -1" % url)
        return "HTTP" in out.upper()

    res["net_modelscope"] = http_ok("https://modelscope.cn")
    res["net_obs"] = http_ok("https://public-download.obs.cn-east-2.myhuaweicloud.com")
    res["net_gitcode"] = http_ok("https://gitcode.com")
    return res


# ---------------------------------------------------------------- 路径判定（冗余核心）

def decide_path(hw, sw, pre):
    """返回 (path, reasons, capabilities)。任一路径都能继续，只是能力不同。"""
    reasons = []
    cap = {}

    die = hw.get("chip_count")
    die = int(die) if (die and str(die).isdigit()) else None
    if die is None and hw.get("torch_npu_device_count"):
        die = int(hw["torch_npu_device_count"])

    npu_ok = bool(die and die >= 1)
    triton_ok = (sw.get("triton_ascend_backend") == "ok" and sw.get("triton_npu_kernel") == "ok")

    cap["can_analyze"] = True            # P1 永远可做（静态）
    cap["can_plan"] = True               # P2 永远可做
    cap["can_verify_shape"] = True       # P3 CPU 轨可证 shape
    cap["can_verify_numeric"] = npu_ok   # CPU↔NPU 一致性需卡
    cap["can_train"] = npu_ok
    cap["can_bench"] = npu_ok
    cap["can_judge"] = True              # P7 判定链 0-GPU 可跑

    if not npu_ok:
        reasons.append("no_npu_visible: 无可用 NPU（走 CPU 轨）")
        return "cpu_only", reasons, cap

    if not triton_ok:
        reasons.append("triton_unavailable: triton-ascend backend "
                       + str(sw.get("triton_ascend_backend"))
                       + " / kernel " + str(sw.get("triton_npu_kernel"))
                       + " → 算子后端降级 ascendc/eager")
    if not pre.get("python3_dev_ok"):
        reasons.append("missing_python3_dev: 缺 Python.h，triton 驱动无法编译（装 python3-dev + build-essential）")

    # 官方几何需要 2 个 die（world_size=2）；die 数与档位
    if die >= 2:
        return ("full_npu" if triton_ok else "npu_triton_off"), reasons, cap
    reasons.append("single_die: 仅 %s 个 die → 无法复刻官方 dp2 几何，逐点不可比（仅窗口口径）" % die)
    return ("single_die" if triton_ok else "single_die_triton_off"), reasons, cap


def decide_config_profile(hw, sw):
    """按芯片/显存自动推荐配置档（供 P2 使用）。"""
    chip = (hw.get("chip_model") or "")
    hbm = hw.get("hbm_mb_per_chip")
    hbm_gb = round(hbm / 1024) if hbm else None
    die = hw.get("chip_count")
    die = int(die) if (die and str(die).isdigit()) else 1

    prof = {"chip_model": chip, "hbm_gb_per_chip": hbm_gb, "die_count": die}
    if hbm_gb is not None and hbm_gb >= 60 and die >= 2:
        prof.update({
            "profile": "910c_dual_die_official_geometry",
            "world_size": 2, "mbs": 4, "gas": 1, "dp": 2,
            "recompute": False, "enable_chunk_loss": False, "enable_activation_offload": False,
            "pregather": True, "prefetch": 1,
            "note": "128GB 双 die：官方几何 + 关闭显存节省开关 + pregather（实测 51× 且超官方）",
        })
    elif hbm_gb is not None and hbm_gb >= 60:
        prof.update({
            "profile": "910c_single_die",
            "world_size": 1, "mbs": 8, "gas": 1, "dp": 1,
            "recompute": False, "enable_chunk_loss": False, "enable_activation_offload": False,
            "pregather": True, "prefetch": 1,
            "note": "64GB 单 die：单 micro 上界几何；几何与官方不同，仅窗口口径可比",
        })
    else:
        prof.update({
            "profile": "910b4_low_mem",
            "world_size": 1, "mbs": 2, "gas": 4, "dp": 1,
            "recompute": True, "enable_chunk_loss": True, "enable_activation_offload": True,
            "pregather": False, "prefetch": 1,
            "note": "32GB 档：显存节省开关全开；mbs 上限 2（m4g2 会 OOM）",
        })
    prof["operator_backend"] = ("triton" if sw.get("triton_npu_kernel") == "ok" else "eager")

    # ---- ★ 坑 108：CANN beta/RC 上 dcp.load() 的 gather 路径在 aicpu 上不可用
    #   （RuntimeError: ACL stream synchronize failed 507018 / HcclLaunchAicpuKernel /
    #    libscatter_aicpu_kernel.so）→ 改走"rank0 加载 + 广播"。
    #   关键：这个开关由**探测到的环境**决定，而不是靠人记得去改配置（坑 104 的同一教训），
    #   并且把判据落盘到 env.json，判定链/复盘时能溯源。
    cann_full = " ".join([str(x) for x in (sw.get("cann_dirs") or [])]
                         + [str(sw.get("cann_version") or "")]).strip()
    cann_beta = bool(re.search(r"(beta|alpha|rc\d|\.dev)", cann_full, re.I))
    prof["load_rank0_and_broadcast"] = cann_beta
    prof["load_rank0_reason"] = (
        "CANN=%s 含 beta/RC/dev 标识 → 规避 dcp.load() 的 HCCL gather aicpu 缺陷（坑 108），"
        "改为 rank0 加载后广播" % (cann_full or "unknown")) if cann_beta else (
        "CANN=%s 未含 beta/RC 标识（正式版）→ 各 rank 自行加载分片（上游默认路径）"
        % (cann_full or "unknown"))

    # ---- ★ 坑 113：**save 路径是另一个方向**，坑 108 的开关管不到它。
    #   不修时：100 步训练正常跑完，然后在收尾保存处 `dcp.save` 的 SavePlan
    #   （`distW.reduce_scatter('plan')` → `scatter_object_list`）撞 aicpu → ACL 507018 → SIGABRT
    #   → **进程 rc≠0**（"成功 = rc=0 + 日志内容"这条红线因此不过）。
    #   已验证的配置级解：`training.save_format: hf`（走 HF safetensors 保存，不经 DCP 计划广播）。
    #   注意 `trainer.py:439-448`：`save_format != dcp` 时必须同时
    #   `no_save_optim` 与 `no_save_rng` 为真，否则会被**静默强制回退 dcp**（bug 又回来）。
    #   同坑 104/108 的教训：由**探测到的环境**决定，不靠人记得改配置。
    prof["save_format"] = "hf" if cann_beta else "auto"
    prof["save_format_reason"] = (
        "CANN=%s 含 beta/RC/dev → 收尾保存改走 HF safetensors，规避 dcp.save 的 "
        "SavePlan 计划广播缺陷（坑 113）；需 no_save_optim/no_save_rng 同为真"
        % (cann_full or "unknown")) if cann_beta else (
        "CANN=%s 为正式版 → save_format 用上游默认 auto" % (cann_full or "unknown"))

    # ---- ★ 2026-09-21：`skip_gdn_recompute` 也要**由环境决定**，不能靠人记得改。
    #   官方验收日志里是 **True**（`examples/train/official_baseline.log:94` 与 L27 注册行；
    #   `config/baselines/officialB.yaml:244` 留证），而我方 A3 验收锚点跑的是 `false`
    #   —— 这是一处**未披露的计算量偏离**，且此前判据链完全不比对它。
    #   ⚠ 源码耦合（读 `modeling_qwen3_5.py:625` 才知，**不能按名字推断**）：
    #     `gdn_implementation == 'eager'` 时 `skip_gdn_recompute=True` 会 raise NotImplemented。
    #   ⇒ 规则：后端是 triton/ascendc 时取官方值 True；后端退到 eager 时**必须** False。
    backend = str(prof.get("operator_backend") or "triton")
    prof["skip_gdn_recompute"] = (backend != "eager")
    prof["skip_gdn_recompute_reason"] = (
        "operator_backend=%s（非 eager）→ 取**官方值 True**（official_baseline.log:94 / L27 注册行）"
        % backend) if backend != "eager" else (
        "operator_backend=eager → **必须 False**：源码 modeling_qwen3_5.py:625 "
        "`eager + skip_gdn_recompute=True` 会 raise NotImplemented（两个开关耦合）")
    return prof


# ---------------------------------------------------------------- 档位自检（可证伪）

def triton_version_verdict(imp, ascend_dist, upstream_dist):
    """纯函数：三路 triton 版本 → (是否一致, 异常说明列表)。

    ★ 坑 123：一个版本号不够 —— 实测新机器上三个数**互不相同**
    （`import triton.__version__=3.2.0` / dist `triton-ascend=3.2.2` / dist `triton`(上游)=**3.5.0**），
    且 `triton/`、`triton-3.5.0.dist-info`、`triton_ascend-3.2.2.dist-info` **共存**。
    `triton-ascend` 与上游 `triton` **提供同名包**，同时安装即"混装"——性能归因必须先看见它。
    取主次版本比较（补丁号差异不算异常）。
    """
    def _mm(v):
        return v.split(".")[:2]
    bad = []
    dists = [(n, v) for n, v in (("triton-ascend", ascend_dist), ("triton", upstream_dist))
             if v and v != "ABSENT"]
    for name, v in dists:
        if imp and _mm(imp) != _mm(v):
            bad.append("import(%s) != %s(%s)" % (imp, name, v))
    if len(dists) >= 2 and dists[0][1] != dists[1][1]:
        bad.append("上游 triton(%s) 与 triton-ascend(%s) 同时安装（同名包混装）"
                   % (dists[1][1], dists[0][1]))
    return (not bad), bad


def selftest_profile():
    """档位与环境开关自检（★ 坑 110/111：可证伪 —— 好例必过、坏例必拒）。

    重点验证 `load_rank0_and_broadcast` **随环境走**：
      * CANN 含 beta/RC/dev → True（规避坑 108 的 dcp.load HCCL gather aicpu 缺陷）
      * CANN 为正式版 → False（上游默认路径）
    以及坑 103 的复发路径（HBM 探测失败）必须掉到 world_size=1 档。
    """
    HW = {"chip_model": "9382", "hbm_mb_per_chip": 65536, "chip_count": "2"}
    # 元组第 7 项 = 期望的 skip_gdn_recompute（★ 2026-09-21 新增）：
    #   官方值是 **True**，但源码耦合要求 eager 后端必须 False
    #   （modeling_qwen3_5.py:625 → eager + True 会 raise NotImplemented）。
    #   所以"后端降级 eager"那条**必须**期望 False —— 这正是双向对照的坏例。
    cases = [
        ("CANN 9.1.0-beta.3（新机器）", dict(HW),
         {"triton_npu_kernel": "ok", "cann_dirs": ["/usr/local/Ascend/cann-9.1.0-beta.3"],
          "cann_version": "9.1.0"}, True, 2, "hf", True),
        ("CANN 9.0.0 正式版（旧机器 A3）", dict(HW),
         {"triton_npu_kernel": "ok", "cann_dirs": ["/usr/local/Ascend/cann-9.0.0"],
          "cann_version": "9.0.0"}, False, 2, "auto", True),
        ("CANN 目录含 RC1", dict(HW),
         {"triton_npu_kernel": "ok", "cann_dirs": ["/usr/local/Ascend/cann-9.1.RC1"],
          "cann_version": "9.1"}, True, 2, "hf", True),
        ("HBM 探测失败（坑103 复发路径）",
         {"chip_model": "9382", "hbm_mb_per_chip": None, "chip_count": "2"},
         {"triton_npu_kernel": "ok", "cann_dirs": ["/usr/local/Ascend/cann-9.0.0"],
          "cann_version": "9.0.0"}, False, 1, "auto", True),
        ("triton 不可用 → 后端降级 eager", dict(HW),
         {"triton_npu_kernel": "fail", "cann_dirs": ["/usr/local/Ascend/cann-9.0.0"],
          "cann_version": "9.0.0"}, False, 2, "auto", False),
    ]
    ok = True
    for name, hw, sw, want_flag, want_world, want_save, want_skip in cases:
        prof = decide_config_profile(hw, sw)
        got = (prof.get("load_rank0_and_broadcast") == want_flag
               and prof.get("world_size") == want_world
               and prof.get("save_format") == want_save
               and prof.get("skip_gdn_recompute") is want_skip)
        ok = ok and got
        print("  [%s] %-30s world=%s load_rank0=%-5s save_format=%-4s backend=%-6s skip_gdn=%-5s"
              % ("PASS" if got else "FAIL", name, prof.get("world_size"),
                 prof.get("load_rank0_and_broadcast"), prof.get("save_format"),
                 prof.get("operator_backend"), prof.get("skip_gdn_recompute")))
        if not got:
            print("         期望 world=%s load_rank0=%s save_format=%s skip_gdn=%s"
                  % (want_world, want_flag, want_save, want_skip))
    print("PROFILE_SELFTEST_%s cases=%d" % ("OK" if ok else "FAIL", len(cases)))
    if not ok:
        return 1

    # ---- ★ 坑 123：triton 三路版本一致性判据的双向自检（好例过 / 坏例必被抓）
    tri_cases = [
        ("真实新机器：imp3.2.0 / asc3.2.2 / 上游3.5.0（混装）", ("3.2.0", "3.2.2", "3.5.0"), False),
        ("干净环境：只有 triton-ascend", ("3.2.0", "3.2.2", "ABSENT"), True),
        ("import 与 dist 主次版本不同", ("3.5.0", "3.2.2", "ABSENT"), False),
        ("只有上游 triton 且 import 不符", ("3.2.0", "ABSENT", "3.5.0"), False),
        ("补丁号差异不算异常", ("3.2.0", "3.2.7", "ABSENT"), True),
        ("版本都未知 → 不作断言", (None, None, None), True),
    ]
    print("-- triton 版本一致性判据（坑 123）--")
    tri_ok = True
    for name, (i, a, u), want in tri_cases:
        got, bad = triton_version_verdict(i, a, u)
        good = (got == want)
        tri_ok = tri_ok and good
        print("  [%s] %-44s consistent=%-5s %s"
              % ("PASS" if good else "FAIL", name, got, ("(%d 条异常)" % len(bad)) if bad else ""))
    print("TRITON_VERDICT_SELFTEST_%s cases=%d" % ("OK" if tri_ok else "FAIL", len(tri_cases)))
    return 0 if tri_ok else 1


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="P0 环境探测（决定迁移路径与配置档）")
    ap.add_argument("--out", default="out/probe/env.json", help="输出 JSON 路径")
    ap.add_argument("--data-dir", default="/root", help="数据盘路径（探测剩余空间）")
    ap.add_argument("--quick", action="store_true", help="跳过耗时的 triton kernel 实测")
    ap.add_argument("--selftest", action="store_true", help="只跑档位/环境开关自检并退出（不探测硬件）")
    args = ap.parse_args()

    if args.selftest:
        return selftest_profile()

    env = {"schema": "migrator_env.v1",
           "probed_at": datetime.now(timezone.utc).isoformat()}

    print("== P0 环境探测 ==")
    hw = probe_hardware();        env["hardware"] = hw;    print("硬件:", json.dumps(hw, ensure_ascii=False)[:200])
    sw = probe_software();        env["software"] = sw;    print("软件:", json.dumps(sw, ensure_ascii=False)[:260])
    pre = probe_prereq();         env["prereq"] = pre;     print("前置:", json.dumps(pre, ensure_ascii=False))
    res = probe_resources(args.data_dir); env["resources"] = res; print("资源:", json.dumps(res, ensure_ascii=False))

    path, reasons, cap = decide_path(hw, sw, pre)
    profile = decide_config_profile(hw, sw)
    env["path"] = path
    env["degraded"] = bool(reasons)
    env["degrade_reasons"] = reasons
    env["capabilities"] = cap
    env["recommended_profile"] = profile

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(env, f, ensure_ascii=False, indent=1)

    print("\n" + "=" * 64)
    print("执行路径 : %s" % path)
    print("芯片     : %s | die=%s | HBM/chip=%sGB" % (
        hw.get("chip_model"), hw.get("chip_count"), profile.get("hbm_gb_per_chip")))
    print("triton   : backend=%s kernel=%s" % (sw.get("triton_ascend_backend"), sw.get("triton_npu_kernel")))
    print("推荐配置档: %s" % profile.get("profile"))
    print("  几何   : world=%s mbs=%s gas=%s dp=%s" % (
        profile.get("world_size"), profile.get("mbs"), profile.get("gas"), profile.get("dp")))
    print("  开关   : recompute=%s chunk_loss=%s act_offload=%s pregather=%s" % (
        profile.get("recompute"), profile.get("enable_chunk_loss"),
        profile.get("enable_activation_offload"), profile.get("pregather")))
    print("  后端   : %s" % profile.get("operator_backend"))
    print("能力矩阵 : %s" % json.dumps(cap, ensure_ascii=False))
    if reasons:
        print("\n降级原因（degraded=true）：")
        for r in reasons:
            print("  - %s" % r)
    print("=" * 64)
    print("PROBE_OK path=%s env=%s" % (path, out))
    print("下一步: P1 迁移点识别 → python3 scripts/10_analyze_points.py <modeling_source.py>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
