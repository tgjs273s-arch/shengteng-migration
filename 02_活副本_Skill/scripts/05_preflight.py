#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_preflight.py —— 开工前预检：能力矩阵（鲁棒性的第一道闸门）

目的
----
把"跑到一半才发现环境不支持"变成"**开工前就知道会怎样**"。
对每一项能力给出三态判定，并**显式说明影响**（跳过哪些阶段 / 走哪条降级路径）。

    OK        该能力可用
    DEGRADED  可用但有限（会走降级路径，已登记到 out/degradations.json）
    BLOCKED   不可用（依赖它的阶段将被跳过，**不会伪造结果**）
    SKIP      该项在本次模式下不适用

用法
----
    python3 scripts/05_preflight.py                  # 人读矩阵
    python3 scripts/05_preflight.py --json           # 机读（写入 out/preflight/capability.json）
    python3 scripts/05_preflight.py --want full      # full 模式：全量数据/100 步为必需项

退出码
------
    0  所有**必需**能力 OK 或 DEGRADED（可继续，降级已记录）
    3  存在 BLOCKED 的必需能力（继续也能跑，但产物只能到降级档 —— 由上游决定是否接受）
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _envcompat import (  # noqa: E402
    SOURCES, cann_root, cann_version, default_data_dir, default_msmm_dir,
    degrade, driver_version, has_python_h, install_hint, is_prerelease,
    missing_commands, npu_info, pkg_manager, python_exe, python_version,
    safe_hostname, torch_npu_info, which,
)

SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(SKILL_ROOT, "out", "preflight")
DEG_FILE = os.path.join(SKILL_ROOT, "out", "degradations.json")

ROWS = []


def row(cap, status, evidence, impact=""):
    ROWS.append({"capability": cap, "status": status, "evidence": str(evidence)[:200],
                 "impact": impact})
    sym = {"OK": "OK  ", "DEGRADED": "DEGR", "BLOCKED": "BLOCK", "SKIP": "SKIP"}[status]
    print("  %-5s %-34s %s" % (sym, cap, str(evidence)[:96]))
    if impact and status != "OK":
        print("        └─ 影响: %s" % impact)


def _http_ok(url, timeout=8):
    try:
        import subprocess
        r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                            "--max-time", str(timeout), url],
                           capture_output=True, text=True, timeout=timeout + 5)
        return (r.stdout or "").strip()
    except Exception:
        return "000"


def main():
    ap = argparse.ArgumentParser(description="Skill 开工前预检（能力矩阵）")
    ap.add_argument("--json", action="store_true", help="写 out/preflight/capability.json")
    ap.add_argument("--want", choices=["quick", "full"], default="quick",
                    help="quick=流程连通即可; full=需全量数据与 100 步")
    a = ap.parse_args()
    want_full = (a.want == "full")

    print("=" * 78)
    print("qwen35-ascend-migrator 预检能力矩阵   want=%s" % a.want)
    print("  host=%s  py=%s(%s)  建议 MSMM=%s  数据=%s"
          % (safe_hostname(), python_version() or "?", python_exe(),
             default_msmm_dir(), default_data_dir()))
    print("=" * 78)

    profile = {"want": a.want, "host": safe_hostname(), "python": python_version(),
               "python_exe": python_exe()}

    # ---------------- 1. 硬件 ----------------
    print("\n[硬件 / 驱动 / CANN]")
    ni = npu_info()
    tn = torch_npu_info()
    profile["npu"] = ni
    profile["torch_npu"] = tn

    # ★ 坑 137：npu_visible 曾**只看 npu-smi 一条路径** —— 实测存在"npu-smi 报
    #   `npu get board type failed. ret is -9005`、但 torch_npu 能正常分配并算对数值"的机器
    #   （受限容器里 npu-smi 常不可用）。此时旧逻辑会把**可用的机器**判成 BLOCKED →
    #   直接跳过 P5/P6（假阴性）。现改为**三路回退**，并记录命中的是哪一路：
    #     ① npu-smi（原路径）② torch_npu 设备数 + 真机分配/计算自证 ③ /dev/davinci* 节点存在
    tn_ok = bool((tn or {}).get("available")) and int((tn or {}).get("count") or 0) > 0
    tn_name = ((tn or {}).get("names") or ["?"])[0]
    devs = sorted(d for d in os.listdir("/dev") if d.startswith("davinci"))
    if ni.get("visible"):
        row("npu_visible", "OK", "npu-smi 可见, Chip Count=%s（路径 ①）" % ni.get("chip_count"))
    elif tn_ok:
        row("npu_visible", "DEGRADED",
            "npu-smi 不可用，但 torch_npu 自证可用：device_count=%s name=%s（路径 ②；/dev 下 %s）"
            % ((tn or {}).get("count"), tn_name, ",".join(devs) or "无"),
            "npu-smi 相关能力（HBM/利用率采样）不可用 → 改用 torch_npu 内存接口；"
            "**不得**把 npu-smi 缺失写成『无 NPU』")
        degrade(DEG_FILE, "npu_smi_unavailable",
                "npu-smi 无输出但 torch_npu 可用 → HBM/利用率类采样改用 torch_npu 接口")
    elif devs:
        row("npu_visible", "DEGRADED",
            "npu-smi 与 torch_npu 均未自证，但 /dev 下存在 %s（路径 ③）" % ",".join(devs),
            "设备节点在但运行栈未自证 → 先用 00b_triton_min_kernel.py 定案，再决定是否跑训练")
    else:
        row("npu_visible", "BLOCKED", "三路均不可用：npu-smi 无输出 / torch_npu 不可用 / 无 /dev/davinci*",
            "P5/P6 跳过；P1/P2/P3(shape-only)/P7 仍可跑（cpu_only 轨）")

    chips = ni.get("chip_count") or (tn or {}).get("count") or 0
    if chips >= 2:
        row("official_geometry_dp2", "OK", "%d 个 die → 可复刻 world2/mbs4/gas1(dp2)" % chips)
    elif chips == 1:
        row("official_geometry_dp2", "DEGRADED", "仅 1 die",
            "几何与官方不同 → 只能窗口口径可比，不能逐点断言")
        degrade(DEG_FILE, "official_geometry_dp2", "仅 1 die，无法复刻官方 dp2 几何")
    else:
        row("official_geometry_dp2", "BLOCKED", "无可见 die",
            "无法进行任何真实训练")

    # ★ 真实 SoC 名：npu-smi 可能只给通用名，torch_npu 才给型号（坑 8）
    if tn and tn.get("names"):
        row("soc_identity_torch_npu", "OK", ",".join(tn["names"]),
            "与 versions.lock 的 9382 比对以判断是否同款芯片")
        profile["soc"] = tn["names"][0] if tn["names"] else None
    else:
        row("soc_identity_torch_npu", "SKIP", "torch_npu 未装/不可用 → 无法取真实 SoC 名")

    cv = cann_version()
    if cv:
        st = "DEGRADED" if is_prerelease(cv) else "OK"
        row("cann_available", st, "CANN %s @ %s" % (cv, cann_root()),
            "预发布版 CANN：torch_npu/triton 配套可能不同于 GA，需实测" if st == "DEGRADED" else "")
        if st == "DEGRADED":
            degrade(DEG_FILE, "cann_prerelease", "CANN %s 为预发布版" % cv)
    else:
        row("cann_available", "BLOCKED", "找不到 CANN 根目录/版本", "无法编译任何 NPU 算子")
    profile["cann"] = cv
    profile["driver"] = driver_version()

    # ---------------- 2. 工具链 ----------------
    print("\n[工具链 / 前置依赖]")
    pm = pkg_manager()
    profile["pkg_manager"] = pm
    if pm:
        row("package_manager", "OK", pm)
    else:
        row("package_manager", "DEGRADED", "无 dnf/yum/apt",
            "无法自动装系统依赖；缺 Python.h/编译器时需手工装")
        degrade(DEG_FILE, "package_manager", "无可用包管理器，系统依赖需手工安装")

    need = ["tar", "grep", "sed", "awk", "find", "curl"]
    miss = missing_commands(need)
    if miss:
        row("required_commands", "DEGRADED", "缺: %s" % ",".join(miss),
            "相关步骤可能失败；脚本应避免依赖缺失命令")
        degrade(DEG_FILE, "required_commands", "缺少基础命令: %s" % ",".join(miss))
    else:
        row("required_commands", "OK", "tar/grep/sed/awk/find/curl 齐备")

    # ★ 坑 39：极简镜像可能没有 hostname —— 记录但不阻塞
    row("optional_commands", "OK" if which("hostname") else "DEGRADED",
        "hostname %s, unzip %s, git %s"
        % ("有" if which("hostname") else "缺失", "有" if which("unzip") else "缺失",
           "有" if which("git") else "缺失"),
        "缺失不影响主流程（脚本已改用 /proc 读取主机名）" if not which("hostname") else "")

    if has_python_h():
        row("python_dev_headers", "OK", "Python.h 就位 → triton kernel 可现场编译")
    else:
        row("python_dev_headers", "BLOCKED", "缺 Python.h",
            "triton 编译会失败并**静默回退 CPU**（坑 22）；修复: %s" % install_hint("pyd"))
        degrade(DEG_FILE, "python_dev_headers", "缺 Python.h，triton 有静默回退 CPU 风险",
                severity="blocked")

    if which("gcc") and which("make"):
        row("compiler", "OK", "gcc + make 就位")
    else:
        row("compiler", "BLOCKED", "缺 gcc/make", "无法编译算子；修复: %s" % install_hint("cc"))

    import importlib.util as _ilu
    yaml_ok = _ilu.find_spec("yaml") is not None
    row("pyyaml", "OK" if yaml_ok else "BLOCKED",
        "pyyaml %s" % ("已装" if yaml_ok else "缺失"),
        "判定链的 yaml 解析依赖（坑 45）；修复: python3 -m pip install pyyaml" if not yaml_ok else "")
    if not yaml_ok:
        degrade(DEG_FILE, "pyyaml_missing", "缺 pyyaml，判定链 yaml 检查将 SKIP")

    # ---------------- 3. 计算栈 ----------------
    print("\n[计算栈]")
    if tn and tn.get("available"):
        row("torch_npu", "OK", "torch %s / torch_npu %s, %d device"
            % (tn.get("torch"), tn.get("torch_npu"), tn.get("count")))
    else:
        row("torch_npu", "BLOCKED", "torch_npu 不可用",
            "P5/P6 无法执行；修复: bash scripts/bringup.sh --stage=1")
        degrade(DEG_FILE, "torch_npu_unavailable", "torch_npu 不可用")

    tri_probe = os.path.join(SKILL_ROOT, "scripts", "00b_triton_min_kernel.py")
    tri_status, tri_ev = "BLOCKED", "无判据脚本"
    if os.path.isfile(tri_probe) and tn and tn.get("available"):
        try:
            import subprocess
            r = subprocess.run([python_exe(), tri_probe], capture_output=True,
                               text=True, timeout=600)
            # ★ 坑 56：判据**绝不能取"最后一行"** —— 把 stdout 与 stderr 串起来后取末行，
            #   会取到 CANN 的 NPUCachingAllocator Warning，于是**明明通过却判成降级**。
            #   必须 grep 可证伪的显式标记（与坑 42/46 同族：不要依赖脆弱的输出位置）。
            merged = ((r.stdout or "") + "\n" + (r.stderr or ""))
            ok_line = [l for l in merged.splitlines() if "TRITON_NPU_OK" in l]
            bad_line = [l for l in merged.splitlines() if "TRITON_NPU_FAIL" in l]
            if (r.returncode == 0) and ok_line:
                tri_status = "OK"
                tri_ev = ok_line[-1][:120]
            elif bad_line:
                tri_status = "DEGRADED"
                tri_ev = bad_line[-1][:120]
            else:
                tri_status = "DEGRADED"
                tri_ev = ("未见明确标记 rc=%s；末行: %s"
                          % (r.returncode, (merged.strip().splitlines() or ["<空>"])[-1][:90]))
        except Exception as e:
            tri_ev = "%s: %s" % (type(e).__name__, e)
    elif not (tn and tn.get("available")):
        tri_status, tri_ev = "SKIP", "torch_npu 不可用，无法实测 kernel"
    row("triton_ascend_kernel", tri_status, tri_ev,
        "triton 不可用 → 后端降级 triton→ascendc→eager（数值等价已验证 Δ<0.2%），"
        "性能会显著下降" if tri_status == "DEGRADED" else "")
    if tri_status == "DEGRADED":
        degrade(DEG_FILE, "triton_backend", "triton kernel 实测未通过，降级后端链")

    # ---------------- 4. 资产 ----------------
    print("\n[框架与资产]")
    msmm = default_msmm_dir()
    msmm_ok = os.path.isdir(msmm)
    row("mindspeed_mm", "OK" if msmm_ok else "BLOCKED", msmm,
        "P5 无法执行；修复: bash scripts/bringup.sh --stage=2" if not msmm_ok else "")

    dd = default_data_dir()
    hf = os.path.join(os.path.dirname(dd.rstrip("/")), "Qwen3.5-0.8B-hf")
    hf_ok = any(os.path.isdir(p) for p in (hf, "/root/Qwen3.5-0.8B-hf"))
    llava = os.path.join(dd, "llava", "llava_instruct_150k.json")
    conv = os.path.join(dd, "output_llava_coco_data.json")
    coco = os.path.join(dd, "coco", "train2017")
    n_img = 0
    try:
        n_img = len([f for f in os.listdir(coco) if f.lower().endswith((".jpg", ".png"))])
    except Exception:
        pass
    llava_ok = os.path.isfile(llava)
    conv_ok = os.path.isfile(conv)

    row("weight_hf", "OK" if hf_ok else ("BLOCKED" if want_full else "DEGRADED"),
        hf if hf_ok else "缺失",
        "P5 需权重；由 P4 下载（modelscope）" if not hf_ok else "")
    row("data_llava_json", "OK" if llava_ok else ("BLOCKED" if want_full else "DEGRADED"),
        "%s (%s B)" % (llava, os.path.getsize(llava) if llava_ok else "-") if llava_ok else "缺失",
        "无 llava json → 数据可比性降为不可比" if not llava_ok else "")
    row("data_coco_images", "OK" if n_img >= 118287 else (
        "DEGRADED" if n_img > 0 else ("BLOCKED" if want_full else "DEGRADED")),
        "%d / 118287 张" % n_img,
        "图片不全 → 只能窗口口径可比" if n_img < 118287 else "")
    row("data_converted_json", "OK" if conv_ok else ("BLOCKED" if want_full else "DEGRADED"),
        "%s (%s 样本)" % (conv, "-") if conv_ok else "缺失",
        "P5 需转换后的训练 json" if not conv_ok else "")

    if not (llava_ok and conv_ok and n_img >= 118287):
        degrade(DEG_FILE, "data_comparability",
                "数据未达全量同源（llava=%s converted=%s coco=%d/118287）"
                % (llava_ok, conv_ok, n_img))
        # 重要：明确写出"因此不得宣称什么"
        row("pointwise_claim_allowed", "BLOCKED", "数据非全量同源",
            "**不得**宣称逐点可比/精度已对齐；只能宣称窗口口径或流程连通")

    # ---------------- 5. 网络 ----------------
    print("\n[网络与源]")
    reach = {}
    for name in ("pypi", "msmm_git", "modelscope", "hf"):
        codes = [_http_ok(u) for u in SOURCES[name][:2]]
        best = next((c for c in codes if c.startswith("2") or c.startswith("3")), codes[0])
        reach[name] = {"codes": codes, "usable": any(c.startswith(("2", "3")) for c in codes)}
        st = "OK" if reach[name]["usable"] else "DEGRADED"
        note = ""
        if "429" in codes:
            note = "HTTP 429（限流）→ 安装/下载已内建指数退避重试"
        row("net_%s" % name, st, "HTTP %s" % ",".join(codes), note)
        if not reach[name]["usable"]:
            degrade(DEG_FILE, "network_%s" % name, "源不可达 HTTP %s" % ",".join(codes))
    profile["network"] = reach

    # ---------------- 6. 目标硬盘空间 ----------------
    print("\n[磁盘]")
    try:
        du = shutil.disk_usage("/")
        free_gb = du.free / (1024 ** 3)
        need_gb = 45 if want_full else 5
        st = "OK" if free_gb >= need_gb else "BLOCKED"
        row("disk_space", st, "%.1f GB 可用（%s 模式需 ~%d GB）" % (free_gb, a.want, need_gb),
            "全量 COCO 压缩包 19GB + 解压 19GB + 权重 2GB" if st != "OK" else "")
    except Exception as e:
        row("disk_space", "SKIP", "无法获取: %s" % e)

    # ---------------- 汇总 ----------------
    summary = {}
    for r in ROWS:
        summary[r["status"]] = summary.get(r["status"], 0) + 1
    crit_blocked = [r["capability"] for r in ROWS
                    if r["status"] == "BLOCKED" and r["capability"] not in
                    ("data_llava_json", "data_coco_images", "data_converted_json",
                     "weight_hf", "pointwise_claim_allowed", "package_manager")]

    print("\n" + "=" * 78)
    print("预检汇总: " + "  ".join("%s=%d" % (k, v) for k, v in sorted(summary.items())))
    try:
        degs = json.load(open(DEG_FILE, encoding="utf-8"))
    except Exception:
        degs = []
    print("已登记降级: %d 项  → %s" % (len(degs), DEG_FILE))
    if crit_blocked:
        print("关键能力受阻: %s" % ", ".join(crit_blocked))
    print("PREFLIGHT_%s" % ("OK" if not crit_blocked else "BLOCKED"))

    profile["rows"] = ROWS
    profile["summary"] = summary
    profile["degradations"] = degs
    if a.json:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "capability.json"), "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=1)
        print("已写: %s" % os.path.join(OUT_DIR, "capability.json"))
    return 0 if not crit_blocked else 3


if __name__ == "__main__":
    sys.exit(main())
