#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""61_mem_comm_probe.py — 第 4 步数据：训练期的**设备 HBM + cgroup 节流**时间序列

为什么这样设计（以及做不到什么）
--------------------------------
* **能做**：从进程外在 0.5 s 粒度采
  - `npu-smi info -t memory -i <id>`（设备级 HBM 使用/总量）
  - `/sys/fs/cgroup/**/cpu.stat`（节流增量）
  - 训练日志里 torch 侧的 `memory (MB) allocated/reserved` 行（若训练脚本打印）
  并给出 identity（boot/hostname/affinity）+ 步时长，落地 JSON。
* **做不到（必须如实标注）**：**torch 分配器的逐帧 allocated/reserved/峰值/分配重试**
  只有**在训练进程内**才能取到（`torch_npu.npu.memory_stats()`）。
  本次不改训练脚本 → 因此本探针**给不出**分配器级曲线；若后续需要，只能在**诊断副本**里加 3 行 hook，
  且该副本**不得**进入交付 Skill。

判据（跑之前定好，避免事后解释）
--------------------------------
1. 设备 HBM 峰值相对单 die 容量占比：给出 `peak_used / total`；
2. 节流：训练窗口内 `nr_throttled` 与 `throttled_time` 的**增量**（不是历史累计）；
3. **只有在"等待下降或步时长下降"时，才把显存占用升高算作收益**（本脚本本身不下收益结论）；
4. 采样间隔 0.5 s 与步长 0.43 s 仍会混叠 → **只看峰值与趋势，不当稳态值**；
5. **消费侧前置守卫（坑 140）**：本脚本允许 `--config` 直接喂一份配置，而**手工配置会绕过
   P0/P2 按实测 CANN 写入的规避开关** → dp2 在 `load()` 就崩在 `dcp.load` 的
   `scatter_object_list`（AICPU 507018），报错长得像"HCCL/设备坏了"，实际**训练一步都没跑**。
   故启动前先用 `out/probe/env.json` 里 P0 落的档位核对配置，不一致则**拒绝启动**。

用法：bash scripts/p61_launch.sh /root/ops/<新目录>
      python3 scripts/61_mem_comm_probe.py --selftest   # 只看前置守卫自检
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time

PAT = re.compile(r"iteration\s+(\d+)\s*/\s*\d+.*?elapsed time per iteration \(ms\):\s*([\d.]+)", re.S)
ENVLIB = ("export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:"
          "/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:"
          "$LD_LIBRARY_PATH; ")


def sh(cmd, timeout=60):
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    return (r.stdout or "") + (r.stderr or "")


def npu_id():
    """★ 坑 137：`npu-smi` 不可用时**不得致命** —— 返回 None，
    设备级 HBM 随后由 `hbm_via_torch()`（torch_npu 接口）提供。"""
    m = re.search(r"NPU ID\s*:\s*(\d+)", sh(ENVLIB + "npu-smi info -l 2>&1"))
    return m.group(1) if m else None


def hbm(nid):
    if nid is None:
        return {"raw_first_line": "npu-smi 不可用（跳过该路径）", "hbm_usage_pct": None,
                "hbm_capacity_mb": None, "used_total": [None, None]}
    txt = sh(ENVLIB + "npu-smi info -t memory -i %s -c 0 2>&1" % nid)
    used = total = None
    m = re.search(r"HBM Usage Rate\(%\)\s*:\s*([\d.]+)", txt)
    m2 = re.search(r"HBM Capacity\(MB\)\s*:\s*(\d+)", txt)
    cap = float(m2.group(1)) if m2 else None
    mu = re.search(r"(\d+)\s*/\s*(\d+)", txt)
    if mu:
        used, total = float(mu.group(1)), float(mu.group(2))
    return {"raw_first_line": (txt.strip().splitlines() or [""])[0][:80],
            "hbm_usage_pct": float(m.group(1)) if m else None,
            "hbm_capacity_mb": cap, "used_total": [used, total]}


def hbm_via_torch():
    """★ 坑 137：npu-smi 不可用时（受限容器常见）改用 torch_npu 设备级内存接口。

    设备级 free/total 是**全局**的，旁路进程也能读到，因此**不需要**改训练脚本；
    但分配器逐帧曲线仍取不到（那需要进程内 hook，见文件头"做不到"一节）。
    """
    code = ("import json,torch,torch_npu\n"
            "f,t=torch.npu.mem_get_info()\n"
            "d={'free_mb':f/1048576,'total_mb':t/1048576,"
            "'alloc_mb':torch.npu.memory_allocated()/1048576,"
            "'reserved_mb':torch.npu.memory_reserved()/1048576}\n"
            "print(json.dumps(d))\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=120)
    for ln in (r.stdout or "").splitlines():
        if ln.strip().startswith("{"):
            d = json.loads(ln)
            tot = d.get("total_mb") or 0
            used = tot - (d.get("free_mb") or 0)
            return {"source": "torch_npu.mem_get_info", "used_mb": used, "total_mb": tot,
                    "hbm_usage_pct": round(used / tot * 100, 2) if tot else None,
                    "allocated_mb": d.get("alloc_mb"), "reserved_mb": d.get("reserved_mb")}
    return None


def cpu_stat():
    for root in ("/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct", "/sys/fs/cgroup"):
        p = os.path.join(root, "cpu.stat")
        if os.path.isfile(p):
            d = {}
            for ln in open(p).read().splitlines():
                parts = ln.split()
                if len(parts) == 2:
                    try:
                        d[parts[0]] = int(parts[1])
                    except ValueError:
                        pass
            return p, d
    return None, {}


def apply_single_die(doc, single):
    """★ 单 die 受限环境：保留 GBS=8 红线，几何改为 world1/mbs8/gas1。

    边界：official_comparable=False（每 rank micro-batch 8 vs 4、无 FSDP 通信）
    → 只能做发射/停顿/内存类诊断，**不得**用于通信类或"优于官方"的结论。
    """
    if not single:
        return doc, int(doc["parallel"]["data_parallel_size"]), True
    doc["parallel"]["data_parallel_size"] = 1
    doc["training"]["micro_batch_size"] = 8
    doc["training"]["gradient_accumulation_steps"] = 1
    return doc, 1, False


# ---------------------------------------------------------------- 消费侧前置守卫
# ★ 坑 140（= 坑 108 的真身）：`--config` 允许喂一份**手工**配置。P0/P2 会按**实测到的 CANN**
#   自动写入规避开关，但手工配置**绕过了生产侧**，而消费侧当时**没有任何检查** ——
#   于是真机上 dp2 在 `TrainEngine.__init__ → load()` 就崩：
#       dcp.load → distW.reduce_scatter("plan") → scatter_object_list
#       → soName=libscatter_aicpu_kernel.so, funcName=HcclLaunchAicpuKernel,
#         errcode=11006, runtime result=507018
#   这个报错**长得像"HCCL/设备坏了"**（同一台机器上 2 rank all_reduce 完全正常），
#   于是方向被带偏；真相是**训练一步都没跑到**，问题在配置里少了两个开关。
#   ★ 判据必须与生产侧**同源**：不按"配置自己声称什么"判断，而是读 P0 落的
#     `out/probe/env.json → recommended_profile`（那是按实测 CANN 决定的档位）。
BETA_TOKENS = r"(beta|alpha|rc\d|\.dev)"


def find_values(doc, key, acc=None):
    """递归收集 `key` 的所有取值（不猜路径、不取第一个 —— 同族缺陷见坑 111）。"""
    if acc is None:
        acc = []
    if isinstance(doc, dict):
        for k, v in doc.items():
            if k == key:
                acc.append(v)
            find_values(v, key, acc)
    elif isinstance(doc, list):
        for v in doc:
            find_values(v, key, acc)
    return acc


def detect_beta_requirement(env_path):
    """判定"本机是否需要 beta 规避开关"。返回 (need, lines)。

    need: True=必须满足 / False=正式版 CANN（不强制）/ None=**不可判定 → 保守按 True**。
    最后一种**刻意不放行**：环境读不到时静默放行，等于把守卫退化成"什么都没查"。
    """
    lines = []
    try:
        with open(env_path, encoding="utf-8") as fh:
            env = json.load(fh)
    except Exception as exc:
        lines.append("BETA_GUARD env.json 不可读(%s): %s → 保守按「需要规避」处理"
                     % (type(exc).__name__, env_path))
        return None, lines

    prof = (find_values(env, "recommended_profile") or [None])[0]
    if isinstance(prof, dict) and "load_rank0_and_broadcast" in prof:
        need = bool(prof.get("load_rank0_and_broadcast"))
        lines.append("BETA_GUARD 来源=env.json:recommended_profile（P0 按实测 CANN 落的档位）")
        lines.append("BETA_GUARD need=%s reason=%s"
                     % (need, str(prof.get("load_rank0_reason") or "")[:180]))
        if need and prof.get("save_format") not in (None, "hf"):
            lines.append("BETA_GUARD **档位自相矛盾**：need=True 但 save_format=%r" % prof.get("save_format"))
        return need, lines

    dirs = [str(x) for sub in find_values(env, "cann_dirs") if isinstance(sub, list) for x in sub]
    ver = [str(v) for v in find_values(env, "cann_version")]
    cann_full = " ".join(dirs + ver).strip()
    need = bool(re.search(BETA_TOKENS, cann_full, re.I))
    lines.append("BETA_GUARD 无 recommended_profile → 退回 CANN 标识判定：%s" % (cann_full or "unknown"))
    return need, lines


def beta_config_check(doc, env_path):
    """核对配置是否满足本机档位。返回 (lines, errors)。纯函数（不碰 NPU/网络）便于自检。"""
    need, lines = detect_beta_requirement(env_path)
    errors = []
    tr = doc.get("training") or {}

    if need is False:
        lines.append("BETA_GUARD 正式版 CANN → 本机**不强制**规避开关（如实标注，不假装查过）")
        return lines, errors
    if need is None:
        lines.append("BETA_GUARD 环境不可判定 → 按最保守处理（要求规避开关）")

    # ① load 侧（坑 108/140）：少了它，训练一步都跑不到
    if tr.get("load_rank0_and_broadcast") is not True:
        errors.append(
            "training.load_rank0_and_broadcast=%r 必须为 true —— 否则载入走 "
            "dcp.load → scatter_object_list，本 CANN 上 AICPU 崩溃（坑 108/140），"
            "训练一步都跑不到" % tr.get("load_rank0_and_broadcast"))

    # ② save 侧（坑 113）：与 load 侧是**两条独立路径**，只修一条会"载入正常、收尾才炸"
    if tr.get("save") in (None, "", "null"):
        lines.append("BETA_GUARD training.save 为空 → 本次不保存，save 侧无需规避")
    else:
        fmt = str(tr.get("save_format") or "dcp")
        if fmt == "dcp":
            errors.append(
                "training.save_format=%r 必须 != dcp —— dcp.save 同样走 SavePlan 的 "
                "scatter_object_list（坑 113），100 步跑完才在收尾 SIGABRT" % fmt)
        if not (tr.get("no_save_optim") and tr.get("no_save_rng")):
            errors.append(
                "save_format=%s 时必须 no_save_optim 与 no_save_rng 同为 true，"
                "否则 trainer **静默回退 dcp**（坑 113）→ 收尾仍会崩" % fmt)
    return lines, errors


def beta_guard_selftest():
    """★ INV-2：守卫本身也是判据，必须能证伪。每个用例都是**真的坏样本**。"""
    import tempfile

    beta = {"load_rank0_and_broadcast": True, "load_rank0_reason": "CANN 含 beta → 坑 108",
            "save_format": "hf"}
    ga = {"load_rank0_and_broadcast": False, "save_format": "auto"}
    raw_beta = {"software": {"cann_dirs": ["/usr/local/Ascend/cann-9.1.0-beta.3"],
                             "cann_version": "9.1.0"}}

    def base_cfg(**kw):
        tr = {"load": "/w", "save": "/s/ckpt", "load_rank0_and_broadcast": True,
              "save_format": "hf", "no_save_optim": True, "no_save_rng": True}
        tr.update(kw)
        return {"training": tr}

    table = [
        ("beta + 合规两开关 → 放行", {}, {"recommended_profile": beta}, 0),
        ("beta + load_rank0=false（坑 108/140）→ 拒",
         {"load_rank0_and_broadcast": False}, {"recommended_profile": beta}, 1),
        ("beta + save_format=dcp（坑 113）→ 拒", {"save_format": "dcp"}, {"recommended_profile": beta}, 1),
        ("beta + hf 但 no_save_optim=false（静默回退 dcp）→ 拒",
         {"no_save_optim": False}, {"recommended_profile": beta}, 1),
        ("beta + save 为空 → 放行（不保存则无需规避）", {"save": None}, {"recommended_profile": beta}, 0),
        ("★ 正式版 GA + 无开关 → 放行（**假阳性对照**，不得误拦）",
         {"load_rank0_and_broadcast": False, "save_format": "auto"}, {"recommended_profile": ga}, 0),
        ("env.json 不可读 → 保守拒（不静默放行）",
         {"load_rank0_and_broadcast": False}, "MISSING", 1),
        ("无档位 + CANN 含 beta → 按标识拒", {"load_rank0_and_broadcast": False}, raw_beta, 1),
    ]

    ok = True
    tmps = []
    for name, kw, env_doc, want_n in table:
        if env_doc == "MISSING":
            path = os.path.join(tempfile.gettempdir(), "no_such_env_%d.json" % os.getpid())
            if os.path.exists(path):
                os.remove(path)
        else:
            fd, path = tempfile.mkstemp(suffix=".json")
            tmps.append(path)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(env_doc, fh)
        _lines, errs = beta_config_check(base_cfg(**kw), path)
        good = (len(errs) == want_n)
        ok = ok and good
        print("  [%s] %-52s want_errs=%d got=%d%s"
              % ("PASS" if good else "FAIL", name, want_n, len(errs),
                 "" if good else " :: " + "; ".join(errs)[:160]))
    for path in tmps:
        try:
            os.remove(path)
        except OSError:
            pass
    print("BETA_GUARD_SELFTEST cases=%d failed=%d" % (len(table), 0 if ok else 1))
    return 0 if ok else 1


# ---------------------------------------------------------------- 容器内存口径
# ★ 坑 143：dp2 首次跑到取数据时，torch 报
#     `RuntimeError: DataLoader worker (pid ...) is killed by signal: Killed`
#     `ERROR: Unexpected bus error encountered in worker. This might be caused by
#      insufficient shared memory (shm).`
#   —— **这句提示把人指向 /dev/shm，但它是错的**：本机 /dev/shm 16 GB、用量 0%。
#   真正的判据在**容器 cgroup 自己的计数**：`memory.max_usage_in_bytes` 撞到
#   `memory.limit_in_bytes`（本机 240 GB），`memory.oom_control` 的 `oom_kill` 计数 +2。
#   ⇒ 报错文案点名了错误的资源，**资源归属必须去问该资源的记账文件**，不能读文案。
CGROUP_MEM = "/sys/fs/cgroup/memory"


def cgroup_mem():
    """容器内存口径：usage / max_usage / limit / oom_kill（读不到就如实返回 None）。"""
    out = {}

    def _read_int(name):
        try:
            with open(os.path.join(CGROUP_MEM, name)) as fh:
                return int(fh.read().split()[0])
        except Exception:
            return None

    out["usage_bytes"] = _read_int("memory.usage_in_bytes")
    out["max_usage_bytes"] = _read_int("memory.max_usage_in_bytes")
    out["limit_bytes"] = _read_int("memory.limit_in_bytes")
    try:
        with open(os.path.join(CGROUP_MEM, "memory.oom_control")) as fh:
            for line in fh:
                if line.startswith("oom_kill "):
                    out["oom_kill"] = int(line.split()[1])
    except Exception:
        out["oom_kill"] = None
    try:
        with open(os.path.join(CGROUP_MEM, "memory.stat")) as fh:
            for line in fh:
                # ★ 必须区分 rss(匿名) 与 cache(页缓存)：`max_usage_in_bytes` **含页缓存**，
                #   只看它会把"读了大文件"误读成"进程吃光了内存"。
                if line.startswith("rss "):
                    out["rss_bytes"] = int(line.split()[1])
                elif line.startswith("total_rss "):
                    out["total_rss_bytes"] = int(line.split()[1])
                elif line.startswith("cache "):
                    out["cache_bytes"] = int(line.split()[1])
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    out["mem_available_kb"] = int(line.split()[1])
    except Exception:
        pass
    return out


def py_proc_count():
    """容器内 python 进程数 —— 用于识别"子进程爆炸"（每 worker 再起预处理池）。

    读 `/proc/<pid>/comm`（不依赖 ps/procps 是否安装）。
    """
    n = 0
    try:
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open("/proc/%s/comm" % pid) as fh:
                    if "python" in fh.read():
                        n += 1
            except OSError:
                continue
    except OSError:
        return None
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", default="/root/qwen35-ascend-migrator")
    ap.add_argument("--config", default="/root/qwen35-ascend-migrator/out/plan/train_config.yaml")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--single-die", action="store_true", default=False,
                    help="只有 1 块 die 时使用：world1/mbs8/gas1（GBS=8 不变，official_comparable=False）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="只跑消费侧前置守卫自检（坑 140）并退出")
    ap.add_argument("--mind", default="/root/MindSpeed-MM")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    if args.selftest:
        return beta_guard_selftest()
    if not args.out:
        print("FATAL 缺少 --out（只有 --selftest 时才可以不传）")
        return 2

    if os.path.exists(args.out):
        print("FATAL 输出目录已存在（拒绝覆盖）：%s" % args.out)
        return 3
    import yaml
    doc = yaml.safe_load(open(args.config, encoding="utf-8"))
    doc, world, _geom_official = apply_single_die(doc, args.single_die)
    # ★ 坑 146：**可比性不能只看几何**。"官方几何 + mock 数据"的实跑必须判为不可比 ——
    #   真机实测该组合的窗口中位数 710.55 ms vs 官方 431.3 ms（差 1.65 倍），
    #   旧实现却把 geometry.official_comparable 标成 True（over-claim）。
    #   判据取自 `_envcompat.official_comparable`（**唯一定义处**，与 P2 计划同源对比）。
    # ★ 本脚本自己的目录**优先**（这样把探针连同 `_envcompat.py` 一起放到 /root/ops 就能跑，
    #   不必去改动机器上已安装的 Skill）；Skill 的 scripts/ 作为回退。
    for _p in (os.path.join(args.skill, "scripts"), os.path.dirname(os.path.abspath(__file__))):
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)
    _plan_path = os.path.join(args.skill, "out", "plan", "train_config.yaml")
    try:
        from _envcompat import official_comparable as _oc
        official, _cdetails, _onote = _oc(doc, _plan_path, _geom_official)
    except Exception as _exc:      # ★ fail closed：判据加载不到就**不宣称可比**
        official = False
        _cdetails = {"error": type(_exc).__name__,
                     "plan": os.path.join(args.skill, "scripts")}
        _onote = ("无法加载同源可比性判据（%s）→ 保守判为不可比" % type(_exc).__name__)
    print("P61_COMPARABLE official_comparable=%s 依据=%s" % (official, _onote), flush=True)
    env_json = os.path.join(args.skill, "out", "probe", "env.json")

    # ★ 坑 140 消费侧守卫：`--config` 可以喂手工配置，那样会**绕过** P0/P2 按实测 CANN
    #   写入的规避开关 → 载入阶段就崩（AICPU 507018），而这看起来像"HCCL/设备坏了"。
    #   判据与生产侧同源：读 P0 产物 env.json 的档位，而不是看配置自己声称什么。
    glines, gerrs = beta_config_check(doc, env_json)
    for _ln in glines:
        print(_ln)
    if gerrs:
        for _e in gerrs:
            print("BETA_GUARD_FAIL %s" % _e)
        print("FATAL 前置未满足（坑 108/113/140）→ 拒绝启动（不浪费一次真机运行）")
        return 3
    print("BETA_GUARD_OK 环境档位与配置一致（来源：%s）" % env_json)
    tr = doc["training"]
    dp = int(doc["parallel"]["data_parallel_size"])
    gbs = int(tr["micro_batch_size"]) * int(tr["gradient_accumulation_steps"]) * dp
    if gbs != 8:
        print("FATAL 只做官方几何 GBS=8（当前 %d）" % gbs)
        return 3
    os.makedirs(args.out)
    cfg = json.loads(json.dumps(doc))
    cfg["training"]["train_iters"] = args.steps
    cfg["training"]["save"] = os.path.join(args.out, "checkpoint")
    config = os.path.join(args.out, "config.yaml")
    open(config, "w", encoding="utf-8").write(yaml.safe_dump(cfg, sort_keys=False))

    nid = npu_id()
    boot = open("/proc/sys/kernel/random/boot_id").read().strip()
    # ★ 坑 154：不要直接读 `/etc/hostname`（部分容器/非 Linux 上不存在，且这行在**记录落盘**处，
    #   会在跑完训练之后才炸）。用可移植的 stdlib `platform.node()`。
    host = __import__("platform").node() or "unknown"
    path0, stat0 = cpu_stat()
    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                h = hbm(nid)
                if h.get("hbm_usage_pct") is None:      # ★ 坑 137：npu-smi 不可用 → 回退 torch_npu
                    alt = hbm_via_torch()
                    if alt:
                        h = {"raw_first_line": "npu-smi 不可用 → 已回退 torch_npu", **alt}
                s = {"t": time.time(), "hbm": h}
            except Exception as e:
                s = {"t": time.time(), "error": str(e)[:100]}
            samples.append(s)
            stop.wait(args.interval)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    # ★ 坑 143：容器内存曲线用**独立线程 1 Hz** 采 —— 主采样线程被 `npu-smi` 拖到
    #   7~9 s 一次（见 actual_interval），用它看内存峰值的窗口太粗，会漏掉 OOM 瞬间。
    cg_samples, cg_stop = [], threading.Event()

    def cg_sampler():
        while not cg_stop.is_set():
            cg_samples.append({"t": time.time(), "n_py": py_proc_count(), **cgroup_mem()})
            cg_stop.wait(1.0)

    cg_th = threading.Thread(target=cg_sampler, daemon=True)
    cg_th.start()
    cg_before = cgroup_mem()

    log = os.path.join(args.out, "train.log")
    drv = os.path.join(args.out, "driver.log")
    cmd = [sys.executable, os.path.join(args.skill, "scripts", "50_train.py"),
           "--config", config, "--env", env_json, "--log", log, "--workdir", args.mind,
           "--steps", str(args.steps), "--world-size", str(world),
           "--timeout", "900", "--foreground"]
    print("P61_START out=%s steps=%d npu=%s boot=%s host=%s" % (args.out, args.steps, nid, boot, host), flush=True)
    t0 = time.time()
    with open(drv, "w", encoding="utf-8") as fh:
        r = subprocess.run(cmd, cwd=args.skill, env=os.environ.copy(),
                           stdout=fh, stderr=subprocess.STDOUT, timeout=args.timeout)
    wall = time.time() - t0
    stop.set()
    th.join(timeout=5)
    cg_stop.set()
    cg_th.join(timeout=5)
    cg_after = cgroup_mem()
    cg_usages = [s["usage_bytes"] for s in cg_samples if s.get("usage_bytes") is not None]
    cg_rss = [s["total_rss_bytes"] for s in cg_samples if s.get("total_rss_bytes") is not None]
    cg_procs = [s["n_py"] for s in cg_samples if s.get("n_py") is not None]
    cg_peak = max(cg_usages) if cg_usages else None
    limit = cg_after.get("limit_bytes")
    cg_oom_delta = (None if (cg_before.get("oom_kill") is None or cg_after.get("oom_kill") is None)
                    else cg_after["oom_kill"] - cg_before["oom_kill"])
    cgroup_rec = {
        "limit_bytes": limit,
        "usage_before_bytes": cg_before.get("usage_bytes"),
        "usage_after_bytes": cg_after.get("usage_bytes"),
        "peak_usage_sampled_bytes": cg_peak,
        "peak_pct_of_limit": (round(100.0 * cg_peak / limit, 1) if (cg_peak and limit) else None),
        "peak_total_rss_bytes": max(cg_rss) if cg_rss else None,
        "peak_total_rss_pct_of_limit": (round(100.0 * max(cg_rss) / limit, 1)
                                        if (cg_rss and limit) else None),
        "peak_python_procs": max(cg_procs) if cg_procs else None,
        "oom_kill_before": cg_before.get("oom_kill"),
        "oom_kill_after": cg_after.get("oom_kill"),
        "oom_kill_delta": cg_oom_delta,
        "n_samples": len(cg_samples),
        "mem_available_kb_after": cg_after.get("mem_available_kb"),
    }
    print("P61_CGROUP limit=%s peak_usage=%s (%.1f%%) peak_rss=%s peak_py_procs=%s "
          "oom_kill_delta=%s n=%d"
          % (limit, cg_peak, cgroup_rec["peak_pct_of_limit"] or 0.0,
             cgroup_rec["peak_total_rss_bytes"], cgroup_rec["peak_python_procs"],
             cg_oom_delta, len(cg_samples)), flush=True)
    path1, stat1 = cpu_stat()

    dtext = open(drv, encoding="utf-8", errors="replace").read()
    ttext = open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else ""
    rows = [(int(i), float(ms)) for i, ms in PAT.findall(ttext)]
    train_rc = re.findall(r"train_rc=(\d+)", dtext)
    # ★ 复核意见（2026-09-21）：**59 检查完整步号，61 原先只检查解析行数** ——
    #   同一次运行会得到不同的 `valid`。现统一走共享校验器 `_envcompat.run_valid`
    #   （判据只留一处实现）。fail closed：拿不到判据 ⇒ 该次运行判为无效。
    _ids = [i for i, _ms in rows]
    try:
        from _envcompat import run_valid as _rv61
        valid, steps_complete = _rv61(r.returncode, train_rc, _ids, args.steps)
        _valid_judge = "shared"
    except Exception as _exc:      # pragma: no cover
        valid, steps_complete, _valid_judge = False, False, "unavailable:%s" % type(_exc).__name__
    print("P61_VALID valid=%s steps_complete=%s judge=%s ids=%d/%d"
          % (valid, steps_complete, _valid_judge, len(_ids), args.steps), flush=True)
    win = [ms for i, ms in rows if 11 <= i <= args.steps]
    thr = {k: stat1.get(k, 0) - stat0.get(k, 0) for k in ("nr_periods", "nr_throttled", "throttled_time")}
    pcts = [s["hbm"]["hbm_usage_pct"] for s in samples if s.get("hbm", {}).get("hbm_usage_pct") is not None]
    # ★ 复核修正 1：`allocated:` 与 `max allocated:` 在同一行 —— 必须分开解析，
    #   否则"当前值"和"峰值"会被当成两次观测（原实现的正则就犯了这个错）。
    cur_alloc = [float(x) for x in re.findall(r"(?<!max )allocated:\s*([\d.]+)", ttext)]
    max_alloc = [float(x) for x in re.findall(r"max allocated:\s*([\d.]+)", ttext)]
    cur_resv = [float(x) for x in re.findall(r"(?<!max )reserved:\s*([\d.]+)", ttext)]
    max_resv = [float(x) for x in re.findall(r"max reserved:\s*([\d.]+)", ttext)]
    # ★ 复核修正 2：实际采样间隔与声明值可能差一个数量级，必须如实统计
    ts = [s["t"] for s in samples if "t" in s]
    gaps = [round(b - a, 3) for a, b in zip(ts, ts[1:])]
    actual = ({"n_intervals": len(gaps), "min_s": min(gaps), "median_s": round(statistics.median(gaps), 3),
               "max_s": max(gaps)} if gaps else {"n_intervals": 0})
    # ★ 复核修正 3：无变化 = 正对照缺失，必须显式标出（不能读成"内存很稳"）
    variation = (max(pcts) - min(pcts)) if len(pcts) > 1 else 0.0
    positive_control = ("no_variation_observed" if variation == 0.0 else "variation_observed")

    rec = {"schema": "mem_comm_probe.v1", "valid": valid, "driver_rc": r.returncode,
           "steps_complete": steps_complete, "valid_judge": _valid_judge,
           "step_ids_tail": _ids[-5:],
           "train_rc": train_rc, "steps_parsed": len(rows), "wall_seconds": round(wall, 1),
           "geometry": {"mbs": tr["micro_batch_size"], "gas": tr["gradient_accumulation_steps"],
                        "dp": dp, "gbs": gbs, "official_comparable": official},
           # ★ 只放**证据**，不重复那个布尔值（避免"同一语义两个来源"）：判据见 geometry.official_comparable
           "comparability": {"reason": _onote, "details": _cdetails,
                             "is_official_geometry": _geom_official,
                             "note": ("★ 三个独立结论：plan_consistent / ab_comparable / "
                                      "official_comparable（复核意见 2026-09-21）。"
                                      "「本机没有可对标官方的数据」不等于「没有有效的同机 "
                                      "A/B 证据」—— 前者 False 不妨碍后者成立")},
           "window_11_end_median_ms": statistics.median(win) if win else None,
           "identity": {"boot": boot, "hostname": host, "npu_id": nid,
                        "affinity": sorted(os.sched_getaffinity(0))},
           "hbm_usage_pct": {"samples": len(pcts), "min": min(pcts) if pcts else None,
                             "median": statistics.median(pcts) if pcts else None,
                             "max": max(pcts) if pcts else None,
                             "positive_control": positive_control,
                             "caveat": ("采样间隔见 sampling.actual_interval；若与步长同量级会混叠，"
                                        "只看峰值与趋势")},
           "cgroup_delta": {"path": path0, **thr},
           "cgroup_memory": {**cgroup_rec,
                             "note": ("★ 坑 143：torch 把 worker 被杀提示成 "
                                      "`insufficient shared memory (shm)`，但本机 /dev/shm 用量 0% —— "
                                      "**报错文案点名了错误的资源**。真正的判据是这里的 "
                                      "cgroup 计数：peak 撞 limit + oom_kill 增量 > 0。"
                                      "`peak_pct_of_limit` 为 None 时表示 cgroup 口径读不到，"
                                      "**不得据此判断内存充足**")},
           "device_level_hbm": {"source": "torch_npu.mem_get_info / npu-smi（**设备级**，含其他进程）",
                                 "samples": len(pcts),
                                 "note": "设备级读数的正对照：%s；无变化时不得读成「内存很稳」"
                                         % positive_control},
           "process_level_allocator": {
               "source": "train.log 中 torch 打印（**训练进程内**，但只在少数节点打印）",
               "current_allocated_mb": cur_alloc, "max_allocated_mb": max_alloc,
               "current_reserved_mb": cur_resv, "max_reserved_mb": max_resv,
               "note": ("旁路进程的 allocated/reserved **不代表训练进程**；"
                        "逐帧分配器曲线必须**在训练进程内**采（hook），本次未做 ⇒ 不据此判断稳态显存")},
           "sampling": {"declared_interval_s": args.interval,
                        "actual_interval": actual,
                        "warning": ("声明的 %s s 与实际不符是**回退路径的固有开销**："
                                    "每次采样新起进程并初始化 torch/NPU。"
                                    "此类数据只能作**量级参考**，不得当时间序列用于峰值定位。"
                                    % args.interval) if actual.get("n_intervals", 0) > 3
                                   and (actual.get("median_s") or 0) > 3 * args.interval else None},
           "samples": samples}
    json.dump(rec, open(os.path.join(args.out, "result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("P61_RESULT valid=%s steps=%s wall=%.0fs  HBM%% min/med/max=%s/%s/%s  "
          "节流增量 nr_throttled=%s throttled_time=%s"
          % (valid, len(rows), wall, rec["hbm_usage_pct"]["min"], rec["hbm_usage_pct"]["median"],
             rec["hbm_usage_pct"]["max"], thr["nr_throttled"], thr["throttled_time"]), flush=True)
    print("P61_RESULT cgroup peak=%s/%s (%.1f%%) oom_kill_delta=%s"
          % (cg_peak, limit, cgroup_rec["peak_pct_of_limit"] or 0.0,
             cgroup_rec["oom_kill_delta"]), flush=True)
    print("P61_%s out=%s" % ("OK" if valid else "FAIL", args.out), flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
