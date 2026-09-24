#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""59_config_ab.py — 通用「单键 A/B/A」真实训练消融（性能用）

为什么要有它
------------
`57_fsdp_group_ablation.py` 只针对 fsdp 分组；但我们要试的候选都是**单个配置键**：
  · `training.adam_fused: true → false`（现有 true 等价 foreach=False=逐参数；见 Codex 的只读复核）
  · `data.dataset_param.basic_parameters.preprocess_on_fly: true → false`（数据预处理是否在关键路径）
与其一稿一脚本，不如一个**通用**工具：只改一个点号路径的键，其余逐字节不动。

判据（跑之前已定，写在脚本里，事后不得改）
------------------------------------------
* 三组 A/B/A **同一 boot**、背靠背；窗口 11–N 中位；
* 两次基线 A 的相对漂移 **> 3%** → **UNCERTAIN**（不下结论）；
* `B/A ≤ 0.95` → **WIN（≥5%）**；`0.95 < B/A ≤ 1.00` → **SMALL_GAIN（未过门槛，只能当观察）**；`>1.00` → **NO_GAIN**；
* 三组都必须有退出证据（driver_rc=0 且 train_rc=['0'] 且解析步数==N）；
* **逐点核验**：step1 loss 必须仍为官方锚点 1.924621（偏差 <1e-6）；
  B 与 A 的 loss/grad_norm 逐点相对差必须 <2%（否则说明改动动了数值，需单独评估）。

红线
----
* 官方几何（mbs4/gas1/dp2 → GBS=8）**不变**，不加 `--allow-gbs-mismatch`；
* 只改一个键，回读校验；**不碰** `examples/judge` 与 `sk04_judge`；
* 结果只能用于"性能"讨论，**不得**据此宣称精度达标（裁决是黄灯，两道门按构造未闭合）。

用法：bash scripts/p59_launch.sh /root/ops/<新目录> <点号键> <A值> <B值>
"""
import argparse
import copy
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time

PAT = re.compile(r"iteration\s+(\d+)\s*/\s*\d+.*?"
                 r"elapsed time per iteration \(ms\):\s*([\d.]+).*?"
                 r"loss:\s*([\d.eE+-]+).*?"
                 r"grad norm:\s*([\d.eE+-]+)", re.S)
STEP1_ANCHOR = 1.924621
ORDER = "A,B,A"


def get_key(doc, dotted):
    cur = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError("配置里没有键：%s（在 %s 处断了）" % (dotted, part))
        cur = cur[part]
    return cur


def set_key(doc, dotted, value):
    parts = dotted.split(".")
    cur = doc
    for part in parts[:-1]:
        cur = cur[part]
    cur[parts[-1]] = value


def parse_value(text, like):
    """按原值类型严格解析命令行文本。

    ★ 复核发现的缺陷：原实现 `return low == "true"` 会把任何非 "true" 的文本（例如 `typo`）
    **静默解释成 False** —— 一个拼错的值会变成"合法输入"，直接把实验做成另一个配置。
    现改为：布尔只接受 true/false（大小写不敏感）；其余非法输入**直接报错**。
    """
    low = str(text).strip().lower()
    if isinstance(like, bool):
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise ValueError("布尔键只接受 true/false，收到 %r（拒绝静默取 False）" % text)
    if low in ("true", "false"):
        raise ValueError("目标键现值是 %r 而非布尔，不能用 true/false 赋值" % like)
    if isinstance(like, int) and not isinstance(like, bool):
        return int(str(text).strip())
    if isinstance(like, float):
        return float(str(text).strip())
    return str(text)


def compute_valid(returncode, train_rc, ids, steps):
    """单组实跑的**有效性判据** —— 实现已上移到共享校验器 `_envcompat.run_valid`。

    ★ 复核意见（2026-09-21）：59 与 61 必须对**同一次运行**给出同一个 `valid`，
      因此判据只留**一处**实现，本函数只做转发。
    ★ fail closed：共享校验器加载不到就返回 `(False, False)`（该组判无效），
      **不**在本地复制一份实现 —— 复制就是"同一语义两个来源"（坑 111 同族）。

    保留本函数名是为了让 `--selftest` 继续覆盖"被判的东西"（坑 147 的教训）。
    """
    try:
        from _envcompat import run_valid as _rv
    except Exception:
        return False, False
    return _rv(returncode, train_rc, ids, steps)


def run_one(tag, variant, key, value, doc0, args, boot):
    import yaml
    case = os.path.join(args.out, tag)
    os.makedirs(case, exist_ok=False)
    cfg = copy.deepcopy(doc0)
    if variant == "B":
        set_key(cfg, key, value)
    cfg["training"]["train_iters"] = args.steps
    cfg["training"]["save"] = os.path.join(case, "checkpoint")
    text = yaml.safe_dump(cfg, sort_keys=False)
    back = yaml.safe_load(text)
    gbs = (int(back["training"]["micro_batch_size"]) *
           int(back["training"]["gradient_accumulation_steps"]) *
           int(back["parallel"]["data_parallel_size"]))
    assert gbs == 8, "只做官方几何 GBS=8，当前 %d" % gbs
    # 单变量校验（★ 复核意见：改用**共享校验器** `_envcompat.ab_comparable`）——
    #   A/B 可比性是三个独立结论里的第二个，必须与 plan_consistent / official_comparable
    #   **分开**下结论；判据只留一处实现，避免"同一语义多个来源"。
    #   它同时覆盖"除登记变量外还有别的差异"（旧实现用还原+整体 JSON 比较，语义相同但只在 59 里）。
    _here = os.path.dirname(os.path.abspath(__file__))
    for _p in (os.path.join(args.skill, "scripts"), _here):
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)
    try:
        from _envcompat import ab_comparable as _ab_cmp
        single_var, sv_details = _ab_cmp(doc0, back, key)
    except Exception as _exc:      # fail closed：判据加载不到就判"不可比"
        single_var, sv_details = False, {"error": type(_exc).__name__, "diffs": []}
    config = os.path.join(case, "config.yaml")
    open(config, "w", encoding="utf-8").write(text)

    log, drv = os.path.join(case, "train.log"), os.path.join(case, "driver.log")
    env_json = os.path.join(args.skill, "out", "probe", "env.json")
    cmd = [sys.executable, os.path.join(args.skill, "scripts", "50_train.py"),
           "--config", config, "--env", env_json, "--log", log, "--workdir", args.mind,
           "--steps", str(args.steps), "--world-size", str(int(back["parallel"]["data_parallel_size"])),
           "--timeout", "900", "--foreground"]
    t0 = time.time()
    with open(drv, "w", encoding="utf-8") as fh:
        r = subprocess.run(cmd, cwd=args.skill, env=os.environ.copy(),
                           stdout=fh, stderr=subprocess.STDOUT, timeout=args.timeout)
    wall = time.time() - t0
    dtext = open(drv, encoding="utf-8", errors="replace").read()
    ttext = open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else ""
    rows = [(int(i), float(ms), float(lo), float(gn)) for i, ms, lo, gn in PAT.findall(ttext)]
    train_rc = re.findall(r"train_rc=(\d+)", dtext)
    # ★ 复核要求：缺一步 / 重复步号都必须被识别（不能只看"步数对不对"）。
    #   ★★ 真机教训（坑 147）：这段判据**原先写成内联表达式**，其中 `steps_complete`
    #   在**定义之前**被使用 → 真机一跑就 `UnboundLocalError: steps_complete`，
    #   而 `P59_SELFTEST_OK cases=16` 全绿 —— 因为**自检根本不走 `run_one`**。
    #   处置：把判据抽成纯函数 `compute_valid()`，并让自检直接覆盖它。
    ids = [i for i, _, _, _ in rows]
    valid, steps_complete = compute_valid(r.returncode, train_rc, ids, args.steps)
    win = [ms for i, ms, _, _ in rows if 11 <= i <= args.steps]
    rec = {"tag": tag, "variant": variant, "key": key, "steps_complete": steps_complete,
           "step_ids_tail": ids[-5:],
           "value": (value if variant == "B" else get_key(doc0, key)),
           "single_variable_ok": single_var, "valid": valid,
           "single_variable_details": sv_details,
           "driver_rc": r.returncode, "train_rc": train_rc, "steps_parsed": len(rows),
           "wall_seconds": round(wall, 1), "window": [11, args.steps],
           "window_median_ms": statistics.median(win) if win else None,
           "iter_ms": [ms for _, ms, _, _ in rows],
           "loss": [lo for _, _, lo, _ in rows], "grad_norm": [gn for _, _, _, gn in rows],
           "config_sha256": hashlib.sha256(open(config, "rb").read()).hexdigest(),
           "source_sha256": hashlib.sha256(
               open(os.path.join(args.skill, "scripts", "50_train.py"), "rb").read()).hexdigest(),
           # ★ 坑 154：原来直接 `open("/etc/hostname")` —— 该文件在部分容器/非 Linux 上不存在，
           #   而这一行位于**记录落盘**处，于是会在**跑完一整个训练之后**才炸掉（白烧一次实跑）。
           #   改用可移植的 stdlib `platform.node()`（与 `_envcompat.safe_hostname()` 同义；
           #   这里用 stdlib 以免为一行身份字段引入路径/导入依赖）。
           "identity": {"boot": boot,
                        "hostname": __import__("platform").node() or "unknown"}}
    json.dump(rec, open(os.path.join(case, "result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("END %s variant=%s valid=%s single_var=%s win_med=%s wall=%.0fs"
          % (tag, variant, valid, single_var, rec["window_median_ms"], wall), flush=True)
    return rec


def pointwise(a, b, max_step=None):
    """逐点相对差（%）：loss / grad_norm。

    ★ 复核发现的缺陷：原实现 `... if x[i] else 0.0` 把**零分母**吞成 0% ——
    基线为 0、候选为 1 时会报告"误差 0%"，是最危险的一类假阴性。
    现改为：零分母**不计入统计**并单独计数；全部为 0 分母时 `max_pct` 记 None（不可判定）。

    ★★ `max_step`（2026-09-21，真机实测后的设计修正）：**只对前若干步做逐点比较**。
    证据：同配置两次跑（`A_w2/00_NULL` vs `01_NULL`）
      · 步 1/2 的 loss 打印值完全相同；但 `grad_norm` 第 1 步已是 A=35.185 / B=35.176
        （相对差 **2.6e-4** ≫ fp32 的 1e-7）⇒ **差异从第 1 步起就存在**，只是被 6 位打印掩盖；
      · 步 3 起被优化器**混沌放大**：loss 相对差 1.14%(步3) → 18.6%(步6)，最大绝对差 0.0457(步15)。
    ⇒ 步 ≥3 的逐点 loss **对同一配置的两次跑都不成立**，拿它当门槛只会**恒判 REJECT_NUMERIC**
      （`pregather` 那轮就是这么被拒的），而拒绝理由与被测键无关。
      **只比前几步即可完整保留判据的力量**：配置若真改动了数学，第 1 步就会变。
    """
    out = {}
    for name, x, y in (("loss", a["loss"], b["loss"]), ("grad_norm", a["grad_norm"], b["grad_norm"])):
        # ★ 复核要求：**长度不一致不得静默按前缀比较** —— 显式记为不可比
        length_ok = (len(x) == len(y))
        n = len(x) if length_ok else 0
        n = min(n, max_step) if max_step else n
        rel, undef = [], 0
        for i in range(n):
            if abs(x[i]) < 1e-12:
                undef += 1
                continue
            rel.append(abs(y[i] - x[i]) / abs(x[i]) * 100)
        out[name] = {"len_a": len(x), "len_b": len(y), "length_ok": length_ok, "n": n,
                     "compared": len(rel), "undefined_zero_baseline": undef,
                     "max_pct": round(max(rel), 4) if rel else None,
                     "mean_pct": round(sum(rel) / len(rel), 4) if rel else None}
    return out


T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def pointwise_all(A, B, max_step=None):
    """★ 复核意见（2026-09-21）：逐点数值检查必须覆盖**全部配对**，不能只看第一组 A/B。

    把各对同名字段按「最坏情形」合并，于是 `verdict_of()` 里那段数值检查对**每一对**都生效：
      · `max_pct` 取各对最大（任一超限即整体超限）
      · `length_ok` 取逻辑与（任一对长度不一致即整体不合格）
      · `undefined_zero_baseline` 累加、`compared` 取最小
    返回 `(merged, per_pair)`；`per_pair` 作为证据一并落盘。
    `max_step` 见 `pointwise()` 的说明（只比可比的步，避免混沌步把判据压成恒拒绝）。
    """
    per = [pointwise(a, b, max_step=max_step) for a, b in zip(A, B)]
    if not per:
        return {}, []
    merged = {}
    for k in per[0]:
        worst = dict(per[0].get(k, {}))
        for p in per[1:]:
            v = p.get(k, {})
            if v.get("length_ok", True) is False:
                worst["length_ok"] = False
                worst["len_a"], worst["len_b"] = v.get("len_a"), v.get("len_b")
            got = [x for x in (worst.get("max_pct"), v.get("max_pct")) if x is not None]
            worst["max_pct"] = max(got) if got else None
            worst["undefined_zero_baseline"] = (worst.get("undefined_zero_baseline", 0)
                                               + v.get("undefined_zero_baseline", 0))
            worst["compared"] = min(worst.get("compared", 0), v.get("compared", 0))
        merged[k] = worst
    return merged, per


def win_ms_of(rec):
    """★ 取"窗口中位步时长"。**两个字段名都要认**，且**都不在时必须报错**。

    现场 bug（2026-09-21）：`63_two_config_ab.py` 复用本模块的 `paired_stats()`，而两组 run 记录
    的字段名不同 —— 59 的 `run_one` 写 `window_median_ms`，63 的写 `window_11_end_median_ms`
    ⇒ 63 的 A/B **六组全部跑完且 valid=True**，却在统计那一步 `KeyError: 'window_median_ms'`
    整体报废（性能数据白跑，只留下 results.json 里的一堆原始记录）。

    这是本项目反复出现的同一类错误：**字段名是断言而不是事实**（坑 163/176 同族），而且这次
    发生在**"复用"这条路上** —— 复用别人的函数时，我默认了"大家记录形状一样"。
    处置：① 两个名字都认；② **都没有就 raise**（绝不返回 None/0 —— 那会让下游算出一个假的性能差）；
    ③ 自检补一条"用 63 形状的记录喂进 paired_stats"，把这条路纳入闸门。
    """
    for k in ("window_11_end_median_ms", "window_median_ms"):
        v = rec.get(k)
        if v is not None:
            return float(v)
    raise KeyError("run %s 既没有 window_11_end_median_ms 也没有 window_median_ms"
                   "（实际字段：%s）" % (rec.get("tag"), sorted(rec.keys())))


def paired_stats(A, B, order_list):
    """按**索引配对**做成对统计：pairs = (A[0],B[0]), (A[1],B[1]), ...

    平衡顺序（如 A,B,B,A ⇒ A=[第1组A, 第4组A]、B=[第2组B, 第3组B]）下，
    线性漂移在成对差里相互抵消 —— 这正是"平衡设计"的意义。
    """
    n = min(len(A), len(B))
    diffs = []
    for i in range(n):
        av, bv = win_ms_of(A[i]), win_ms_of(B[i])
        if av is None or bv is None:
            continue
        diffs.append({"pair": i + 1, "A_ms": av, "B_ms": bv,
                      "diff_ms": round(bv - av, 3),
                      "delta_pct": round((bv - av) / av * 100, 3)})
    if not diffs:
        return {"pairs": 0, "diffs": [], "mean_delta_pct": None, "ci95": None}
    vals = [d["delta_pct"] for d in diffs]
    mean = sum(vals) / len(vals)
    if len(vals) > 1:
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        sd = var ** 0.5
        tcrit = T975.get(len(vals) - 1, 1.96)
        half = tcrit * sd / (len(vals) ** 0.5)
        ci = [round(mean - half, 3), round(mean + half, 3)]
    else:
        sd, ci = None, None
    return {"pairs": len(diffs), "diffs": diffs, "mean_delta_pct": round(mean, 3),
            "sd_pct": round(sd, 3) if sd is not None else None,
            "t_crit": T975.get(len(vals) - 1, 1.96) if len(vals) > 1 else None,
            "ci95_delta_pct": ci,
            "improvement_pct": round(-mean, 3),
            "note": "负的 delta 表示 B 更快；改善率 = -mean"}


def verdict_of(A, B, pairs, pw, official, tol_pct, drift_pct):
    """分级的裁决：先把"无效/数值不合格"与"证据不足/无收益"分开，避免互相遮掩。"""
    if not A or not B:
        return "INVALID", "缺少 A 或 B 组"
    if any(r.get("valid") is not True for r in A + B):
        return "INVALID", "存在无退出证据的运行（valid != True）"
    if any(r.get("steps_complete") is False for r in A + B):
        bad = [r["tag"] for r in A + B if r.get("steps_complete") is False]
        return "INVALID", "步号不连续/有重复：%s" % bad
    if any(r.get("single_variable_ok") is not True for r in A + B):
        return "INVALID", "存在非单变量运行"
    boots = {r.get("identity", {}).get("boot") for r in A + B}
    if len(boots) != 1:
        return "INVALID", "跨 identity.boot 混算（%s）" % sorted(str(b) for b in boots)
    # 数值不合格**必须**先于任何性能结论报出（复核意见：不能被漂移遮住）
    for k, v in pw.items():
        if not v.get("length_ok", True):
            return "REJECT_NUMERIC", ("%s 序列长度不一致（A=%d / B=%d）→ 不可比，"
                                      "拒绝按前缀比较" % (k, v["len_a"], v["len_b"]))
        if v["undefined_zero_baseline"] and v["compared"] == 0:
            return "REJECT_NUMERIC", "%s 全部逐点差不可判定（零分母 %d 处）" % (k, v["undefined_zero_baseline"])
        if v["max_pct"] is not None and v["max_pct"] >= tol_pct:
            return "REJECT_NUMERIC", "%s 逐点最大差 %.4f%% ≥ %.2f%%" % (k, v["max_pct"], tol_pct)
    if pairs["pairs"] < 2:
        return "INSUFFICIENT_EVIDENCE", "只有 %d 对配对比较（至少 2 对才谈区间）" % pairs["pairs"]
    ci, mean = pairs["ci95_delta_pct"], pairs["mean_delta_pct"]
    if ci is None or mean is None:
        return "INSUFFICIENT_EVIDENCE", "配对区间不可计算"
    # ★★ 复核意见（2026-09-21）：**不能用点估计与 5% 比**。
    #   旧实现是 `if -mean >= 5.0: WIN` —— 于是"均值 +6%、区间 [1%, 11%]"被判 WIN，
    #   但那个结果只支持"存在正收益、点估计 6%"，**不支持"收益至少 5%"**。
    #   要宣称"至少 5%"，必须让**改善区间的下界 ≥ 5%**（改善量区间 = [-ci[1], -ci[0]]）。
    imp_lo, imp_hi = -ci[1], -ci[0]
    if ci[0] <= 0 <= ci[1]:
        return "UNCERTAIN", ("配对差 95%% 区间 [%.3f, %.3f] 跨 0（含基线漂移 %.2f%%）"
                             % (ci[0], ci[1], drift_pct))
    if ci[1] < 0:                       # B 显著更快
        if imp_lo >= 5.0:
            return "WIN", ("改善 %.2f%%（95%% 区间 [%.2f, %.2f]）**区间下界 ≥ 5%%** ⇒ "
                           "在声明的实验条件与统计假设下支持「至少 5%%」" % (-mean, imp_lo, imp_hi))
        return "SMALL_GAIN", ("有正收益：改善 %.2f%%（95%% 区间 [%.2f, %.2f]），"
                              "但**下界 < 5%%** ⇒ 不能保证达到 5%%" % (-mean, imp_lo, imp_hi))
    return "NO_GAIN", ("B 显著更慢：配对差 +%.2f%%（区间 [%.3f, %.3f] 不含 0）"
                       % (mean, ci[0], ci[1]))


def apply_single_die(doc, single):
    """★ 单 die 受限环境：保留 GBS=8 红线，但把几何改成 world1/mbs8/gas1。

    目的与边界（写进产物，避免误用）：
      * 只有**一块 die** 时无法复刻官方 dp2 几何；此时用 mbs=8/gas=1/world=1 仍满足
        `GBS = mbs x gas x world = 8`（红线不破），可继续做**发射/调用点归因**与
        **与并行无关的配置 A/B**；
      * 但 **official_comparable = False**：与官方几何不同（每 rank micro-batch 8 vs 4、
        无 FSDP 通信），**不得**用于"优于官方"或通信类结论。
    """
    if not single:
        return doc, int(doc["parallel"]["data_parallel_size"]), True
    doc["parallel"]["data_parallel_size"] = 1
    tr = doc["training"]
    tr["micro_batch_size"] = 8
    tr["gradient_accumulation_steps"] = 1
    return doc, 1, False


def null_test(args, doc0, boot):
    """★ 零假设实验：同一配置重复 N 次，量出**单机噪声下限**（= 这把尺子能分辨多少）。

    为什么必须先做：单卡会话实测两次基线差 3.06%，而候选改动预期只有几个百分点 ⇒
    **先证明尺子能量，再用尺子量东西**（对应计划第 1 步"先保证实验能判断结果"）。
    """
    import yaml
    runs = []
    for i in range(args.null_test):
        case = os.path.join(args.out, "%02d_NULL" % i)
        os.makedirs(case, exist_ok=False)
        cfg = copy.deepcopy(doc0)
        cfg["training"]["train_iters"] = args.steps
        cfg["training"]["save"] = os.path.join(case, "checkpoint")
        config = os.path.join(case, "config.yaml")
        open(config, "w", encoding="utf-8").write(yaml.safe_dump(cfg, sort_keys=False))
        log, drv = os.path.join(case, "train.log"), os.path.join(case, "driver.log")
        env_json = os.path.join(args.skill, "out", "probe", "env.json")
        cmd = [sys.executable, os.path.join(args.skill, "scripts", "50_train.py"),
               "--config", config, "--env", env_json, "--log", log, "--workdir", args.mind,
               "--steps", str(args.steps),
               "--world-size", str(int(cfg["parallel"]["data_parallel_size"])),
               "--timeout", "900", "--foreground"]
        t0 = time.time()
        with open(drv, "w", encoding="utf-8") as fh:
            r = subprocess.run(cmd, cwd=args.skill, env=os.environ.copy(),
                               stdout=fh, stderr=subprocess.STDOUT, timeout=args.timeout)
        wall = time.time() - t0
        dtext = open(drv, encoding="utf-8", errors="replace").read()
        ttext = open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else ""
        # ★ 判据正则与 `run_one` **同一条**（`PAT`）—— 同一语义只留一处实现
        rows = [(int(a), float(b), float(c), float(d)) for a, b, c, d in PAT.findall(ttext)]
        train_rc = re.findall(r"train_rc=(\d+)", dtext)
        # ★ 坑 153（真机实测，2026-09-21）：这里原先写的是 `... and steps_complete`，
        #   而 `null_test()` **从来没有定义过 `steps_complete`** → 真机
        #   `NameError: name 'steps_complete' is not defined`（pregather 队列的第一步就死在这）。
        #   根因与坑 147 **一模一样**（判据里引用了不存在的名字），只是发生在
        #   **另一个函数**：我上一轮只修了 `run_one()`，而自检覆盖的是纯函数、
        #   不是编排路径 —— 复核当时就提醒过"尚未覆盖完整编排路径"，这就是那个缺口。
        #   处置：改用共享校验器 `compute_valid()`（同一判据、同一实现），
        #   并新增 `--selftest-orchestration` 用桩子进程真跑 null_test()/run_one()。
        _ids = [i for i, _ms, _lo, _gn in rows]
        valid, steps_complete = compute_valid(r.returncode, train_rc, _ids, args.steps)
        win = [ms for i, ms, _lo, _gn in rows if 11 <= i <= args.steps]
        rec = {"tag": "%02d_NULL" % i, "valid": valid, "steps_complete": steps_complete,
               "step_ids_tail": _ids[-5:], "window_median_ms":
               statistics.median(win) if win else None, "wall_seconds": round(wall, 1),
               "iter_ms": [ms for _i, ms, _lo, _gn in rows],
               "loss": [lo for _i, _ms, lo, _gn in rows],
               "grad_norm": [gn for _i, _ms, _lo, gn in rows],
               "identity": {"boot": boot}}
        json.dump(rec, open(os.path.join(case, "result.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        runs.append(rec)
        print("NULL %s valid=%s win_med=%s wall=%.0fs" % (rec["tag"], valid, rec["window_median_ms"], wall),
              flush=True)
        if not valid:
            print("P59_FAIL null-test 有无效运行，停止", flush=True)
            return 1
    # ★★ 正对照：数值判据的**前置条件** —— 同一配置重复跑，loss 序列必须逐点相同。
    #   复核要求"全部重复运行都要通过数值检查"；但如果**A/A 之间 loss 就已经不同**，
    #   那么任何 A/B 的逐点数值检查都必然 `REJECT_NUMERIC`，而且**拒绝的理由与被测键无关**。
    #   真机 pregather 那轮就是这样：`REJECT_NUMERIC（loss 逐点最大差 304.45%）`，
    #   而 6 组里 **A 与 A 的差（如 0.070691 vs 0.067264）≈ A 与 B 的差** ——
    #   差异根本不是 `pregather` 造成的，是**每步吃到的数据不同**造成的
    #   （`BaseRandomBatchSampler` + 多 dataloader worker ⇒ 全局批次组合在进程间不确定）。
    #   ⇒ 零假设实验因此必须回答两件事：**时间**能分辨多少 + **数值**是否可比。
    losses = [r.get("loss") or [] for r in runs]
    REL_FLOOR = 0.05      # ★ 相对差只在 loss ≥ 0.05 的步上统计（否则小分母会把结果放大到无意义）
    numeric_identical, first_div, worst_rel = True, None, 0.0
    worst_abs, first_div_abs, first_div_rel = 0.0, None, None
    worst_rel_floored = 0.0
    if len(losses) > 1 and losses[0]:
        base = losses[0]
        for ls in losses[1:]:
            for idx in range(min(len(base), len(ls))):
                b, v = base[idx], ls[idx]
                if b == v:
                    continue
                numeric_identical = False
                adiff = abs(b - v)
                rel = (adiff / abs(b) * 100.0) if b else float("inf")
                worst_abs = max(worst_abs, adiff)
                worst_rel = max(worst_rel, rel)
                if abs(b) >= REL_FLOOR:
                    worst_rel_floored = max(worst_rel_floored, rel)
                if first_div is None:
                    first_div = idx + 1
                    first_div_abs, first_div_rel = adiff, rel
    # ★ 为什么要这三个数：只看 `worst_rel_pct` 会**被末段小分母放大**（loss≈0.003 时
    #   0.003 的绝对差就是 100%），于是"分叉极大"的结论可能纯属度量假象。
    #   真正的判别量是**首次分叉那一步的绝对差**：
    #     ~1e-6 量级 → 浮点噪声被放大（换确定性算子/use_deter_comp 可解）
    #     明显更大    → 结构性差异（数据/权重/调度），不是"数值噪声"
    print("P59NULL_NUMERIC first_div_step=%s first_div_abs=%.3e first_div_rel_pct=%.4f "
          "worst_abs=%.3e worst_rel_pct=%.4f worst_rel_pct_loss>=%.2f=%.4f"
          % (first_div, first_div_abs if first_div_abs is not None else 0.0,
             first_div_rel if first_div_rel is not None else 0.0,
             worst_abs, worst_rel, REL_FLOOR, worst_rel_floored), flush=True)
    print("P59NULL_NUMERIC identical=%s first_diverging_step=%s worst_rel_pct=%.4f"
          % (numeric_identical, first_div, round(worst_rel, 4)), flush=True)
    if not numeric_identical:
        print("P59NULL_NUMERIC_WARN 同配置重复跑的 loss 已经不同 ⇒ **逐点数值判据在此数据路径下不可用**；"
              "任何 A/B 的数值拒绝都**不能**归因于被测键。"
              "处置：让 A/B 走确定性数据路径（例如 num_workers=1）后重做零假设实验，"
              "数值逐点相同后再做 A/B。", flush=True)

    # ★ 坑 154 附带发现：窗口是 11..steps，若 `--steps < 11` 则**窗口为空** →
    #   `meds` 全是 None → `max(meds)` 抛 `TypeError: '>' not supported between NoneType`。
    #   判据算不出来就必须**明说算不出来**，而不是抛一个与原因无关的 TypeError。
    meds = [r["window_median_ms"] for r in runs if r.get("window_median_ms") is not None]
    if len(meds) < 2:
        print("P59_FAIL null-test 无法计算散布：有效窗口中位数只有 %d 个"
              "（窗口是 11..steps，--steps=%d 时为空；请用 --steps >= 12）"
              % (len(meds), args.steps), flush=True)
        return 1
    spread = (max(meds) - min(meds)) / min(meds) * 100
    out = {"schema": "p59null.v1", "runs": len(meds), "medians_ms": meds,
           "mean_ms": round(sum(meds) / len(meds), 2), "min_ms": min(meds), "max_ms": max(meds),
           "spread_pct": round(spread, 2),
           # ★ 数值正对照：A/A 逐点相同才有资格对 A/B 用逐点数值判据
           "numeric_identical": numeric_identical,
           "numeric_first_diverging_step": first_div,
           "numeric_worst_rel_pct": round(worst_rel, 4),
           # ★ 判别量：首次分叉那一步的**绝对差**。
           #   ~1e-6 → 浮点噪声被逐步放大；明显更大 → 结构性差异。
           #   只看相对差会被末段小分母放大（loss≈0.003 时 0.003 的差就是 100%）。
           "numeric_first_div_abs": first_div_abs,
           "numeric_first_div_rel_pct": (round(first_div_rel, 4)
                                         if first_div_rel is not None else None),
           "numeric_worst_abs": worst_abs,
           "numeric_rel_floor": REL_FLOOR,
           "numeric_worst_rel_pct_loss_ge_floor": round(worst_rel_floored, 4),
           "numeric_control_note": ("同配置重复跑的 loss 必须逐点相同；否则逐点数值判据在此"
                                    "数据路径下不可用（差异与 A/B 差异同量级时尤其危险）"),
           "resolution_note": ("单卡噪声下限：同一配置重复 %d 次的最大相对差 = %.2f%%。"
                               "后续 A/B 的差异若小于它，则**不可判定**。" % (len(meds), spread)),
           "identity": {"boot": boot}}
    json.dump(out, open(os.path.join(args.out, "null_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("NULL_RESULT n=%d medians=%s spread=%.2f%%" % (len(meds), meds, spread), flush=True)
    print("P59NULL_OK out=%s" % args.out, flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", default="/root/qwen35-ascend-migrator")
    ap.add_argument("--config", default="/root/qwen35-ascend-migrator/out/plan/train_config.yaml")
    ap.add_argument("--key", default=None, help="点号路径，例如 training.adam_fused（null-test 时可省）")
    ap.add_argument("--a", default=None, help="A 值（默认用配置文件里的现值）")
    ap.add_argument("--b", default=None, help="B 值（null-test 时可省）")
    ap.add_argument("--numeric-tol", type=float, default=2.0, dest="numeric_tol",
                    help="逐点相对差上限（%%），超过即 REJECT_NUMERIC")
    ap.add_argument("--numeric-steps", type=int, default=2, dest="numeric_steps",
                    help=("只对前 N 步做逐点数值比较（默认 2）。理由（真机实测）：同配置两次跑在"
                          "步 1 的 grad_norm 已相对差 2.6e-4，步 3 起被混沌放大到 1%%→19%%"
                          "⇒ 步 ≥3 的逐点比较对同一配置都不成立，当门槛只会恒判 REJECT_NUMERIC。"
                          "配置若真改动了数学，第 1 步就会变，故只比前几步**不损失判别力**。"
                          "0 = 比较全部步（仅用于复现旧行为/诊断）"))
    ap.add_argument("--selftest", action="store_true", default=False,
                    help="用合成坏例自检判据（不跑训练）")
    ap.add_argument("--null-test", type=int, default=0, dest="null_test",
                    help="零假设实验：同配置重复 N 次，量单机噪声下限（与 --key/--b 互斥）")
    ap.add_argument("--order", default=ORDER,
                    help="运行顺序，支持平衡设计（如 A,B,B,A 可抵消单调漂移）")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--single-die", action="store_true", default=False,
                    help="只有 1 块 die 时使用：world1/mbs8/gas1（GBS=8 不变，但 official_comparable=False）")
    ap.add_argument("--out", default=None, help="输出目录（--selftest 时可省）")
    ap.add_argument("--mind", default="/root/MindSpeed-MM")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--selftest-orchestration", action="store_true", default=False,
                    dest="selftest_orch",
                    help="只用桩子进程跑编排路径（run_one/null_test），不碰 NPU（坑 153）")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if getattr(args, "selftest_orch", False):
        return selftest_orchestration()

    if not args.out:
        print("FATAL 非 selftest 模式必须给 --out")
        return 3
    if os.path.exists(args.out):
        print("FATAL 输出目录已存在（拒绝覆盖）：%s" % args.out)
        return 3
    import yaml
    doc0 = yaml.safe_load(open(args.config, encoding="utf-8"))
    doc0, world, _geom_official = apply_single_die(doc0, args.single_die)
    # ★ 坑 146：`official` 决定了 **step1 锚点规则**（官方几何要求 step1 == 1.924621）。
    #   旧实现只看几何 → "官方几何 + mock 数据"会被要求匹配官方锚点（mock 数据永远匹配不上）
    #   → 有效的 A/B 被误判成 `REJECT_NUMERIC`。判据改取 `_envcompat`（唯一定义处，与 P2 计划同源）。
    # ★ 本脚本自己的目录**优先**（把 59 连同 `_envcompat.py` 一起放 /root/ops 就能跑，
    #   不必改动机器上已安装的 Skill）；Skill 的 scripts/ 作为回退。
    for _p in (os.path.join(args.skill, "scripts"), os.path.dirname(os.path.abspath(__file__))):
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)
    _plan_path = os.path.join(args.skill, "out", "plan", "train_config.yaml")
    try:
        from _envcompat import official_comparable as _oc
        official, _cdetails, _onote = _oc(doc0, _plan_path, _geom_official)
    except Exception as _exc:      # ★ fail closed：判据加载不到就**不宣称可比**
        official = False
        _cdetails = {"error": type(_exc).__name__,
                     "plan": os.path.join(args.skill, "scripts")}
        _onote = ("无法加载同源可比性判据（%s）→ 保守判为不可比" % type(_exc).__name__)
    print("P59_COMPARABLE official_comparable=%s 依据=%s" % (official, _onote), flush=True)
    os.makedirs(args.out)
    boot = open("/proc/sys/kernel/random/boot_id").read().strip()

    if args.null_test:                      # ★ 零假设实验：只重复，不改任何键
        print("P59NULL_START out=%s repeat=%d steps=%d world=%d official_comparable=%s boot=%s"
              % (args.out, args.null_test, args.steps, world, official, boot), flush=True)
        return null_test(args, doc0, boot)

    if not args.key or args.b is None:
        print("FATAL 非 null-test 模式必须给 --key 与 --b")
        return 3
    cur = get_key(doc0, args.key)
    try:
        a_val = parse_value(args.a, cur) if args.a is not None else cur
        b_val = parse_value(args.b, cur)
    except ValueError as e:
        print("FATAL %s" % e)
        return 3
    if a_val == b_val:
        print("FATAL A 值与 B 值相同（%r），无从对比" % (a_val,))
        return 3
    if args.a is not None:
        set_key(doc0, args.key, a_val)          # 允许命令行显式钉住 A 值
    print("P59_START out=%s key=%s A=%r B=%r steps=%d order=%s world=%d official_comparable=%s boot=%s"
          % (args.out, args.key, a_val, b_val, args.steps, args.order, world, official, boot), flush=True)

    results = []
    order_list = [x.strip().upper() for x in args.order.split(",") if x.strip()]
    if sorted(set(order_list)) != ["A", "B"] or len(order_list) < 3:
        print("FATAL --order 必须同时含 A 与 B，且至少 3 组（示例 A,B,B,A）")
        return 3
    for i, v in enumerate(order_list):
        rec = run_one("%02d_%s" % (i, v), v, args.key, b_val, doc0, args, boot)
        results.append(rec)
        json.dump(results, open(os.path.join(args.out, "results.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        if not rec["valid"] or not rec["single_variable_ok"]:
            print("P59_FAIL 该组无退出证据或非单变量（valid=%s single_var=%s），停止"
                  % (rec["valid"], rec["single_variable_ok"]), flush=True)
            return 1

    A = [r for r in results if r["variant"] == "A"]
    B = [r for r in results if r["variant"] == "B"]
    a_mean = sum(r["window_median_ms"] for r in A) / len(A) if A else None
    drift = ((max(r["window_median_ms"] for r in A) - min(r["window_median_ms"] for r in A))
             / min(r["window_median_ms"] for r in A) * 100) if len(A) > 1 else None
    pairs = paired_stats(A, B, order_list)
    # ★ 复核意见：逐点数值检查必须覆盖**全部配对**（旧实现只看 A[0] vs B[0]）
    # ★ 2026-09-21：只比**前若干步**（默认 2）—— 步 ≥3 对同一配置的两次跑都会分叉
    #   （混沌放大），拿它当门槛等于恒判 REJECT_NUMERIC。实测见 pointwise() 的说明。
    _nsteps = int(getattr(args, "numeric_steps", 2) or 0) or None
    pw, pw_per_pair = pointwise_all(A, B, max_step=_nsteps) if (A and B) else ({}, [])
    step1_a = A[0]["loss"][0] if (A and A[0]["loss"]) else None
    step1_b = B[0]["loss"][0] if (B and B[0]["loss"]) else None
    if step1_a is None or step1_b is None:
        anchor_ok, step1_note = False, "缺少 step1 loss，无法判定"
    elif official:
        anchor_ok = abs(step1_a - STEP1_ANCHOR) < 1e-6 and abs(step1_b - STEP1_ANCHOR) < 1e-6
        step1_note = "官方几何：step1 必须等于官方锚点 %.6f" % STEP1_ANCHOR
    else:
        anchor_ok = abs(step1_a - step1_b) < 1e-6
        step1_note = "非官方几何（official_comparable=False）：官方锚点不适用，只要求 A/B 两次 step1 彼此一致"

    verdict, why = verdict_of(A, B, pairs, pw, official, args.numeric_tol,
                             drift if drift is not None else float("nan"))
    if not anchor_ok and verdict not in ("INVALID", "REJECT_NUMERIC"):
        verdict, why = "REJECT_NUMERIC", "step1 不满足锚点规则（A=%r B=%r；%s）" % (step1_a, step1_b, step1_note)

    out = {"schema": "p59ab.v2", "key": args.key, "official_comparable": official,
           "comparability": {"reason": _onote, "details": _cdetails,
                             "is_official_geometry": _geom_official,
                             "note": ("★ 三个独立结论：plan_consistent / ab_comparable / "
                                      "official_comparable（复核意见 2026-09-21）。"
                                      "official_comparable 决定 step1 锚点规则；只看几何会让 "
                                      "mock 实跑的 A/B 被误判为 REJECT_NUMERIC。"
                                      "**absence of official-comparable data ≠ absence of "
                                      "valid same-machine A/B evidence**")},
           "order_used": order_list, "steps": args.steps,
           "A_value": a_val, "B_value": b_val,
           "n_A": len(A), "n_B": len(B), "baseline_A_mean_ms": round(a_mean, 2) if a_mean else None,
           "baseline_drift_pct": round(drift, 2) if drift is not None else None,
           "paired": pairs,
           "step1_loss": {"A": step1_a, "B": step1_b, "official_anchor": STEP1_ANCHOR,
                          "anchor_ok": anchor_ok, "anchor_rule": step1_note},
           "pointwise_rel_diff_pct": pw, "numeric_tol_pct": args.numeric_tol,
           "pointwise_per_pair": pw_per_pair,
           "verdict": verdict, "verdict_reason": why,
           "runs": [{k: r[k] for k in ("tag", "variant", "valid", "single_variable_ok",
                                       "window_median_ms", "wall_seconds", "identity")}
                    for r in results],
           "note": "统计单位是**运行**；配对差按索引配对（平衡顺序可抵消线性漂移）；"
                   "性能结论不得用于宣称精度达标"}
    if official:
        # ★ 复核意见：非官方几何不得输出对标比值，避免下游误用
        out["official_median_ms"] = 431.3
        out["A_vs_official"] = round(431.3 / a_mean, 3) if a_mean else None
        out["B_vs_official"] = (round(431.3 / B[0]["window_median_ms"], 3) if B else None)
    else:
        out["official_median_ms"] = None
        out["official_ratio_suppressed"] = ("official_comparable=False：**不输出**与官方的比值"
                                           "（几何不同，比值无意义）")
    json.dump(out, open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("P59_RESULT key=%s nA=%d nB=%d pairs=%d  A=%.1f ms  B(first)=%s  paired_delta=%s%%  drift=%s"
          % (args.key, len(A), len(B), pairs["pairs"], a_mean or -1,
             ("%.1f" % B[0]["window_median_ms"]) if B else "n/a",
             pairs["mean_delta_pct"], ("%.2f%%" % drift) if drift is not None else "n/a"), flush=True)
    for k, v in pw.items():
        print("  逐点最大相对差 %-10s %s%%（可比 %d 点，零分母 %d 点）"
              % (k, v["max_pct"], v["compared"], v["undefined_zero_baseline"]), flush=True)
    print("VERDICT %s —— %s" % (verdict, why), flush=True)
    print("P59_OK out=%s" % args.out, flush=True)
    return 0


def selftest_orchestration():
    """★ 复核意见（2026-09-21）：「抽出 `compute_valid()` 解决了判据覆盖，
    但**尚未覆盖完整编排路径**。增加一次 stub 子进程驱动的 `run_one()` 集成测试。」

    真机证明了这个担忧不是理论问题 —— 坑 153：`null_test()` 里 `steps_complete`
    **从未定义**，队列第一步就 `NameError` 死掉，而 `--selftest` 全绿。

    本测试**不碰 NPU、不跑训练**：把 `subprocess.run` 换成桩子，它按 `cmd` 里的
    `--log/--steps` 伪造 train.log（格式与 `PAT` 一致）与 `train_rc=0`，
    然后让 `run_one()` 与 `null_test()` **真跑一遍**完整编排路径。
    """
    import shutil as _sh
    import tempfile as _tf
    import types as _types

    fails = []

    def ck(name, cond, detail=""):
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                               "" if cond or not detail else "  ← " + detail))
        if not cond:
            fails.append(name)

    here = os.path.dirname(os.path.abspath(__file__))
    skill = os.path.dirname(here)
    td = _tf.mkdtemp(prefix="p59orch_")
    # ★ steps 必须 ≥12：窗口是 11..steps，步数太小窗口为空（`null_test` 会如实报"算不出来"）
    steps = 12
    real_run = subprocess.run
    state = {"vary_loss": False, "n": 0}     # 数值正/反例开关（见下方两项检查）

    def fake_run(cmd, **kw):
        """桩子：伪造与 `PAT` 匹配的 train.log + driver 里的 train_rc=0。"""
        log = cmd[cmd.index("--log") + 1]
        n = int(cmd[cmd.index("--steps") + 1])
        state["n"] += 1
        # `vary_loss=True` 时让每次运行的 loss 不同 → 用来验证"判据能测出数值不同"
        loss = 1.5 + (0.1 if (state["vary_loss"] and state["n"] % 2 == 0) else 0.0)
        with open(log, "w", encoding="utf-8") as fh:
            for i in range(1, n + 1):
                fh.write("[Rank 0 | Local Rank 0] iteration %8d/%8d | consumed samples: %d | "
                         "elapsed time per iteration (ms): %.1f | learning rate: 1.0E-03 | "
                         "global batch size: 8 | loss: %.6E | grad norm: 1.000 |\n"
                         % (i, n, i * 8, 100.0 + i, loss))
        out = kw.get("stdout")
        if hasattr(out, "write"):
            out.write("train_rc=0\n")
            out.flush()
        return _types.SimpleNamespace(returncode=0, stdout="", stderr="")

    doc0 = {"parallel": {"data_parallel_size": 2},
            "training": {"micro_batch_size": 4, "gradient_accumulation_steps": 1,
                         "train_iters": 100, "save": os.path.join(td, "save"),
                         "load": "/root/Qwen3.5-0.8B-dcp",
                         "load_rank0_and_broadcast": True, "save_format": "hf",
                         "no_save_optim": True, "no_save_rng": True}}
    args = _types.SimpleNamespace(skill=skill, mind=skill, timeout=60, steps=steps,
                                  out=os.path.join(td, "ab"), numeric_tol=2.0,
                                  single_die=False, null_test=2)
    os.makedirs(args.out)

    subprocess.run = fake_run
    try:
        rec = run_one("00_A", "A", "training.micro_batch_size", 4, doc0, args, "bootX")
        ck("编排：run_one() 真跑一遍（桩子进程）", rec.get("valid") is True,
           "valid=%s steps_parsed=%s" % (rec.get("valid"), rec.get("steps_parsed")))
        ck("编排：run_one() 判出单变量", rec.get("single_variable_ok") is True,
           "details=%s" % (rec.get("single_variable_details") or {}).get("diffs", "")[:120])
        ck("编排：run_one() 步号齐全", rec.get("steps_complete") is True)
        null_rec = null_test(args, doc0, "bootX")
        ck("编排：null_test() 真跑一遍（★ 坑 153 的现场）", null_rec == 0,
           "rc=%s（旧实现在这里 NameError）" % null_rec)
        # 数值正对照：同配置重复跑必须逐点相同（否则逐点数值判据不可用）
        _ns = json.load(open(os.path.join(args.out, "null_summary.json"), encoding="utf-8"))
        ck("数值正对照：同配置重复 → identical=True",
           _ns.get("numeric_identical") is True,
           "identical=%s first_div=%s worst=%.4f%%"
           % (_ns.get("numeric_identical"), _ns.get("numeric_first_diverging_step"),
              _ns.get("numeric_worst_rel_pct") or 0.0))
        # 反例：让每次运行的 loss 不同 → 必须被如实测出（判据不是"永远说相同"）
        args.out = os.path.join(td, "numerr")
        os.makedirs(args.out)
        state["vary_loss"] = True
        state["n"] = 0
        null_rec2 = null_test(args, doc0, "bootX")
        _ns2 = json.load(open(os.path.join(args.out, "null_summary.json"), encoding="utf-8"))
        ck("数值反例：loss 每次不同 → identical=False（判据能测出差异）",
           null_rec2 == 0 and _ns2.get("numeric_identical") is False,
           "identical=%s first_div=%s worst=%.4f%%"
           % (_ns2.get("numeric_identical"), _ns2.get("numeric_first_diverging_step"),
              _ns2.get("numeric_worst_rel_pct") or 0.0))
    finally:
        subprocess.run = real_run
        _sh.rmtree(td, ignore_errors=True)
    print("P59_ORCH_SELFTEST cases=%d failed=%d" % (6, len(fails)))
    return 0 if not fails else 1


def selftest():
    """★ 合成坏例：证明上述判据真的会失败（复核意见要求"必须配的坏例"）。"""
    import copy as _c
    fails = []
    total = [0]        # ★ 原先结尾把 cases 写死成 16，历次新增用例后**数字从不变** ——
    #   "声明 ≠ 实况"（同族：`_envcompat` 曾把 fails 当 items 打印）。改为实数。

    def chk(name, cond, detail=""):
        total[0] += 1
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                               "" if cond or not detail else "  ← " + detail))
        if not cond:
            fails.append(name)

    # 坏例 0（★ 坑 147）：**有效性判据必须被自检覆盖**。
    #   原先它内联在 `run_one()` 里、`steps_complete` 用在定义之前 → 真机 UnboundLocalError
    #   而自检全绿（自检不走 run_one）。抽成 compute_valid() 后逐条喂坏样本。
    ok_ids = list(range(1, 11))
    v, sc = compute_valid(0, ["0"], ok_ids, 10)
    chk("有效判据：齐全 → valid=True", v is True and sc is True)
    v, sc = compute_valid(0, ["0"], ok_ids[:-1], 10)
    chk("有效判据：缺最后一步 → 无效", v is False and sc is False)
    v, sc = compute_valid(0, ["0"], [1, 2, 3, 4, 5, 6, 7, 7, 9, 10], 10)
    chk("有效判据：步号重复（缺 8）→ 无效", v is False and sc is False)
    v, sc = compute_valid(1, ["0"], ok_ids, 10)
    chk("有效判据：进程 rc≠0 → 无效", v is False)
    v, sc = compute_valid(0, ["1"], ok_ids, 10)
    chk("有效判据：train_rc≠0 → 无效", v is False)
    v, sc = compute_valid(0, ["0"], ok_ids + [11], 10)
    chk("有效判据：多出步号 → 无效", v is False and sc is False)

    # 坏例 1：非法布尔值不得静默取 False
    try:
        parse_value("typo", True)
        chk("非法布尔值被拒", False)
    except ValueError:
        chk("非法布尔值被拒", True)
    chk("合法布尔值仍可用", parse_value("FALSE", True) is False and parse_value("true", True) is True)

    # 坏例 2：零基线不得返回 0%
    a = {"loss": [0.0, 1.0], "grad_norm": [2.0, 2.0], "window_median_ms": 100.0}
    b = {"loss": [1.0, 1.0], "grad_norm": [2.0004, 2.0], "window_median_ms": 90.0}
    pw = pointwise(a, b)
    chk("零分母被显式计数而非记 0%", pw["loss"]["undefined_zero_baseline"] == 1
        and pw["loss"]["compared"] == 1)
    z = pointwise({"loss": [0.0], "grad_norm": [0.0]}, {"loss": [1.0], "grad_norm": [1.0]})
    chk("全零分母 → max_pct 记 None（不可判定）",
        z["loss"]["max_pct"] is None and z["grad_norm"]["max_pct"] is None)

    # 坏例 3：平衡顺序必须用上**所有** B（复核核心缺陷）
    def run(tag, var, med, loss, boot="b1", valid=True):
        return {"tag": tag, "variant": var, "window_median_ms": med,
                "loss": loss, "grad_norm": loss, "valid": valid, "single_variable_ok": True,
                "identity": {"boot": boot}, "wall_seconds": 1.0}
    A4 = [run("00_A", "A", 650.0, [1.0]), run("03_A", "A", 645.0, [1.0])]
    B4 = [run("01_B", "B", 600.0, [1.0]), run("02_B", "B", 640.0, [1.0])]
    ps = paired_stats(A4, B4, ["A", "B", "B", "A"])
    chk("平衡顺序成对统计覆盖全部 B（2 对）", ps["pairs"] == 2
        and [d["B_ms"] for d in ps["diffs"]] == [600.0, 640.0])
    chk("配对差值符合手算", [d["delta_pct"] for d in ps["diffs"]] == [round((600 - 650) / 650 * 100, 3),
                                                             round((640 - 645) / 645 * 100, 3)])
    slow_B = [run("01_B", "B", 700.0, [1.0]), run("02_B", "B", 690.0, [1.0])]
    ps2 = paired_stats(A4, slow_B, ["A", "B", "B", "A"])
    chk("第二组 B 变慢也能被计入", ps2["pairs"] == 2 and ps2["mean_delta_pct"] > 0)
    v, _ = verdict_of(A4, slow_B, ps2, pw, False, 2.0, 0.8)
    chk("B 显著更慢 → NO_GAIN", v == "NO_GAIN")

    # ★ 坏例 / 正例：**63 形状的记录**（它的 run 记录用 `window_11_end_median_ms`）
    #   现场：63 复用本函数时六组数据全部白跑（KeyError）。正例与反例都要有。
    def run63(tag, var, med):
        return {"tag": tag, "variant": var, "window_11_end_median_ms": med,
                "loss": [1.0], "grad_norm": [1.0], "valid": True, "steps_complete": True,
                "identity": {"boot": "b1"}, "wall_seconds": 1.0}
    A63 = [run63("00_A", "A", 536.6), run63("03_A", "A", 566.0), run63("04_A", "A", 536.4)]
    B63 = [run63("01_B", "B", 452.65), run63("02_B", "B", 427.3), run63("05_B", "B", 444.85)]
    ps63 = paired_stats(A63, B63, ["A", "B", "B", "A", "A", "B"])
    chk("63 形状记录（window_11_end_median_ms）能被成对统计",
        ps63["pairs"] == 3 and ps63["mean_delta_pct"] is not None
        and ps63["mean_delta_pct"] < 0, "pairs=%s mean=%s" % (ps63["pairs"], ps63["mean_delta_pct"]))
    try:
        paired_stats([{"tag": "x", "variant": "A"}], [{"tag": "y", "variant": "B"}], ["A", "B"])
        chk("两种字段名都缺 → 必须报错（不得静默当 0）", False, "没有抛错")
    except KeyError as exc:
        chk("两种字段名都缺 → 必须报错（不得静默当 0）", "window_11_end_median_ms" in str(exc))
    # ★ 这里必须用**成对差紧密**的数据：n=2 时 t(0.975,df=1)=12.706，
    #   区间会被放得很宽 —— 即"2 对很难判 WIN"。这本身就是对判据的诚实说明。
    fast_B = [run("01_B", "B", 560.0, [1.0]), run("02_B", "B", 555.0, [1.0])]
    ps_fast = paired_stats(A4, fast_B, ["A", "B", "B", "A"])
    v, _ = verdict_of(A4, fast_B, ps_fast, pw, False, 2.0, 0.8)
    chk("B 显著更快且 ≥5%（成对差紧密）→ WIN", v == "WIN")
    v, _ = verdict_of(A4, B4, ps, pw, False, 2.0, 0.8)
    chk("B 略快但区间跨 0 → UNCERTAIN（不因「看着更快」就宣称收益）", v == "UNCERTAIN")
    chk("n=2 的 t 临界值=12.706（区间宽，如实反映）", ps["t_crit"] == 12.706)

    # 坏例 4：数值不合格必须先报出，不能被漂移遮住
    badnum = pointwise({"loss": [1.0], "grad_norm": [1.0]}, {"loss": [1.5], "grad_norm": [1.0]})
    v, why = verdict_of(A4, B4, ps, badnum, False, 2.0, 9.9)
    chk("数值超限 → REJECT_NUMERIC（即使漂移很大）", v == "REJECT_NUMERIC")

    # 坏例 5：boot 不一致 / 无退出证据 / 只 1 对
    A_boot = [run("00_A", "A", 650.0, [1.0]), run("03_A", "A", 645.0, [1.0], boot="b2")]
    v, _ = verdict_of(A_boot, B4, paired_stats(A_boot, B4, []), pw, False, 2.0, 1.0)
    chk("跨 boot 混算 → INVALID", v == "INVALID")
    A_bad = [run("00_A", "A", 650.0, [1.0], valid=False), run("03_A", "A", 645.0, [1.0])]
    v, _ = verdict_of(A_bad, B4, ps, pw, False, 2.0, 1.0)
    chk("无退出证据 → INVALID", v == "INVALID")
    v, _ = verdict_of([A4[0]], [B4[0]], paired_stats([A4[0]], [B4[0]], []), pw, False, 2.0, 1.0)
    chk("仅 1 对 → INSUFFICIENT_EVIDENCE", v == "INSUFFICIENT_EVIDENCE")

    # 坏例 6：区间跨 0 → UNCERTAIN（不因"看着更快"就宣称收益）
    A_ci = [run("00_A", "A", 650.0, [1.0]), run("03_A", "A", 645.0, [1.0])]
    B_ci = [run("01_B", "B", 600.0, [1.0]), run("02_B", "B", 800.0, [1.0])]
    ps3 = paired_stats(A_ci, B_ci, ["A", "B", "B", "A"])
    v, _ = verdict_of(A_ci, B_ci, ps3, pw, False, 2.0, 1.0)
    chk("配对区间跨 0 → UNCERTAIN", v == "UNCERTAIN")

    # ★★ 复核意见（2026-09-21）合成坏例：均值 +6%、区间 [1%, 11%] —— 旧判据返回 WIN（错）。
    #   正确结论是「有正收益，但不能保证达到 5%」，因为**改善区间下界 1.03% < 5%**。
    A_six = [run("00_A", "A", 650.0, [1.0]), run("03_A", "A", 650.0, [1.0]),
             run("04_A", "A", 650.0, [1.0])]
    B_six = [run("01_B", "B", 624.0, [1.0]), run("02_B", "B", 611.0, [1.0]),
             run("05_B", "B", 598.0, [1.0])]
    ps6 = paired_stats(A_six, B_six, ["A", "B", "B", "A", "A", "B"])
    v6, why6 = verdict_of(A_six, B_six, ps6, pw, False, 2.0, 0.5)
    chk("均值+6% 但区间下界<5% → 不得判 WIN", v6 == "SMALL_GAIN",
        "verdict=%s | %s" % (v6, why6))
    # 只有**下界确实 ≥5%** 时才允许 WIN（同样的效应，把成对差做得更紧密）
    A_t = [run("00_A", "A", 650.0, [1.0]), run("03_A", "A", 650.0, [1.0]),
           run("04_A", "A", 650.0, [1.0])]
    B_t = [run("01_B", "B", 611.0, [1.0]), run("02_B", "B", 611.0, [1.0]),
           run("05_B", "B", 611.0, [1.0])]
    ps7 = paired_stats(A_t, B_t, ["A", "B", "B", "A", "A", "B"])
    v7, why7 = verdict_of(A_t, B_t, ps7, pw, False, 2.0, 0.5)
    chk("改善区间下界 ≥5% → WIN", v7 == "WIN", "verdict=%s | %s" % (v7, why7))

    # ★ 复核意见：逐点数值检查必须覆盖**全部配对**（旧实现只看 A[0] vs B[0]）
    A_np = [run("00_A", "A", 650.0, [1.0, 1.0]), run("03_A", "A", 650.0, [1.0, 3.0])]
    B_np = [run("01_B", "B", 600.0, [1.0, 1.0]), run("02_B", "B", 600.0, [1.0, 1.0])]
    ps8 = paired_stats(A_np, B_np, ["A", "B", "B", "A"])
    pw_first = pointwise(A_np[0], B_np[0])
    pw_multi, pw_per = pointwise_all(A_np, B_np)
    v8, _ = verdict_of(A_np, B_np, ps8, pw_first, False, 2.0, 1.0)
    v8b, why8b = verdict_of(A_np, B_np, ps8, pw_multi, False, 2.0, 1.0)
    chk("只看第一对 → 漏掉第二对数值不合格（旧实现的实际行为）",
        v8 == "WIN" and len(pw_per) == 2, "verdict=%s pairs=%d" % (v8, len(pw_per)))
    chk("覆盖全部配对 → REJECT_NUMERIC", v8b == "REJECT_NUMERIC",
        "verdict=%s | %s" % (v8b, why8b))

    # 坏例 8（复核要求）：序列长度不一致必须**显式拒绝**，不得静默按前缀比较
    short_b = dict(run("01_B", "B", 600.0, [1.0]), loss=[1.0, 1.0], grad_norm=[1.0, 1.0])
    pw_len = pointwise(A4[0], short_b)
    chk("长度不一致被标出（length_ok=False）", pw_len["loss"]["length_ok"] is False
        and pw_len["loss"]["len_a"] == 1 and pw_len["loss"]["len_b"] == 2)
    v, why = verdict_of(A4, [short_b, B4[1]], ps, pw_len, False, 2.0, 1.0)
    chk("长度不一致 → REJECT_NUMERIC", v == "REJECT_NUMERIC" and "长度不一致" in why)

    # 坏例 7（复核要求）：缺一步 / 重复步号必须判 INVALID
    A_step = [dict(A4[0], steps_complete=False), A4[1]]
    v, _ = verdict_of(A_step, B4, ps, pw, False, 2.0, 1.0)
    chk("缺一步/重复步号 → INVALID", v == "INVALID")

    print("\n注：n=2 时 t(0.975,df=1)=12.706 —— 配对区间很宽，2 对实验很难判 WIN；\n    要稳地判 ≥5% 收益，至少要 3 对（t=4.303），最好 4 对以上。")
    print("P59_SELFTEST_%s cases=%d failed=%d" % ("OK" if not fails else "FAIL", total[0], len(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())

