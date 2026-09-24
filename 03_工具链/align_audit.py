# -*- coding: utf-8 -*-
r"""align_audit.py —— 「尽量与官方一致」的**可审计形式**（重建版）

★ 重建说明（2026-09-22）：原文件在 `04_核心资产\交付物生成脚本\`，已在坑 180 中被删除。
  本版按事故前会话里读到的全文 + 本轮对它做过的全部改动**合并还原**，并保留两条核心设计：
    · 锚点**推导**（从官方日志的配置 dump 里自动取，而不是我手抄一张表）；
    · 推导失败/覆盖面不可知时**拒绝给绿灯**（fail-closed），并**必须同时报覆盖面**。

为什么需要它
-----------
用户指令：「使用 B 机器，然后尽量和官方的一致」。口头约定无法被复核，所以把规则做成一个
**会失败的检查**：

    官方值优先；凡与官方锚点不同者，**必须在 versions.lock 的偏离清单里登记**；
    未登记的差异 = `UNDECLARED` ⇒ 退出码 1。

两段检查（缺一不可）
------------------
  第一段：**手写锚点表** OFFICIAL（13 项，逐条注明出处）—— 覆盖"我明确知道的官方值"。
  第二段：**官方日志自己的配置 dump 逐键比对** —— 覆盖面由官方日志决定，**不由我的记性决定**。
    这一段是坑 170 的产物：只靠第一段时，`parallel.fsdp_plan.reshard_after_forward`
    （官方 True / 我方 False，且是显存与性能相关的键）**从来没被比过**，而审计照样全绿。

用法
----
    python align_audit.py --run <配置 yaml> [--lock <versions.lock>] [--official-log <官方日志>]
退出码：0 = 全部 ALIGNED 或已登记；1 = 存在 UNDECLARED 差异 / 锚点不可得；2 = 用法/IO 错误
"""
import argparse
import os
import re
import sys

from _project_paths import bootstrap_paths
PATHS = bootstrap_paths(parse_cli=__name__ == "__main__")

try:
    import yaml
except ImportError:
    print("ALIGN_AUDIT_FAIL no_pyyaml")
    sys.exit(2)

_SKILL = PATHS.skill
DEFAULT_OFFICIAL_LOG = os.path.join(_SKILL, "examples", "train", "official_baseline.log")
DEFAULT_LOCK = os.path.join(_SKILL, "config", "versions.lock")

# 官方锚点（出处见模块 docstring）：
#   官方日志 official_baseline.log:235 区域 → load_rank0_and_broadcast: False
#   官方日志 official_baseline.log:94 / L27 → skip_gdn_recompute: True
#   versions.lock → geometry(world2/mbs4/gas1) / dataloader(8,16) / gdn(triton,triton)
OFFICIAL = {
    "parallel.data_parallel_size": (2, "versions.lock config.geometry = world2/mbs4/gas1"),
    "training.micro_batch_size": (4, "versions.lock config.geometry"),
    "training.gradient_accumulation_steps": (1, "versions.lock config.geometry"),
    "data.dataloader_param.num_workers": (8, "versions.lock config.dataloader = num_workers=8"),
    "data.dataset_param.basic_parameters.preprocessing_num_workers": (
        16, "versions.lock config.dataloader = preprocessing_num_workers=16"),
    "data.dataset_param.basic_parameters.cutoff_len": (1024, "官方日志配置 dump（COCO 单图口径）"),
    # ★ 2026-09-21 修正：该键在 **preprocess_parameters** 下（我原先写 basic_parameters，
    #   于是真实配置里永远读不到它 ⇒ 误报成 UNDECLARED，并掩盖了真正的偏离清单）
    "data.dataset_param.preprocess_parameters.image_max_pixels": (262144, "官方/几何模板同值"),
    "model.gdn_implementation": ("triton", "versions.lock config.gdn（官方认可路径）"),
    "model.causal_conv1d_implementation": ("triton", "versions.lock config.gdn"),
    "model.skip_gdn_recompute": (True, "official_baseline.log:94 与 L27 注册行"),
    "training.load_rank0_and_broadcast": (False, "official_baseline.log:235 区域"),
    "training.save_format": ("auto", "官方未设该键 ⇒ 上游默认 auto"),
    "data.dataset_param.basic_parameters.dataset": (
        "output_llava_coco_data.json", "versions.lock data.full_json（157,712 样本 / 205MB）"),
}

# 键 → 偏离清单条目名（清单里登记过就允许不同）
LEDGER_MAP = {
    "data.dataloader_param.num_workers": ("deviation.config.dataloader",),
    "data.dataset_param.basic_parameters.preprocessing_num_workers": ("deviation.config.dataloader",),
    "data.dataset_param.basic_parameters.cutoff_len": ("deviation.data.cutoff_len",),
    "data.dataset_param.basic_parameters.dataset": ("deviation.data.dataset",),
    "training.load_rank0_and_broadcast": ("deviation.config.load_rank0_and_broadcast",),
    "training.save_format": ("deviation.config.save_format",),
    # 2026-09-21：官方 `parallel.fsdp_plan` 与我们的两个差异（真实配置差异，登记后才允许不同）
    "parallel.fsdp_plan.pregather": ("deviation.config.fsdp_plan",),
    "parallel.fsdp_plan.reshard_after_forward": ("deviation.config.fsdp_plan",),
    # 2026-09-21：下面这一组是"审计锚点改成官方 dump"之后**才暴露**的 13 处差异里的主体。
    #   其中 `features.*` 三条 + `skip_flash_attn_recompute` 全是**官方的省显存开关**，
    #   我方全关 ⇒ 每步激活全留 ⇒ 这比 `skip_gdn_recompute` 更直接地解释第 96 步 OOM。
    "features.recompute": ("deviation.config.features",),
    "features.enable_chunk_loss": ("deviation.config.features",),
    "features.enable_activation_offload": ("deviation.config.features",),
    "model.skip_flash_attn_recompute": ("deviation.config.skip_flash_attn_recompute",),
    # 推荐配置 A 把官方 true 改成 false ⇒ 这个键真的偏离了；条目在 versions.lock 里，
    #   但映射此前没接上 ⇒ 等于"登记了、审计里却看不见"（审计拦得对）
    "model.skip_gdn_recompute": ("deviation.config.skip_gdn_recompute",
                                 "deviation.profile.recommended_A"),
    "parallel.fully_shard_parallel_size": ("deviation.config.fully_shard_parallel_size",),
    "training.save_interval": ("deviation.config.save_interval",),
    # 仅路径不同（同一份权重/各自实验目录），语义相同：
    "model.model_name_or_path": ("deviation.paths.model",),
    "data.dataset_param.preprocess_parameters.model_name_or_path": ("deviation.paths.model",),
    "training.load": ("deviation.paths.model",),
    "training.save": ("deviation.paths.save",),
    # 数据目录与 `deviation.data.dataset` 是同一条偏离（mock 数据）：
    "data.dataset_param.basic_parameters.dataset_dir": ("deviation.data.dataset",),
}

# 刻意不参与"官方 dump × 本次运行"逐键比对的路径前缀（只影响**怎么测**，不改变被测对象）
IGNORE_PREFIXES = ("tools.",)


def dig(doc, dotted, default=None):
    cur = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def parse_ledger(lock_path):
    """从 versions.lock 读出已声明的偏离条目名集合（只看 `deviation.` 开头的行）。"""
    declared = set()
    if not lock_path or not os.path.isfile(lock_path):
        return declared
    for line in open(lock_path, encoding="utf-8").read().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue
        name = line.split("|")[0].strip()
        if name.startswith("deviation."):
            declared.add(name)
    return declared


def load_official_dump(log_path):
    """★ 从官方日志里**推导**官方配置（而不是由我手抄一份锚点表）。

    为什么必须这样：本脚本原先只有一张手写的 13 键 `OFFICIAL` 表，于是
    `official_baseline.log:57 reshard_after_forward: True` 与我方模板的 `False` **根本没被比过** ——
    审计照样打印 `ALIGNED=9 / DECLARED=4 / UNDECLARED=0`（全绿）。**绿灯是"没看"出来的**。
    官方日志里本来就有一整段 `Configuration Details`（`parallel:`/`model:`/`data:`/… 的 YAML dump），
    直接把它当锚点集：**官方写了什么键，就比什么键**。

    返回 (doc, note)；doc 为 None 表示推导失败（此时**必须报错**，不能退回"只看手抄表"）。
    """
    if not log_path or not os.path.isfile(log_path):
        return None, "找不到官方日志 %s" % log_path
    lines = open(log_path, encoding="utf-8", errors="replace").read().splitlines()
    top = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*:$")
    start = None
    for i, ln in enumerate(lines):
        if top.match(ln) and i + 1 < len(lines) and lines[i + 1].startswith(" "):
            start = i
            break
    if start is None:
        return None, "官方日志里找不到配置 dump 的起点（顶层键 + 缩进）"
    end = start
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        if ln.strip() == "" or ln.startswith(" ") or top.match(ln):
            end = j
        else:
            break
    text = "\n".join(lines[start:end + 1])
    try:
        doc = yaml.safe_load(text)
    except Exception as exc:
        return None, "官方 dump（第 %d..%d 行）YAML 解析失败：%s" % (start + 1, end + 1, exc)
    if not isinstance(doc, dict) or "parallel" not in doc:
        return None, "官方 dump（第 %d..%d 行）解析结果不像配置（顶层键=%s）" % (
            start + 1, end + 1, sorted(doc.keys()) if isinstance(doc, dict) else type(doc).__name__)
    return doc, "官方日志第 %d..%d 行（Configuration Details）" % (start + 1, end + 1)


def flatten(doc, prefix=""):
    """把嵌套配置摊平成 `点号路径 -> 值`（列表整体作为一个叶子，按值比较）。"""
    out = {}
    if isinstance(doc, dict):
        for k, v in doc.items():
            p = "%s.%s" % (prefix, k) if prefix else str(k)
            if isinstance(v, dict):
                out.update(flatten(v, p))
            else:
                out[p] = v
    elif isinstance(doc, list):
        out[prefix] = doc
    return out


def same_value(a, b):
    return a == b or str(a) == str(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="本次运行配置 yaml")
    ap.add_argument("--lock", default=None, help="versions.lock（读偏离清单）")
    ap.add_argument("--official-log", default=DEFAULT_OFFICIAL_LOG,
                    help="官方日志（内含 Configuration Details 的 YAML dump）")
    a = ap.parse_args()
    if a.lock is None:
        a.lock = DEFAULT_LOCK

    if not os.path.isfile(a.run):
        print("ALIGN_AUDIT_FAIL 缺 --run 文件: %s" % a.run)
        return 2
    run = yaml.safe_load(open(a.run, encoding="utf-8"))
    declared = parse_ledger(a.lock)
    print("== 官方对齐审计 ==")
    print("  run  = %s" % a.run)
    print("  lock = %s（已声明偏离 %d 条）" % (a.lock, len(declared)))

    aligned, declared_rows, undeclared = [], [], []
    for key, (want, prov) in sorted(OFFICIAL.items()):
        got = dig(run, key, "<absent>")
        if key.endswith("dataset") and isinstance(got, str):
            got_cmp = os.path.basename(got)
        else:
            got_cmp = got
        same = (got_cmp == want) or (str(got_cmp) == str(want))
        if same:
            aligned.append(key)
            status = "ALIGNED"
        else:
            names = LEDGER_MAP.get(key, ())
            if names and any(n in declared for n in names):
                declared_rows.append((key, want, got_cmp, names[0]))
                status = "DECLARED"
            else:
                undeclared.append((key, want, got_cmp))
                status = "**UNDECLARED**"
        print("  %-12s %-62s official=%-28s run=%s" % (status, key, want, got_cmp))
    print("\n  出处提示：official 值来自 -> %s" % OFFICIAL["model.skip_gdn_recompute"][1])

    # ---- 第二段：以**官方日志自己的配置 dump** 为锚点逐键比对 ---------------------------
    off_doc, note = load_official_dump(a.official_log)
    if off_doc is None:
        print("\n== 官方 dump 逐键比对 ==")
        print("  **FAIL** %s" % note)
        print("ALIGN_AUDIT_FAIL 无法从官方日志推导锚点集合 ⇒ **覆盖面不可知**，拒绝给绿灯")
        return 1
    print("\n== 官方 dump 逐键比对（锚点来源：%s）==" % note)
    off = flatten(off_doc)
    runflat = flatten(run)
    common = sorted(k for k in (set(off) & set(runflat)) if not k.startswith(IGNORE_PREFIXES))
    diff_declared, diff_undeclared = [], []
    for k in common:
        if same_value(off[k], runflat[k]):
            continue
        names = LEDGER_MAP.get(k, ())
        if names and any(n in declared for n in names):
            diff_declared.append((k, off[k], runflat[k], names[0]))
        else:
            diff_undeclared.append((k, off[k], runflat[k]))
    only_off = sorted(k for k in (set(off) - set(runflat)) if not k.startswith(IGNORE_PREFIXES))
    only_run = sorted(k for k in (set(runflat) - set(off)) if not k.startswith(IGNORE_PREFIXES))
    print("  两侧都有（**真的比过**）%d 键；只有官方有 %d 键；只有我方有 %d 键"
          % (len(common), len(only_off), len(only_run)))
    for k, w, g in diff_undeclared:
        print("      ! **UNDECLARED** %s: official=%r run=%r" % (k, w, g))
    for k, w, g, n in diff_declared:
        print("      · DECLARED     %s: official=%r run=%r  ← 登记于 %s" % (k, w, g, n))
    if only_off:
        print("  只有官方有的键（我方配置**没写**，跑的是上游默认值 ⇒ 本审计**无法判定**是否一致，"
              "因此**不声称一致**、也不计入 ALIGNED）前 20：")
        for k in only_off[:20]:
            print("      ? %s: official=%r" % (k, off[k]))
        if len(only_off) > 20:
            print("      ...（共 %d 键，其余省略）" % len(only_off))
    if only_run:
        print("  只有我方有的键（官方 dump 里没有 ⇒ 官方走默认值）前 10：%s"
              % ", ".join(only_run[:10]))

    print("\n== 汇总 ==")
    print("  ALIGNED  = %d 项（手写锚点）" % len(aligned))
    print("  DECLARED = %d 项（手写锚点，已在偏离清单登记）" % len(declared_rows))
    for k, w, g, n in declared_rows:
        print("      · %s: official=%s run=%s  ← 登记于 %s" % (k, w, g, n))
    print("  UNDECLARED = %d 项（手写锚点）" % len(undeclared))
    for k, w, g in undeclared:
        print("      ! %s: official=%s run=%s" % (k, w, g))
    print("  dump 比对：真比过 %d 键 / 值不同且已登记 %d 键 / **值不同且未登记 %d 键** / 未比对 %d 键"
          % (len(common), len(diff_declared), len(diff_undeclared), len(only_off)))

    if undeclared or diff_undeclared:
        print("ALIGN_AUDIT_FAIL 存在未登记的偏离 → 要么改回官方值，要么在 "
              "versions.lock 的偏离清单里登记并写明原因/影响范围")
        return 1
    print("ALIGN_AUDIT_OK 手写锚点全部一致或已登记；dump 逐键比对无未登记差异")
    return 0


if __name__ == "__main__":
    sys.exit(main())
