#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20_plan_migration.py — P2 迁移方案与配置生成

输入：out/probe/env.json（P0 产物）、out/analyze/migrate_points.json（P1 产物，可选）
输出：
  out/plan/train_config.yaml   — 按探测到的环境自动生成的训练配置（可直接用于 P5）
  out/plan/migrate_plan.md     — 迁移点 → 落点映射方案（人读）

设计要点（冗余/健壮）：
  * 不走 sed：用"逐行状态机"精确改 yaml，**每处改动都回读校验**，失败即报错退出（避免静默失败）
  * 配置档来自 config/env_matrix.yaml 的 chip_profiles；字段缺失时按段插入而非静默跳过
  * 生成的配置必须与探测环境自洽（例如无 triton 时后端降级为 eager 并标注）

退出码：0 成功；2 输入缺失/IO 错误；3 配置生成校验失败
"""

import argparse
import json
import os
import re
import sys

try:
    import yaml
except ImportError:
    yaml = None

SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 待生成/覆盖的配置字段：(段路径, 键名, 期望缩进空格数)
FIELD_SPECS = [
    (("parallel",), "data_parallel_size", 2),
    (("parallel",), "fully_shard_parallel_size", 2),
    (("parallel", "fsdp_plan"), "pregather", 4),
    (("parallel", "fsdp_plan"), "num_to_forward_prefetch", 4),
    (("parallel", "fsdp_plan"), "num_to_backward_prefetch", 4),
    (("training",), "micro_batch_size", 2),
    (("training",), "gradient_accumulation_steps", 2),
    (("training",), "load_rank0_and_broadcast", 2),
    (("training",), "save_format", 2),
    (("features",), "recompute", 2),
    (("features",), "enable_chunk_loss", 2),
    (("features",), "enable_activation_offload", 2),
    (("model",), "gdn_implementation", 2),
    (("model",), "causal_conv1d_implementation", 2),
    # ★ 2026-09-21：官方日志里该键是 True，我方 A3 曾是 false 且**判据链完全不比对它**
    #   （复核意见：新比较必须覆盖它）。现由 P0 档位写入并在下方回读校验。
    (("model",), "skip_gdn_recompute", 2),
]


def yaml_val(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    return str(v)


def _section_ranges(lines):
    """返回 {顶层键: (start_idx, end_idx)}，end 为下一个顶层键或文件尾。"""
    tops = [(i, m.group(1)) for i, l in enumerate(lines)
            for m in [re.match(r"^([A-Za-z_][\w\.\-]*)\s*:", l)] if m]
    ranges = {}
    for k, (idx, name) in enumerate(tops):
        end = tops[k + 1][0] if k + 1 < len(tops) else len(lines)
        ranges[name] = (idx, end)
    return ranges


def _sub_range(lines, start, end, subkey):
    """在 [start,end) 内找子段 'subkey:'（缩进 >0），返回其行区间。"""
    sub_start = None
    base_indent = None
    for i in range(start, end):
        m = re.match(r"^(\s+)" + re.escape(subkey) + r"\s*:", lines[i])
        if m:
            sub_start = i
            base_indent = len(m.group(1))
            break
    if sub_start is None:
        return None
    sub_end = end
    for j in range(sub_start + 1, end):
        if lines[j].strip() and not lines[j].startswith(" " * (base_indent + 1)):
            sub_end = j
            break
    return (sub_start, sub_end)


def locate_field(lines, section_path, key, indent):
    """定位字段行号。section_path 如 ('parallel','fsdp_plan')。返回 (lineno, cur_value) 或 (None, None)。"""
    ranges = _section_ranges(lines)
    if section_path[0] not in ranges:
        return None, None
    s, e = ranges[section_path[0]]
    for sub in section_path[1:]:
        r = _sub_range(lines, s, e, sub)
        if r is None:
            return None, None
        s, e = r
    # 在 (s,e] 内找精确缩进的 key
    for i in range(s + 1, e):
        m = re.match(r"^(\s*)" + re.escape(key) + r"\s*:\s*(.*?)(\s*#.*)?$", lines[i])
        if m and len(m.group(1)) == indent:
            return i, m.group(2).strip()
    return None, None


def apply_field(lines, section_path, key, indent, value):
    """设置字段：存在则替换，不存在则插入该段（子段）末尾。返回 (ok, action)。"""
    lineno, cur = locate_field(lines, section_path, key, indent)
    new_line = " " * indent + "%s: %s" % (key, yaml_val(value))
    if lineno is not None:
        if cur == yaml_val(value):
            return True, "unchanged"
        lines[lineno] = new_line
        return True, "replaced"
    # 插入：定位父段（最后一个存在的段路径）
    ranges = _section_ranges(lines)
    if section_path[0] not in ranges:
        return False, "section_missing:%s" % section_path[0]
    s, e = ranges[section_path[0]]
    for sub in section_path[1:]:
        r = _sub_range(lines, s, e, sub)
        if r is None:
            return False, "sub_section_missing:%s" % sub
        s, e = r
    # 插到该段最后一行有效内容之后
    ins = s + 1
    for i in range(s + 1, e):
        if lines[i].strip():
            ins = i + 1
    lines.insert(ins, new_line)
    return True, "inserted"


def main():
    ap = argparse.ArgumentParser(description="P2 迁移方案与配置生成（按环境自适应）")
    ap.add_argument("--env", default="out/probe/env.json", help="P0 产物 env.json")
    ap.add_argument("--points", default="out/analyze/migrate_points.json", help="P1 产物（可选）")
    ap.add_argument("--template", default=os.path.join(SKILL_ROOT, "config", "templates", "qwen3_5_0_8B_base.yaml"),
                    help="训练配置模板 yaml")
    ap.add_argument("--out", default="out/plan", help="输出目录")
    ap.add_argument("--data-json", default=None, help="数据集 json 路径（覆盖模板默认值）")
    ap.add_argument("--data-dir", default=None, help="数据集目录（覆盖模板默认值）")
    ap.add_argument("--weight-hf", default=None, help="hf 权重路径（覆盖模板默认值）")
    ap.add_argument("--weight-dcp", default=None, help="dcp 权重路径（覆盖模板默认值）")
    ap.add_argument("--steps", type=int, default=100, help="训练步数")
    args = ap.parse_args()

    # ---- 读环境档案
    if not os.path.isfile(args.env):
        print("FATAL 缺少 P0 产物：%s（先跑 scripts/00_probe_env.py）" % args.env, file=sys.stderr)
        return 2
    env = json.loads(open(args.env, encoding="utf-8").read())
    profile = env.get("recommended_profile", {})
    path = env.get("path", "unknown")
    print("== P2 迁移方案与配置生成 ==")
    print("环境路径 : %s" % path)
    print("配置档   : %s（%s）" % (profile.get("profile"), profile.get("note", "")))

    # ---- 读模板
    if not os.path.isfile(args.template):
        print("FATAL 模板不存在：%s" % args.template, file=sys.stderr)
        return 2
    text = open(args.template, encoding="utf-8").read()
    lines = text.splitlines()

    # ---- 环境 → 目标字段值
    targets = {
        ("parallel", "data_parallel_size"): profile.get("world_size", 1),
        ("parallel", "fully_shard_parallel_size"): "auto",
        ("parallel", "fsdp_plan", "pregather"): bool(profile.get("pregather", False)),
        ("parallel", "fsdp_plan", "num_to_forward_prefetch"): profile.get("prefetch", 1),
        ("parallel", "fsdp_plan", "num_to_backward_prefetch"): profile.get("prefetch", 1),
        ("training", "micro_batch_size"): profile.get("mbs", 4),
        ("training", "gradient_accumulation_steps"): profile.get("gas", 1),
        # ★ 坑 108：CANN beta/RC 上 dcp.load() 的 HCCL gather 路径在 aicpu 不可用；
        #   由 P0 探测（CANN 版本标识）决定，而不是靠人记得改配置。
        ("training", "load_rank0_and_broadcast"): bool(profile.get("load_rank0_and_broadcast", False)),
        # ★ 坑 113：save 路径的对称工作区（坑 108 只管 load）。
        #   CANN beta/RC 上收尾保存会撞 dcp.save 的 SavePlan 计划广播 → ACL 507018 → rc≠0；
        #   改走 HF safetensors 保存可绕开（需 no_save_optim/no_save_rng 同为真）。
        ("training", "save_format"): profile.get("save_format", "auto"),
        ("features", "recompute"): bool(profile.get("recompute", False)),
        ("features", "enable_chunk_loss"): bool(profile.get("enable_chunk_loss", False)),
        ("features", "enable_activation_offload"): bool(profile.get("enable_activation_offload", False)),
        ("model", "gdn_implementation"): profile.get("operator_backend", "triton"),
        ("model", "causal_conv1d_implementation"): profile.get("operator_backend", "triton"),
        # ★ 默认 True = **官方值**；P0 在 backend 退到 eager 时会把它置 False（源码耦合）
        ("model", "skip_gdn_recompute"): bool(profile.get("skip_gdn_recompute", True)),
    }

    # ---- 应用并校验
    spec_map = {tuple(sp): (key, indent) for sp, key, indent in
                [((*s, k) if len(s) > 0 else (k,), k, ind) for s, k, ind in
                 [((*sp,), key, ind) for sp, key, ind in FIELD_SPECS]]}
    report = []
    failed = []
    for sp, key, indent in FIELD_SPECS:
        target_key = (*sp, key)
        if target_key not in targets:
            continue
        val = targets[target_key]
        ok, action = apply_field(lines, sp, key, indent, val)
        report.append(((".".join(sp) + "." + key), yaml_val(val), action, ok))
        if not ok:
            failed.append((".".join(sp) + "." + key, action))

    # ---- 数据/权重路径（可选覆盖）
    def set_path(section, key, value, indent, note):
        nonlocal lines
        ok, action = apply_field(lines, section, key, indent, value)
        report.append((".".join(section) + "." + key, str(value), action, ok))
        if not ok:
            failed.append((".".join(section) + "." + key, action))

    if args.data_json:
        set_path(("data", "dataset_param", "basic_parameters"), "dataset",
                 "['%s']" % args.data_json, 6, "数据集 json")
        # dataset 是列表：模板里可能是内联列表，此处直接按字符串写入（保持引号形式）
    if args.data_dir:
        set_path(("data", "dataset_param", "basic_parameters"), "dataset_dir", args.data_dir, 6, "数据集目录")
    if args.weight_hf:
        set_path(("model",), "model_name_or_path", args.weight_hf, 2, "hf 权重")
        set_path(("data", "dataset_param", "preprocess_parameters"), "model_name_or_path", args.weight_hf, 6, "preprocess 权重")
    if args.weight_dcp:
        set_path(("training",), "load", args.weight_dcp, 2, "dcp 权重")
    set_path(("training",), "train_iters", args.steps, 2, "训练步数")

    if failed:
        print("\n配置生成校验失败（可能存在段结构变化，请人工核对模板）：", file=sys.stderr)
        for k, why in failed:
            print("  - %s (%s)" % (k, why), file=sys.stderr)
        return 3

    # ---- ★ 回读校验（**先校验、后落盘**；不一致=致命，不再是 warning）
    # 旧实现的两个毛病（坑 111）：
    #   ① 先 `open(cfg_path,"w")` 落盘、再校验 → 校验失败时**坏配置已经躺在磁盘上**，
    #      后续 P5 会直接消费它（坑 109 的 `ConfigValidationError` 就是这么来的）；
    #   ② 回读不一致只记进 `warnings`，脚本仍返回 0 并打印 `PLAN_OK` → **判据弱于其问题**。
    # 现在：YAML 解析失败或任一字段回读不一致 → **拒绝产出**（rc=3），磁盘上不留坏配置。
    text_out = "\n".join(lines) + "\n"
    outdir = os.path.abspath(args.out)
    warnings = []
    if yaml is None:
        print("FATAL 需要 pyyaml 才能回读校验生成的配置（pip install pyyaml）；"
              "拒绝产出未经验证的配置", file=sys.stderr)
        return 3
    try:
        doc = yaml.safe_load(text_out)
    except Exception as e:
        print("FATAL 生成的 YAML 无法解析（**未落盘**，避免坑 109 类故障被下游消费）：%s" % e,
              file=sys.stderr)
        return 3
    if not isinstance(doc, dict):
        print("FATAL 生成的 YAML 顶层不是映射（got %s）；拒绝产出" % type(doc).__name__, file=sys.stderr)
        return 3
    gp = (doc.get("parallel") or {}).get("fsdp_plan") or {}
    checks = [
        ("parallel.data_parallel_size", (doc.get("parallel") or {}).get("data_parallel_size"),
         profile.get("world_size", 1)),
        ("training.micro_batch_size", (doc.get("training") or {}).get("micro_batch_size"),
         profile.get("mbs", 4)),
        ("training.gradient_accumulation_steps",
         (doc.get("training") or {}).get("gradient_accumulation_steps"), profile.get("gas", 1)),
        ("training.load_rank0_and_broadcast",
         (doc.get("training") or {}).get("load_rank0_and_broadcast"),
         bool(profile.get("load_rank0_and_broadcast", False))),
        ("training.save_format", (doc.get("training") or {}).get("save_format"),
         profile.get("save_format", "auto")),
        ("training.train_iters", (doc.get("training") or {}).get("train_iters"), args.steps),
        ("features.recompute", (doc.get("features") or {}).get("recompute"),
         bool(profile.get("recompute", False))),
        ("features.enable_chunk_loss", (doc.get("features") or {}).get("enable_chunk_loss"),
         bool(profile.get("enable_chunk_loss", False))),
        ("features.enable_activation_offload",
         (doc.get("features") or {}).get("enable_activation_offload"),
         bool(profile.get("enable_activation_offload", False))),
        ("parallel.fsdp_plan.pregather", gp.get("pregather"), bool(profile.get("pregather", False))),
        ("model.gdn_implementation", (doc.get("model") or {}).get("gdn_implementation"),
         profile.get("operator_backend", "triton")),
        ("model.causal_conv1d_implementation",
         (doc.get("model") or {}).get("causal_conv1d_implementation"),
         profile.get("operator_backend", "triton")),
        ("model.skip_gdn_recompute", (doc.get("model") or {}).get("skip_gdn_recompute"),
         bool(profile.get("skip_gdn_recompute", True))),
    ]
    mism = [(n, a, e) for n, a, e in checks if a != e]
    if mism:
        print("FATAL 回读不一致 —— 生成的配置与预期不符，**拒绝产出**（磁盘上不留坏配置）：",
              file=sys.stderr)
        for name, actual, expect in mism:
            print("  - %s: 实际=%s 期望=%s" % (name, actual, expect), file=sys.stderr)
        return 3

    # ★ 坑 113 的前提守卫：`trainer.py:439-448` 规定 `save_format != dcp` 时
    #   必须 `no_save_optim` 与 `no_save_rng` **同为真**，否则会被**静默强制回退 dcp** ——
    #   于是收尾保存的 ACL 507018 又回来了，而且没有任何报错提示。
    #   这类"静默回退"正是本项目反复踩的形态，必须在生成阶段就拦住。
    _trdoc = doc.get("training") or {}
    if str(_trdoc.get("save_format", "auto")).lower() != "dcp":
        _miss_pre = [k for k in ("no_save_optim", "no_save_rng") if _trdoc.get(k) is not True]
        if _miss_pre:
            print("FATAL save_format=%s 要求 %s 同时为 true，否则 trainer 会**静默回退 dcp**，"
                  "坑 113 的收尾保存崩溃（ACL 507018 / rc≠0）会复发。当前不满足: %s"
                  % (_trdoc.get("save_format"), " 与 ".join(_miss_pre), "、".join(_miss_pre)),
                  file=sys.stderr)
            return 3

    # ---- 落盘（此时已证明是合法 YAML 且字段正确）
    os.makedirs(outdir, exist_ok=True)
    cfg_path = os.path.join(outdir, "train_config.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(text_out)

    # ---- 几何速览（红线仍由 50_train.py 在启动前硬断言；此处只为尽早暴露）
    _mbs = (doc.get("training") or {}).get("micro_batch_size")
    _gas = (doc.get("training") or {}).get("gradient_accumulation_steps")
    _dp = (doc.get("parallel") or {}).get("data_parallel_size")
    try:
        _gbs = int(_dp) * int(_mbs) * int(_gas)
        print("PLAN_GEOMETRY world=%s mbs=%s gas=%s → GBS=%s（官方红线 = 8）" % (_dp, _mbs, _gas, _gbs))
        if _gbs != 8:
            warnings.append("GBS=%s ≠ 8：违反官方几何红线（50_train.py 将拒绝执行）" % _gbs)
    except Exception:
        print("PLAN_GEOMETRY world=%s mbs=%s gas=%s → GBS=不可计算" % (_dp, _mbs, _gas))
        warnings.append("GBS 不可计算（world/mbs/gas 非整数）")

    # ---- 迁移方案文档
    points = {}
    if os.path.isfile(args.points):
        try:
            points = json.loads(open(args.points, encoding="utf-8").read())
        except Exception:
            points = {}
    plan_md = build_plan_md(env, profile, points, cfg_path, report, warnings)
    plan_path = os.path.join(outdir, "migrate_plan.md")
    open(plan_path, "w", encoding="utf-8").write(plan_md)

    print("\n字段改动明细：")
    for name, val, action, ok in report:
        print("  %-46s %-10s %s" % (name, val, action))
    if warnings:
        print("\n⚠ 警告：")
        for w in warnings:
            print("  - %s" % w)
    print("\n产出：")
    print("  配置: %s" % cfg_path)
    print("  方案: %s" % plan_path)
    print("PLAN_OK profile=%s path=%s" % (profile.get("profile"), path))
    print("下一步: P3 算子验证 → python3 scripts/30_verify_ops.py；或直接 P5 训练 → scripts/50_train.py")
    return 0


def build_plan_md(env, profile, points, cfg_path, report, warnings):
    sw = env.get("software", {})
    hw = env.get("hardware", {})
    lines = []
    lines.append("# 迁移方案（P2 自动生成）\n")
    lines.append("## 1. 环境画像\n")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append("| 执行路径 | `%s` |" % env.get("path"))
    lines.append("| 芯片 | %s（die=%s，HBM/chip=%sGB） |" % (
        hw.get("chip_model"), hw.get("chip_count"), profile.get("hbm_gb_per_chip")))
    lines.append("| CANN / torch / torch_npu | %s / %s / %s |" % (
        sw.get("cann_version"), sw.get("torch"), sw.get("torch_npu")))
    lines.append("| triton-ascend | %s（backend=%s, kernel=%s） |" % (
        sw.get("triton"), sw.get("triton_ascend_backend"), sw.get("triton_npu_kernel")))
    lines.append("| MindSpeed-MM | %s |" % sw.get("mindspeed_mm_tag"))
    lines.append("")
    lines.append("## 2. 配置档选择\n")
    lines.append("**%s** — %s\n" % (profile.get("profile"), profile.get("note", "")))
    lines.append("| 参数 | 值 | 依据 |")
    lines.append("|---|---|---|")
    lines.append("| world_size / dp | %s / %s | %s |" % (
        profile.get("world_size"), profile.get("dp"),
        "与官方基线同拓扑（逐点可比）" if profile.get("geometry_matches_official", True)
        else "die 数不足，仅窗口口径可比"))
    lines.append("| mbs / gas / GBS | %s / %s / %s | GBS=8 为官方红线 |" % (
        profile.get("mbs"), profile.get("gas"),
        (profile.get("mbs") or 0) * (profile.get("gas") or 0) * (profile.get("world_size") or 1)))
    lines.append("| 显存节省开关 | recompute=%s chunk_loss=%s act_offload=%s | 大显存档关闭以消除开销 |" % (
        profile.get("recompute"), profile.get("enable_chunk_loss"), profile.get("enable_activation_offload")))
    lines.append("| pregather / prefetch | %s / %s | pregather 消除加载阻塞慢步；prefetch 加深反而变慢 |" % (
        profile.get("pregather"), profile.get("prefetch")))
    lines.append("| 算子后端 | %s | triton 可用则用；不可用降级 ascendc→eager |" % profile.get("operator_backend"))
    lines.append("")
    if env.get("degrade_reasons"):
        lines.append("## 3. 降级说明（degraded=true）\n")
        for r in env["degrade_reasons"]:
            lines.append("- %s" % r)
        lines.append("")
    if points:
        lines.append("## 4. 迁移点 → 落点映射\n")
        lines.append("| 类别 | 命中数 | 落点建议 |")
        lines.append("|---|---|---|")
        MAPPING = {
            "linear_attn_GatedDeltaNet": "MindSpeed-MM GDN 算子层：优先 NPU Triton 实现（本机已验证），个别 kernel 可下沉 Ascend C",
            "full_attn": "复用 CANN/MindSpeed 现成注意力实现",
            "vision_3d_conv_patch": "CANN 标准 Conv3d + 图算融合",
            "custom_act_or_norm": "CANN 等价算子映射（RMSNorm/SiLU/quick_gelu）",
        }
        cats = points.get("categories") or points.get("summary") or {}
        if isinstance(cats, dict):
            for k, v in cats.items():
                n = v.get("count") if isinstance(v, dict) else v
                lines.append("| `%s` | %s | %s |" % (k, n, MAPPING.get(k, "按 fsdp2 迁移指南映射")))
        lines.append("")
    lines.append("## 5. 生成产物与校验\n")
    lines.append("训练配置：`%s`\n" % cfg_path)
    lines.append("| 字段 | 目标值 | 操作 |")
    lines.append("|---|---|---|")
    for name, val, action, ok in report:
        lines.append("| `%s` | %s | %s |" % (name, val, action))
    lines.append("")
    if warnings:
        lines.append("**校验警告**：\n")
        for w in warnings:
            lines.append("- %s" % w)
        lines.append("")
    lines.append("## 6. 下一步\n")
    lines.append("1. P3 算子验证：`python3 scripts/30_verify_ops.py`")
    lines.append("2. P4 资产准备：`python3 scripts/40_prepare_assets.py`")
    lines.append("3. P5 训练：`python3 scripts/50_train.py --config %s`" % cfg_path)
    lines.append("4. P6 性能：`python3 scripts/60_bench.py --log <train.log>`")
    lines.append("5. P7 判定：`python3 scripts/70_judge.py --log <train.log>`")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
