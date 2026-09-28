#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20_plan_migration.py — P2 迁移方案与配置生成

输入：out/probe/env.json（P0 产物）、out/analyze/migrate_points.json（P1 产物，可选）
输出：
  out/plan/train_config.yaml   — 显式参考/候选角色的有效训练配置
  out/plan/config_manifest.json — 角色、基线身份、配置哈希和参考差异
  out/plan/migrate_plan.md     — 迁移点 → 落点映射方案（人读）

设计要点（冗余/健壮）：
  * 不走 sed：用"逐行状态机"精确改 yaml，**每处改动都回读校验**，失败即报错退出（避免静默失败）
  * 默认参考角色取已保存 Triton 日志中可核对字段；候选必须显式选择并记录差异
  * 环境不支持参考时标明 blocked，不将自动降级称为等价参考

退出码：0 成功；2 输入缺失/IO 错误；3 配置生成校验失败
"""

import argparse
import hashlib
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
    (("training",), "save_interval", 2),
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
    (("model",), "skip_flash_attn_recompute", 2),
]

# Fields observed in the saved 2026-07-24 Triton reference log. This is a
# configuration reference, not proof of identical weights/data or official rules.
REFERENCE_BASELINE_ID = "qwen35-0p8b-triton-20260724-100step-v1"
REFERENCE_LOG_SHA256 = "c8daabce532d0d7b5a3ffa1668ff490a3ddf8d95448701d17d2dc64ca1c89296"
REFERENCE_LOG_PATH = os.path.join(SKILL_ROOT, "examples", "train", "official_baseline.log")
REFERENCE_TEMPLATE_PATH = os.path.join(SKILL_ROOT, "config", "templates",
                                       "qwen3_5_0_8B_reference.yaml")
REFERENCE_TARGETS = {
    ("parallel", "data_parallel_size"): 2,
    ("parallel", "fully_shard_parallel_size"): 2,
    ("parallel", "fsdp_plan", "pregather"): False,
    ("parallel", "fsdp_plan", "num_to_forward_prefetch"): 1,
    ("parallel", "fsdp_plan", "num_to_backward_prefetch"): 1,
    ("training", "micro_batch_size"): 4,
    ("training", "gradient_accumulation_steps"): 1,
    ("training", "save_interval"): 100,
    ("training", "load_rank0_and_broadcast"): False,
    ("training", "save_format"): "auto",
    ("features", "recompute"): True,
    ("features", "enable_chunk_loss"): True,
    ("features", "enable_activation_offload"): True,
    ("model", "gdn_implementation"): "triton",
    ("model", "causal_conv1d_implementation"): "triton",
    ("model", "skip_gdn_recompute"): True,
    ("model", "skip_flash_attn_recompute"): True,
}


def candidate_targets(profile):
    """Environment-specific candidate values; never used by reference role."""
    return {
        ("parallel", "data_parallel_size"): profile.get("world_size", 1),
        ("parallel", "fully_shard_parallel_size"): "auto",
        ("parallel", "fsdp_plan", "pregather"): bool(profile.get("pregather", False)),
        ("parallel", "fsdp_plan", "num_to_forward_prefetch"): profile.get("prefetch", 1),
        ("parallel", "fsdp_plan", "num_to_backward_prefetch"): profile.get("prefetch", 1),
        ("training", "micro_batch_size"): profile.get("mbs", 4),
        ("training", "gradient_accumulation_steps"): profile.get("gas", 1),
        ("training", "save_interval"): 10000,
        ("training", "load_rank0_and_broadcast"): bool(profile.get("load_rank0_and_broadcast", False)),
        ("training", "save_format"): profile.get("save_format", "auto"),
        ("features", "recompute"): bool(profile.get("recompute", False)),
        ("features", "enable_chunk_loss"): bool(profile.get("enable_chunk_loss", False)),
        ("features", "enable_activation_offload"): bool(profile.get("enable_activation_offload", False)),
        ("model", "gdn_implementation"): profile.get("operator_backend", "triton"),
        ("model", "causal_conv1d_implementation"): profile.get("operator_backend", "triton"),
        ("model", "skip_gdn_recompute"): bool(profile.get("skip_gdn_recompute", True)),
        ("model", "skip_flash_attn_recompute"): True,
    }


def _nested_get(doc, path):
    for key in path:
        doc = doc.get(key) if isinstance(doc, dict) else None
    return doc


def config_leaves(value, prefix=()):
    """Flatten YAML mappings while retaining empty mappings and explicit null."""
    if isinstance(value, dict):
        if not value:
            yield ".".join(prefix), {}
        else:
            for key, child in value.items():
                yield from config_leaves(child, prefix + (str(key),))
    else:
        yield ".".join(prefix), value


def config_differences(actual, reference):
    """Every YAML leaf difference, preserving absent versus explicit null."""
    actual_leaves, reference_leaves = dict(config_leaves(actual)), dict(config_leaves(reference))
    out = []
    for field in sorted(actual_leaves.keys() | reference_leaves.keys()):
        ref_present, eff_present = field in reference_leaves, field in actual_leaves
        if ref_present != eff_present or (ref_present and reference_leaves[field] != actual_leaves[field]):
            out.append({"field": field, "reference_present": ref_present,
                        "reference": reference_leaves.get(field),
                        "effective_present": eff_present,
                        "effective": actual_leaves.get(field)})
    return out


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


def migration_counts(payload):
    """Validate P1 hits; accept legacy count summaries explicitly."""
    if not isinstance(payload, dict):
        raise ValueError("迁移点 JSON 顶层必须是对象")
    field = next((k for k in ("points", "categories", "summary") if k in payload), None)
    if field is None or not isinstance(payload[field], dict) or not payload[field]:
        raise ValueError("需要非空 points 对象（兼容 categories/summary 计数格式）")
    counts = {}
    for category, value in payload[field].items():
        if not isinstance(category, str) or not category.strip():
            raise ValueError("迁移类别必须是非空字符串")
        if field == "points":
            if not isinstance(value, list):
                raise ValueError("points.%s 必须是命中列表" % category)
            for hit in value:
                if (not isinstance(hit, dict) or type(hit.get("line")) is not int
                        or hit["line"] < 0 or not isinstance(hit.get("sym"), str)
                        or not hit["sym"].strip()):
                    raise ValueError("points.%s 的命中项需要非负整数 line 和非空 sym" % category)
            count = len(value)
        else:
            count = value.get("count") if isinstance(value, dict) else value
            if type(count) is not int or count < 0:
                raise ValueError("%s.%s 必须包含非负整数计数" % (field, category))
        counts[category] = count
    return counts


def main():
    ap = argparse.ArgumentParser(description="P2 迁移方案与配置生成（按环境自适应）")
    ap.add_argument("--env", default="out/probe/env.json", help="P0 产物 env.json")
    ap.add_argument("--points", default="out/analyze/migrate_points.json", help="P1 产物（省略时允许仅生成配置；显式指定必须有效）")
    ap.add_argument("--template", default=os.path.join(SKILL_ROOT, "config", "templates", "qwen3_5_0_8B_base.yaml"),
                    help="训练配置模板 yaml")
    ap.add_argument("--out", default="out/plan", help="输出目录")
    ap.add_argument("--data-json", default=None, help="数据集 json 路径（覆盖模板默认值）")
    ap.add_argument("--data-dir", default=None, help="数据集目录（覆盖模板默认值）")
    ap.add_argument("--weight-hf", default=None, help="hf 权重路径（覆盖模板默认值）")
    ap.add_argument("--weight-dcp", default=None, help="dcp 权重路径（覆盖模板默认值）")
    ap.add_argument("--config-role", choices=("reference", "candidate"), default="reference",
                    help="reference=已保存 Triton 日志的配置参考；candidate=显式采用 P0 环境推荐")
    ap.add_argument("--steps", type=int, default=100, help="训练步数")
    args = ap.parse_args()

    # Validate before writing configuration; reject broken upstream artifacts.
    points = {}
    points_explicit = any(a == "--points" or a.startswith("--points=") for a in sys.argv[1:])
    if os.path.isfile(args.points):
        try:
            with open(args.points, encoding="utf-8") as stream:
                points = json.load(stream)
            migration_counts(points)
        except (OSError, ValueError) as exc:
            print("FATAL 无效 P1 产物 %s: %s" % (args.points, exc), file=sys.stderr)
            return 2
    elif points_explicit:
        print("FATAL 指定的 P1 产物不存在: %s" % args.points, file=sys.stderr)
        return 2
    else:
        print("[WARN] 未提供 P1 产物，仅生成配置；迁移映射未验证")

    # ---- 读环境档案
    if not os.path.isfile(args.env):
        print("FATAL 缺少 P0 产物：%s（先跑 scripts/00_probe_env.py）" % args.env, file=sys.stderr)
        return 2
    env = json.loads(open(args.env, encoding="utf-8").read())
    profile = env.get("recommended_profile", {})
    path = env.get("path", "unknown")
    print("== P2 迁移方案与配置生成 ==")
    print("环境路径 : %s" % path)
    print("配置档   : %s（%s）；角色=%s" %
          (profile.get("profile"), profile.get("note", ""), args.config_role))

    # ---- 读模板
    if not os.path.isfile(args.template):
        print("FATAL 模板不存在：%s" % args.template, file=sys.stderr)
        return 2
    if not os.path.isfile(REFERENCE_TEMPLATE_PATH):
        print("FATAL 独立参考模板不存在：%s" % REFERENCE_TEMPLATE_PATH, file=sys.stderr)
        return 2
    try:
        reference_log_sha = hashlib.sha256(open(REFERENCE_LOG_PATH, "rb").read()).hexdigest()
    except OSError as exc:
        print("FATAL 参考日志不可读：%s" % exc, file=sys.stderr)
        return 2
    if reference_log_sha != REFERENCE_LOG_SHA256:
        print("FATAL 参考日志内容已变化，不能继续绑定基线 ID %s" % REFERENCE_BASELINE_ID,
              file=sys.stderr)
        return 2
    with open(args.template, "rb") as stream:
        template_raw = stream.read()
    with open(REFERENCE_TEMPLATE_PATH, "rb") as stream:
        reference_raw = stream.read()
    text = template_raw.decode("utf-8")
    reference_text = reference_raw.decode("utf-8")
    lines = text.splitlines()

    # Keep the saved-log reference separate from P0's performance/memory advice.
    targets = (REFERENCE_TARGETS if args.config_role == "reference"
               else candidate_targets(profile))

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
                 json.dumps([args.data_json], ensure_ascii=False), 6, "数据集 json")
        # dataset 是列表：模板里可能是内联列表，此处直接按字符串写入（保持引号形式）
    if args.data_dir:
        set_path(("data", "dataset_param", "basic_parameters"), "dataset_dir", json.dumps(args.data_dir, ensure_ascii=False), 6, "数据集目录")
    if args.weight_hf:
        set_path(("model",), "model_name_or_path", json.dumps(args.weight_hf, ensure_ascii=False), 2, "hf 权重")
        set_path(("data", "dataset_param", "preprocess_parameters"), "model_name_or_path", json.dumps(args.weight_hf, ensure_ascii=False), 6, "preprocess 权重")
    if args.weight_dcp:
        set_path(("training",), "load", json.dumps(args.weight_dcp, ensure_ascii=False), 2, "dcp 权重")
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
    try:
        reference_doc = yaml.safe_load(reference_text)
    except Exception as exc:
        print("FATAL 独立参考模板无法解析：%s" % exc, file=sys.stderr)
        return 3
    if not isinstance(reference_doc, dict):
        print("FATAL 独立参考模板顶层不是映射", file=sys.stderr)
        return 3
    reference_mismatches = [".".join(path) for path, expected in REFERENCE_TARGETS.items()
                            if _nested_get(reference_doc, path) != expected]
    if reference_mismatches:
        print("FATAL 独立参考模板关键字段漂移：%s" % ", ".join(reference_mismatches),
              file=sys.stderr)
        return 3
    checks = [(".".join(path), _nested_get(doc, path), expected)
              for path, expected in targets.items()]
    checks.append(("training.train_iters", _nested_get(doc, ("training", "train_iters")), args.steps))
    # Path overrides must round-trip as exact strings/list entries in YAML.
    for keys, expected in (
        (("data", "dataset_param", "basic_parameters", "dataset"), [args.data_json] if args.data_json else None),
        (("data", "dataset_param", "basic_parameters", "dataset_dir"), args.data_dir),
        (("model", "model_name_or_path"), args.weight_hf),
        (("data", "dataset_param", "preprocess_parameters", "model_name_or_path"), args.weight_hf),
        (("training", "load"), args.weight_dcp),
    ):
        if expected is not None:
            checks.append((".".join(keys), _nested_get(doc, keys), expected))
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

    # ---- 几何与角色身份（运行时 world 仍由 P5 强校验）
    _mbs = (doc.get("training") or {}).get("micro_batch_size")
    _gas = (doc.get("training") or {}).get("gradient_accumulation_steps")
    _dp = (doc.get("parallel") or {}).get("data_parallel_size")
    try:
        if any(type(value) is not int or value < 1 for value in (_dp, _mbs, _gas)):
            raise ValueError("dp/mbs/gas 必须为正整数")
        _gbs = _dp * _mbs * _gas
        print("PLAN_GEOMETRY world=%s mbs=%s gas=%s → GBS=%s（官方红线 = 8）" % (_dp, _mbs, _gas, _gbs))
        if _gbs != 8:
            print("FATAL GBS=%s ≠ 8：拒绝产出不可训练配置" % _gbs, file=sys.stderr)
            return 3
    except Exception:
        print("PLAN_GEOMETRY world=%s mbs=%s gas=%s → GBS=不可计算" % (_dp, _mbs, _gas))
        print("FATAL GBS 不可计算：拒绝产出", file=sys.stderr)
        return 3
    if args.config_role == "candidate" and profile.get("dp") is not None \
            and profile["dp"] != profile.get("world_size"):
        print("FATAL P0 dp/world 不一致：拒绝产出", file=sys.stderr)
        return 3

    feasibility_reasons = []
    feasibility_unknown = []
    if args.config_role == "reference":
        if profile.get("world_size") is None:
            feasibility_unknown.append("P0 未提供 world_size")
        elif profile["world_size"] != 2:
            feasibility_reasons.append("探测 world_size=%s；参考配置要求 world_size=2" %
                                       profile.get("world_size"))
        if profile.get("operator_backend") is None:
            feasibility_unknown.append("P0 未提供 operator_backend")
        elif profile["operator_backend"] != "triton":
            feasibility_reasons.append("探测后端=%s；参考日志使用 triton" %
                                       profile.get("operator_backend"))
        if profile.get("load_rank0_and_broadcast") or profile.get("save_format") == "hf":
            feasibility_reasons.append("当前 CANN 需要 DCP 加载/保存工作区；参考配置未采用该工作区")
        if "load_rank0_and_broadcast" not in profile or "save_format" not in profile:
            feasibility_unknown.append("P0 未完整提供 DCP 加载/保存工作区判据")
        if feasibility_reasons:
            warnings.append("参考运行环境未对齐：" + "；".join(feasibility_reasons))
        if feasibility_unknown:
            warnings.append("参考运行环境未核对：" + "；".join(feasibility_unknown))

    differences = config_differences(doc, reference_doc)
    reference_fields = set(dict(config_leaves(reference_doc)))
    verified_reference_fields = sorted(".".join(path) for path in REFERENCE_TARGETS
                                       if path != ("training", "save_format"))
    template_default_fields = sorted(reference_fields - set(verified_reference_fields))
    cfg_path = os.path.join(outdir, "train_config.yaml")
    reference_cfg_path = os.path.join(outdir, "reference_config.yaml")
    manifest = {
        "schema": "migrator_config.v1",
        "role": args.config_role,
        "baseline_id": REFERENCE_BASELINE_ID,
        "baseline_source": {"kind": "repository_log_copy",
                            "path": REFERENCE_LOG_PATH, "sha256": reference_log_sha},
        "official_rule_state": "RULE_PENDING",
        "reference_scope": "saved_log_observed_fields_plus_template_defaults",
        "verified_reference_fields": verified_reference_fields,
        "template_default_fields_unverified": template_default_fields,
        "reference_feasibility": ("blocked" if feasibility_reasons else
                                  "unknown" if feasibility_unknown else "unverified")
                                 if args.config_role == "reference" else "not_assessed",
        "reference_feasibility_reasons": feasibility_reasons,
        "reference_feasibility_unknown": feasibility_unknown,
        "effective_config": {"path": cfg_path,
                             "sha256": hashlib.sha256(text_out.encode("utf-8")).hexdigest()},
        "reference_config": {"path": reference_cfg_path,
                             "sha256": hashlib.sha256(reference_text.encode("utf-8")).hexdigest()},
        "source_inputs": {
            "reference_template": {"path": REFERENCE_TEMPLATE_PATH,
                                   "sha256": hashlib.sha256(reference_raw).hexdigest()},
            "template": {"path": os.path.abspath(args.template),
                         "sha256": hashlib.sha256(template_raw).hexdigest()},
            "env": {"path": os.path.abspath(args.env),
                    "sha256": hashlib.sha256(open(args.env, "rb").read()).hexdigest()},
        },
        "geometry": {"world_size": _dp, "data_parallel_size": _dp,
                     "micro_batch_size": _mbs, "gradient_accumulation_steps": _gas,
                     "global_batch_size": _gbs},
        "differences_from_reference": differences,
    }
    # Only write after YAML, all managed fields and geometry have passed.
    os.makedirs(outdir, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text_out)
    with open(reference_cfg_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(reference_text)
    manifest_path = os.path.join(outdir, "config_manifest.json")
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # ---- 迁移方案文档
    plan_md = build_plan_md(env, profile, points, cfg_path, report, warnings, manifest, doc)
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
    print("  参考: %s" % reference_cfg_path)
    print("  身份: %s" % manifest_path)
    print("  方案: %s" % plan_path)
    if args.config_role == "reference" and feasibility_reasons:
        print("PLAN_BLOCKED role=reference reason=%s" % ";".join(feasibility_reasons))
        return 3
    print("PLAN_OK role=%s profile=%s path=%s" % (args.config_role, profile.get("profile"), path))
    print("下一步: P3 算子验证 → python3 scripts/30_verify_ops.py；或直接 P5 训练 → scripts/50_train.py")
    return 0


def build_plan_md(env, profile, points, cfg_path, report, warnings,
                  config_manifest=None, effective=None):
    sw = env.get("software", {})
    hw = env.get("hardware", {})
    lines = []
    lines.append("# 迁移方案（P2 自动生成）\n")
    if config_manifest:
        lines.append("配置角色：**%s**；参考 ID：`%s`；正式精度规则：**RULE_PENDING**。" %
                     (config_manifest["role"], config_manifest["baseline_id"]))
        lines.append("参考仅覆盖已保存日志中可核对的字段；权重、数据和当前实机尚未证明与参考同源。\n")
        lines.append("参考环境适配：**%s**。%s\n" %
                     (config_manifest["reference_feasibility"],
                      "；".join(config_manifest["reference_feasibility_reasons"] +
                               config_manifest["reference_feasibility_unknown"])))
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
    lines.append("P0 建议档：**%s** — %s\n" % (profile.get("profile"), profile.get("note", "")))
    if effective:
        actual_world = _nested_get(effective, ("parallel", "data_parallel_size"))
        actual_mbs = _nested_get(effective, ("training", "micro_batch_size"))
        actual_gas = _nested_get(effective, ("training", "gradient_accumulation_steps"))
        actual_recompute = _nested_get(effective, ("features", "recompute"))
        actual_chunk = _nested_get(effective, ("features", "enable_chunk_loss"))
        actual_offload = _nested_get(effective, ("features", "enable_activation_offload"))
        actual_pregather = _nested_get(effective, ("parallel", "fsdp_plan", "pregather"))
        actual_forward_prefetch = _nested_get(effective, ("parallel", "fsdp_plan", "num_to_forward_prefetch"))
        actual_backward_prefetch = _nested_get(effective, ("parallel", "fsdp_plan", "num_to_backward_prefetch"))
        actual_backend = _nested_get(effective, ("model", "gdn_implementation"))
    else:
        actual_world, actual_mbs, actual_gas = profile.get("world_size"), profile.get("mbs"), profile.get("gas")
        actual_recompute, actual_chunk = profile.get("recompute"), profile.get("enable_chunk_loss")
        actual_offload, actual_pregather = profile.get("enable_activation_offload"), profile.get("pregather")
        actual_forward_prefetch = actual_backward_prefetch = profile.get("prefetch")
        actual_backend = profile.get("operator_backend")
    lines.append("| 参数 | 值 | 依据 |")
    lines.append("|---|---|---|")
    lines.append("| world_size / dp | %s / %s | %s |" % (
        actual_world, actual_world,
        "仅配置几何与参考一致；实机/数据仍待核" if actual_world == 2
        else "与参考拓扑不同，不可称逐点可比"))
    lines.append("| mbs / gas / GBS | %s / %s / %s | GBS=8 为官方红线 |" % (
        actual_mbs, actual_gas,
        (actual_mbs or 0) * (actual_gas or 0) * (actual_world or 1)))
    lines.append("| 显存节省开关 | recompute=%s chunk_loss=%s act_offload=%s | 本次有效配置 |" % (
        actual_recompute, actual_chunk, actual_offload))
    lines.append("| pregather / forward prefetch / backward prefetch | %s / %s / %s | 本次有效配置 |" % (
        actual_pregather, actual_forward_prefetch, actual_backward_prefetch))
    lines.append("| 算子后端 | %s | 本次有效配置 |" % actual_backend)
    lines.append("")
    if config_manifest:
        lines.append("### 与参考配置的逐字段差异\n")
        for item in config_manifest["differences_from_reference"]:
            ref = str(item["reference"]) if item["reference_present"] else "<缺失>"
            eff = str(item["effective"]) if item["effective_present"] else "<缺失>"
            lines.append("- `%s`: 参考 `%s` → 有效 `%s`" % (item["field"], ref, eff))
        if not config_manifest["differences_from_reference"]:
            lines.append("- 已核对字段无差异；不代表资产或正式精度规则已对齐。")
        lines.append("")
    if env.get("degrade_reasons"):
        lines.append("## 3. 降级说明（degraded=true）\n")
        for r in env["degrade_reasons"]:
            lines.append("- %s" % r)
        lines.append("")
    if points:
        lines.append("## 4. 迁移点 → 落点映射\n")
        lines.append("来源：%s；来源标记：%s\n" % (
            points.get("source", "未提供"), points.get("source_tag", "未提供")))
        if "points" not in points:
            lines.append("兼容旧版计数格式；不包含逐条命中位置。\n")
        lines.append("| 类别 | 命中数 | 落点建议 |")
        lines.append("|---|---|---|")
        MAPPING = {
            "linear_attn_GatedDeltaNet": "MindSpeed-MM GDN 算子层：建议优先 NPU Triton 实现（需实际验证），个别 kernel 可下沉 Ascend C",
            "full_attn": "复用 CANN/MindSpeed 现成注意力实现",
            "vision_3d_conv_patch": "CANN 标准 Conv3d + 图算融合",
            "custom_act_or_norm": "CANN 等价算子映射（RMSNorm/SiLU/quick_gelu）",
        }
        for k, n in migration_counts(points).items():
            lines.append("| `%s` | %s | %s |" % (k, n, MAPPING.get(k, "待人工确认落点")))
        lines.append("")
    else:
        lines.append("## 4. 迁移映射未验证\n\n未提供 P1 产物，本次仅生成配置。\n")
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
