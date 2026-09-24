#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fingerprint_cfg.py — R2-SK04 · 配置指纹门（可比性机器化三件套之一，来源 R2-C6-01）

功能
----
从"生效 config"（A2 上 runs/run_<tag>/config.yaml；或任意本地测试 yaml；或训练日志内嵌的
Configuration Details —— --config-from-log）提取 **7+2 维指纹**，与目标基准声明指纹
（configs/baselines/officialA.yaml / officialB.yaml，A2 审计 §3.1 的 7 项偏离清单 + 出处）
做 diff，输出：
  1. 偏离表（每维：run 值 + 证据指针=config 行号，baseline 声明值 + 出处 file:line）；
  2. 派生指纹 N_eff = world×mbs×gas（每步样本数）与 loss 打印语义注释；
  3. 可达性谓词：pointwise_feasible（全等+N_eff 同）/ window_feasible（GBS_eff=8）/ none；
  4. 证据门（gates）：数据字节身份/样本序/数值路径等"声明的逐点可比"前提是否闭合。
退出码：0=成功（--gate 时绿/可逐点），2=参数/IO/缺依赖，3=黄/仅窗口口径（--gate），
       4=红/不可比（--gate；不喂 S-M3，对应 R2-C6-01"门失败 exit 4 不烧 GPU"）。

确定性：输出 JSON 不含墙钟/随机；同输入字节必同输出（judge 的 verdict_id 依赖此性质）。

纯 CPU；依赖：Python 3.8+ 标准库 + PyYAML（yaml 解析；缺 pyyaml 时打印安装提示并退出 2，
             但 --config-from-log 的降级行扫描仍可用）。不依赖 torch/torch_npu/A2。

用法
----
  python3 scripts/fingerprint_cfg.py --config <run_config.yaml> --baseline officialB \
      [--world-size 1] [--data-json <run数据集json>] [--out fingerprint.json]
  python3 scripts/fingerprint_cfg.py --config-from-log <train.log> --baseline officialA ...
      # --config-from-log：直接解析日志内嵌 Configuration Details（= 运行态生效配置，
      #   自带 world_size/data_parallel_size → 无需 --world-size）
  python3 scripts/fingerprint_cfg.py ... --gate     # 按门色退出：0 绿/3 黄/4 红
  python3 scripts/fingerprint_cfg.py --baseline-list # 列出可用基线

A2 单卡 COCO 示例（真实 config + 真实日志）：
  python3 scripts/fingerprint_cfg.py \
      --config-from-log runs/run_<tag>/train.log --baseline officialB \
      --data-json /root/coco_dl/extracted/dataset/annotations_slim.json --out evidence/fp.json
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - 环境缺依赖分支
    yaml = None

SKILL_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = SKILL_ROOT / "configs" / "baselines"
DEFAULT_BASELINES = ("officialA", "officialB")

# ---------------------------------------------------------------- 维度规格
# 维度清单与顺序 = A2 审计 §3.1 的 7 项偏离（#1..#7）+ 数据集身份归并进 dataset_file。
# kind 决定比较/归一化语义：
#   bool/int     直接等值比较
#   list_set     freeze：按集合比较（顺序无关）
#   pair         gdn_causal：{gdn, causal} 分字段；任一字段 null = 未设置（缺省语义未知）
#   dataset      dataset_file：比较 file basename；字节身份走 byte_sha256/order_sha256
DIM_SPECS = [
    {"id": "shuffle", "label": "shuffle(数据序)", "path": ("data", "dataloader_param", "shuffle"),
     "leaf": "shuffle", "kind": "bool",
     "base_note": "官方关闭 shuffle 保证 loss 一致（A2 §3.1 #1；PDF 精度对比前提）"},
    {"id": "freeze", "label": "freeze(冻结)", "path": ("model", "freeze"), "leaf": "freeze", "kind": "list_set",
     "base_note": "求和域：[]=在训视觉塔 / [model.visual]=冻结（C4-02 主因开关）"},
    {"id": "cutoff_len", "label": "cutoff_len(截断)", "path": ("data", "dataset_param", "basic_parameters", "cutoff_len"),
     "leaf": "cutoff_len", "kind": "int"},
    {"id": "mbs", "label": "micro_batch_size", "path": ("training", "micro_batch_size"), "leaf": "micro_batch_size", "kind": "int"},
    {"id": "gas", "label": "gradient_accumulation_steps", "path": ("training", "gradient_accumulation_steps"),
     "leaf": "gradient_accumulation_steps", "kind": "int"},
    {"id": "dp", "label": "data_parallel/world", "path": ("parallel", "data_parallel_size"), "leaf": "data_parallel_size",
     "kind": "int", "resolved": True,
     "base_note": "运行态 world_size（yaml 通常不写 dp；tp/ulysses/cp=1 时 dp==world；A2 单卡 world=1）"},
    {"id": "gdn_causal", "label": "gdn/causal 实现", "path": ("model", "gdn_implementation"),
     "path2": ("model", "causal_conv1d_implementation"),
     "leaf": "gdn_implementation", "kind": "pair",
     "base_note": "数值路径：eager / triton(NPU) / ascendc；B=官方口径 triton；缺省语义未钉死→unverified"},
    {"id": "dataset_file", "label": "数据集文件身份", "path": ("data", "dataset_param", "basic_parameters", "dataset"),
     "path_dir": ("data", "dataset_param", "basic_parameters", "dataset_dir"),
     "leaf": "dataset", "kind": "dataset",
     "base_note": "basename 相等=声明层同文件；字节/顺序身份须 data_id.py 实测（官方文件不可得→unresolved）"},
    {"id": "num_workers", "label": "num_workers(dataloader)", "path": ("data", "dataloader_param", "num_workers"),
     "leaf": "num_workers", "kind": "int",
     "base_note": "性能项，理论不影响数值（A2 §3.1 #7）"},
]

REPO_EVIDENCE_DISPLAY = 3   # 每条 dim 的 baseline evidence 在表格最多展示数

# ---------------------------------------------------------------- 工具函数
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def canon(obj) -> str:
    """确定性序列化（JSON），用于输入摘要/哈希链。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _yaml_load_text(text: str):
    if yaml is None:
        raise RuntimeError("缺少 PyYAML（pip install pyyaml）。可用 --config-from-log 的降级行扫描，"
                           "但读取基线声明 yaml 必须 pyyaml。")
    return yaml.safe_load(text)


def _details_region(lines, header="============ Configuration Details ============"):
    """从训练日志里切出 Configuration Details 区（含行号回推用）。返回 (region_lines, first_lineno1)。"""
    i0 = next((i for i, l in enumerate(lines) if header.strip() in l), None)
    if i0 is None:
        raise ValueError("日志中未找到 Configuration Details 段")
    i1 = i0 + 1
    while i1 < len(lines) and "============" not in lines[i1]:
        i1 += 1
    return lines[i0 + 1:i1], i0 + 2  # region 首行 = 原文件第 i0+2 行


def _norm_scalar(v):
    """yaml 标量/内联列表 → 比较用归一值。str 清注释与引号。"""
    if isinstance(v, str):
        return v.strip().strip("'\"")
    if isinstance(v, list):
        return [_norm_scalar(x) for x in v]
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    return v


def _walk(doc, path):
    cur = doc
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None, False
        cur = cur[key]
    return cur, True


def _line_of_leaf(raw_lines, leaf, expect_value=None, depth=None):
    """尽力而为的'证据指针'：返回 (lineno1|None)。规则：候选行 = 缩进对齐+leaf 键；优先
    单行内联值 == expect_value 的行；否则取第一个对齐行。找不到 → None（dim 行号留空）。"""
    candidates = []
    for i, line in enumerate(raw_lines):
        m = re.match(r"^(\s*)" + re.escape(leaf) + r"\s*:\s*(.*)$", line)
        if not m:
            continue
        indent = len(m.group(1))
        if depth is not None and indent != depth * 2:
            continue
        candidates.append((i + 1, m.group(2).strip()))
    if not candidates:
        return None
    if expect_value is not None:
        for no, valtxt in candidates:
            try:
                if _norm_scalar(_yaml_load_text(valtxt)) == expect_value:
                    return no
            except Exception:
                continue
    return candidates[0][0]


def load_config_lines(config_path: Path):
    """返回 (text, raw_lines)。utf-8-sig：容忍 BOM。"""
    text = config_path.read_text(encoding="utf-8-sig", errors="replace")
    return text, text.splitlines()


def load_config_from_log(log_path: Path):
    """从日志 Configuration Details 取生效配置：返回 (text, raw_lines, first_line_of_region)。"""
    raw = log_path.read_text(encoding="utf-8-sig", errors="replace")
    all_lines = raw.splitlines()
    region, first = _details_region(all_lines)
    text = "\n".join(region)
    return text, all_lines, first


def load_baseline(baseline_ref):
    """baseline_ref: officialA | officialB | <path>。返回 (dict, 路径)。"""
    if Path(baseline_ref).is_file():
        p = Path(baseline_ref)
        bid = p.stem
    else:
        bid = baseline_ref
        p = BASELINE_DIR / ("%s.yaml" % bid)
    if not p.is_file():
        raise FileNotFoundError("基线声明文件不存在: %s（可用: officialA/officialB 或绝对路径）" % p)
    doc = _yaml_load_text(p.read_text(encoding="utf-8-sig"))
    if doc.get("schema") != "sk04_baseline.v1":
        raise ValueError("不是 sk04_baseline.v1 基线文件: %s" % p)
    doc["_file"] = str(p)
    return doc, p


def baseline_list():
    out = []
    for f in sorted(BASELINE_DIR.glob("*.yaml")):
        doc = _yaml_load_text(f.read_text(encoding="utf-8-sig"))
        out.append({"id": doc.get("baseline_id"), "file": str(f),
                    "officiality": doc.get("officiality")})
    return out


# ---------------------------------------------------------------- 指纹提取
def norm_dim_value(kind, raw_value, present):
    """归一化 run/baseline 侧值；present=False → (None, False)。"""
    if not present or raw_value is None:
        return None, present and raw_value is not None
    if kind == "list_set":
        vals = raw_value if isinstance(raw_value, list) else [raw_value]
        return tuple(sorted(_norm_scalar(v) for v in vals)), True
    if kind == "pair":
        return raw_value, True  # dict{gdn,causal}
    if kind == "dataset":
        vals = raw_value if isinstance(raw_value, list) else [raw_value]
        first = vals[0] if vals else None
        return first, True  # 全路径字符串
    return _norm_scalar(raw_value), True


def extract_run(doc, raw_lines, world_cli=None):
    """从 yaml doc 提取 9 维 run 值 + 派生。返回 (dims, derived)。"""
    dims = []
    for spec in DIM_SPECS:
        dim_id = spec["id"]
        # ---- 主路径取值
        if dim_id == "dp":
            val, found = _walk(doc, ("parallel", "data_parallel_size"))
            if not found:
                val, found = _walk(doc, ("training", "world_size"))
            src = "config"
            if not found and world_cli is not None:
                val, found, src = world_cli, True, "cli"
            if not found:
                val, found, src = 1, True, "assumed_single_card"
            norm, _ = norm_dim_value("int", val, True)
            line = _line_of_leaf(raw_lines, "data_parallel_size", norm) or \
                   _line_of_leaf(raw_lines, "world_size", norm)
            dims.append({"dim": dim_id, "value": norm, "present": True, "source": src,
                         "line": line, "kind": "int"})
            continue

        path = spec["path"]
        raw_val, found = _walk(doc, path)
        if dim_id == "gdn_causal":
            path2 = spec["path2"]
            raw2, found2 = _walk(doc, path2)
            norm, _ = norm_dim_value("pair", {"gdn": raw_val if found else None,
                                              "causal": raw2 if found2 else None}, True)
            line = _line_of_leaf(raw_lines, spec["leaf"], None)
            line2 = _line_of_leaf(raw_lines, "causal_conv1d_implementation", None)
            dims.append({"dim": dim_id, "value": norm,
                         "present": found or found2,
                         "line": line or line2, "kind": "pair",
                         "fields": {"gdn_present": found, "causal_present": found2}})
            continue
        if dim_id == "dataset_file":
            dir_val, found_dir = _walk(doc, spec["path_dir"])
            norm, _ = norm_dim_value("dataset", raw_val, found)
            line = _line_of_leaf(raw_lines, "dataset", None)
            dims.append({"dim": dim_id,
                         "value": {"file": norm, "dataset_dir": dir_val if found_dir else None},
                         "present": found, "line": line, "kind": "dataset"})
            continue

        norm, present = norm_dim_value(spec["kind"], raw_val, found)
        line = _line_of_leaf(raw_lines, spec["leaf"], norm,
                             depth=len(spec["path"]) - 1 if found else None)
        dims.append({"dim": dim_id, "value": norm, "present": present,
                     "line": line, "kind": spec["kind"]})

    # ---- 派生：world / N_eff / gbs_eff
    dp_dim = next(d for d in dims if d["dim"] == "dp")
    world = world_cli if world_cli is not None else dp_dim["value"]
    mbs_dim = next(d for d in dims if d["dim"] == "mbs")
    gas_dim = next(d for d in dims if d["dim"] == "gas")
    mbs = mbs_dim["value"] if mbs_dim["present"] else None
    gas = gas_dim["value"] if gas_dim["present"] else None
    n_eff = (world * mbs * gas) if (world and mbs is not None and gas is not None) else None
    print_sem = ("全 batch(%d) 均值（dp=1，无 rank 切片）" % n_eff) if (dp_dim["value"] == 1) \
        else "rank0 本地均值（dp=%s，未见打印前 all-reduce 证据 → 逐点语义不可直接复刻）" % dp_dim["value"]
    derived = {
        "world_size": {"value": world, "source": dp_dim["source"]},
        "n_eff": {"value": n_eff, "formula": "world(%s) × mbs(%s) × gas(%s)" % (world, mbs, gas),
                  "note": "每步全局样本数（官方 CSV Samples 每步增量口径）"},
        "gbs_eff": n_eff,
        "loss_print_semantics": print_sem,
    }
    return dims, derived


# ---------------------------------------------------------------- 比较
def cmp_values(kind, run_rec, base_val):
    """逐维比较。返回 (equal_bool, status, detail)。status ∈ match|match_declared|mismatch|
    unverified|both_unset。"""
    run_val = run_rec.get("value")
    run_present = run_rec.get("present", True)

    if kind == "pair":
        if not isinstance(run_val, dict) or not isinstance(base_val, dict):
            return False, "unverified", "pair 结构异常"
        gd, cd = run_val.get("gdn"), run_val.get("causal")
        gb, cb = base_val.get("gdn"), base_val.get("causal")
        run_set = [x for x in (gd, cd) if x is not None]
        base_set = [x for x in (gb, cb) if x is not None]
        if not run_set and not base_set:
            return True, "both_unset", "双方均未设置（缺省）；等值（无差异可言）"
        if not base_set and run_set:
            return False, "unverified", "baseline 未设置(缺省)；run 显式 %s —— 缺省==%s 未钉死，不作等值断言" \
                   % (run_set, run_set)
        if not run_set and base_set:
            return False, "unverified", "run 未设置(缺省)；baseline 显式 %s —— 缺省语义未知" % base_set
        return (gd == gb and cd == cb), ("match" if gd == gb and cd == cb else "mismatch"), ""

    if kind == "dataset":
        run_file = (run_val or {}).get("file")
        base_file = (base_val or {}).get("file")
        if run_file is None or base_file is None:
            return False, "unverified", "run/baseline 数据集声明缺失"
        rb = Path(str(run_file)).name
        bb = Path(str(base_file)).name
        if rb != bb:
            return False, "mismatch", "basename 不同: run=%s vs baseline=%s" % (rb, bb)
        return True, "match_declared", "basename 相同（%s）；字节身份见 data_identity（官方文件不可得→unresolved）" % rb

    if kind == "list_set":
        rn = tuple(sorted(run_val)) if run_val is not None else None
        bn = tuple(sorted(base_val)) if base_val is not None else None
        equal = rn == bn
        return equal, ("match" if equal else "mismatch"), ""

    # bool / int / str
    if run_val is None and base_val is None:
        return True, "both_unset", ""
    if run_val is None or base_val is None:
        return False, "unverified", "单侧未设置（run=%s, baseline=%s）" % (run_val, base_val)
    equal = run_val == base_val
    return equal, ("match" if equal else "mismatch"), ""


def compute_reachability(dims, derived, base_derived):
    """纯函数：可达性谓词（spec 口径）。返回 (level, level_rule, gates, none_reason)。"""
    gates = []
    bad = [d for d in dims if d["status"] == "mismatch"]
    unk = [d for d in dims if d["status"] == "unverified"]
    neff_run = derived["n_eff"]["value"]
    neff_base = base_derived.get("n_eff", {}).get("value")

    # 数据集字节/顺序身份门
    di = derived.get("_data_identity", {})
    if di.get("status") == "same_bytes":
        gates.append({"gate": "data_identity", "closed": True,
                      "detail": "run 数据集与 baseline 声明字节一致（json_sha256=%s）" % di.get("json_sha256")})
    elif di.get("status") == "unresolved":
        gates.append({"gate": "data_identity", "closed": False,
                      "detail": di.get("note", "官方/baseline 数据文件字节不可得 → unresolved；逐点'同数据'前提未闭合" )})
    else:
        gates.append({"gate": "data_identity", "closed": True,
                      "detail": "run 数据集文件不存在（--data-json 未给或路径不可读）→ 字节身份未核对"})

    # 样本序门（shuffle + 数据字节）
    run_sh = next((d for d in dims if d["dim"] == "shuffle"), None)
    if di.get("status") == "same_bytes" and run_sh and run_sh["value"] is False:
        gates.append({"gate": "sample_order", "closed": True,
                      "detail": "shuffle=False 且数据字节一致 → 声明层同序（逐步 batch 组成仍建议 sampler_probe 确认）"})
    else:
        gates.append({"gate": "sample_order", "closed": False,
                      "detail": "shuffle 未关 或 数据字节未核 → 每步 batch 组成未证实相同（corr 相位噪声结构项，C6-02）"})

    mismatch_dims = [d["dim"] for d in bad]
    unverified_dims = [d["dim"] for d in unk]

    if not bad and not unk and neff_run is not None and neff_base is not None and neff_run == neff_base:
        level = "pointwise_feasible"
        level_rule = "7 维全等 + N_eff 同（run=%d = baseline=%d）" % (neff_run, neff_base)
    elif (neff_run is not None and neff_base is not None
          and neff_run == 8 and neff_base == 8):
        level = "window_feasible"
        level_rule = "GBS_eff=8（run=%d = baseline=%d）；逐点不可达（配置偏差或数据序未闭合）→ 仅窗口均值口径" % (neff_run, neff_base)
    else:
        level = "none"
        level_rule = "N_eff/GBS_eff 不可比（run=%s, baseline=%s）→ 不喂 S-M3" % (neff_run, neff_base)
    return level, level_rule, gates, {"mismatch": mismatch_dims, "unverified": unverified_dims}


# ---------------------------------------------------------------- 输出
def run_fingerprint(args):
    # ---- 加载 run 配置（yaml 文件 / 日志 details）
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_file():
            raise FileNotFoundError("config 不存在: %s" % cfg_path)
        text, raw_lines = load_config_lines(cfg_path)
        cfg_sha = sha256_file(cfg_path)
        cfg_source = {"kind": "yaml_file", "path": str(cfg_path), "sha256": cfg_sha}
    else:
        log_path = Path(args.config_from_log)
        if not log_path.is_file():
            raise FileNotFoundError("log 不存在: %s" % log_path)
        text, raw_lines, _first_region_line = load_config_from_log(log_path)
        cfg_sha = sha256_file(log_path)
        cfg_source = {"kind": "log_configuration_details", "path": str(log_path), "sha256": cfg_sha}

    # ---- run 侧取值（日志 details 自带 world_size → 无需 --world-size）
    doc = None
    parsed_by = "yaml"
    try:
        doc = _yaml_load_text(text)
    except Exception as e:
        parsed_by = "line_scan_fallback"
        doc = {}
        # 降级：不解析（日志 details 理论上可 yaml 解析；此处仅留提示）
        if not args.config_from_log:
            raise ValueError("config 非合法 YAML（%s）。若目标是训练日志，请用 --config-from-log。"
                             % e)
    dims, derived = extract_run(doc or {}, raw_lines, args.world_size)

    # ---- data 身份（--data-json 指向 run 侧实际数据集 json）
    data_identity = {"status": "unresolved", "note": "未提供 --data-json（未核对字节身份）"}
    if args.data_json:
        dp = Path(args.data_json)
        if dp.is_file():
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            try:
                import data_id as _did
                ident = _did.identity_of_file(str(dp))
                data_identity = {"status": "run_file_present", "file": str(dp),
                                 "json_sha256": ident["json_sha256"],
                                 "order_sha256": ident["order_sha256"],
                                 "row_count": ident["row_count"],
                                 "note": "与 baseline 声明字节比较见 identity 段（官方文件 byte_sha256=null → 不可比）"}
            except Exception as e:  # data_id 异常不阻断指纹门
                data_identity = {"status": "run_file_present",
                                 "note": "data_id 调用失败（%s）→ 字节身份未核对" % e}
        else:
            data_identity = {"status": "unresolved", "note": "--data-json 路径不可读: %s" % dp}
    derived["_data_identity"] = data_identity
    # run 侧文件 basename → 与 baseline 的字节比较（baseline byte_sha256 非空才可判同字节）
    run_basename = None
    df_rec = next(d for d in dims if d["dim"] == "dataset_file")
    if isinstance(df_rec.get("value"), dict) and df_rec["value"].get("file"):
        run_basename = Path(str(df_rec["value"]["file"])).name

    # ---- baseline 侧
    base_doc, base_path = load_baseline(args.baseline)
    base_fp = base_doc["fingerprint"]
    base_derived = base_doc["derived"]

    # ---- 逐维 diff
    rows = []
    for spec in DIM_SPECS:
        dim_id = spec["id"]
        run_rec = next(d for d in dims if d["dim"] == dim_id)
        base_entry = base_fp.get(dim_id, {})
        base_val = base_entry.get("value")
        equal, status, detail = cmp_values(spec["kind"], run_rec, base_val)
        if spec["kind"] == "dataset":
            bf = base_val.get("file") if isinstance(base_val, dict) else None
            if bf and run_basename == Path(str(bf)).name and data_identity.get("json_sha256") \
                    and base_entry.get("byte_sha256"):
                if data_identity["json_sha256"] == base_entry["byte_sha256"]:
                    equal, status, detail = True, "match", "字节一致（same_bytes）"
                    data_identity["status"] = "same_bytes"
                    data_identity["detail"] = "run json_sha256 == baseline 声明 byte_sha256"
                else:
                    equal, status, detail = False, "mismatch", "字节不一致（run=%s vs baseline 声明=%s）" \
                        % (data_identity["json_sha256"], base_entry["byte_sha256"])
        base_ev = [{"file": e.get("file"), "line": e.get("line")} for e in base_entry.get("evidence", [])]
        run_out = {"present": run_rec.get("present", True), "value": run_rec.get("value"),
                   "config_line": run_rec.get("line")}
        if run_rec.get("source"):
            run_out["source"] = run_rec["source"]
        rows.append({
            "dim": dim_id, "label": spec["label"],
            "run": run_out,
            "baseline": {"value": base_val,
                         "evidence": base_ev[:REPO_EVIDENCE_DISPLAY],
                         "officiality": base_doc.get("officiality")},
            "equal": equal, "status": status, "detail": detail,
        })

    # ---- 派生比较
    neff_run = derived["n_eff"]["value"]
    neff_base = base_derived.get("n_eff", {}).get("value")
    base_world = base_derived.get("world_size")
    base_print_sem = base_derived.get("loss_print_semantics")
    print_sem_out = {"run": derived["loss_print_semantics"]}
    if base_print_sem:
        print_sem_out = {"run": derived["loss_print_semantics"],
                         "baseline": base_print_sem,
                         "note": "dp 打印语义不同 → 逐点 loss 不可直接复刻（A2 §3.2）" if
                         derived["loss_print_semantics"] != base_print_sem else "打印语义一致"}
    derived_out = {
        "N_eff": {"run": neff_run, "baseline": neff_base, "equal": (neff_run == neff_base),
                  "formula_run": derived["n_eff"]["formula"],
                  "formula_base": base_derived.get("n_eff", {}).get("formula")},
        "GBS_eff": {"run": derived["gbs_eff"], "baseline": base_derived.get("gbs_eff"),
                    "equal": derived["gbs_eff"] == base_derived.get("gbs_eff")},
        "world_size": {"run": derived["world_size"]["value"], "baseline": base_world,
                       "run_source": derived["world_size"]["source"]},
        "loss_print_semantics": print_sem_out,
    }

    # ---- 可达性谓词 + gates
    level, level_rule, gates, none_reason = compute_reachability(rows, derived, base_derived)
    open_gates = [g for g in gates if not g["closed"]]
    color = "green"
    if level == "pointwise_feasible" and not open_gates:
        color = "green"
    elif level == "none":
        color = "red"
    else:
        color = "yellow"   # window 或 pointwise 带未闭合门
    reachability = {
        "level": level, "rule": level_rule,
        "color_hint": color,   # green=可逐点 / yellow=仅窗口或候选逐点(缺证据) / red=不喂 S-M3
        "gates": gates,
        "open_gates": [g["gate"] for g in open_gates],
        "none_reason": none_reason,
    }

    out = {
        "schema": "sk04_fingerprint_cfg.v1",
        "generated_by": "fingerprint_cfg.py",
        "config": cfg_source,
        "baseline": {"id": base_doc.get("baseline_id"), "file": str(base_path),
                     "officiality": base_doc.get("officiality"),
                     "officiality_note": base_doc.get("officiality_note"),
                     "role": base_doc.get("role")},
        "parsed_by": parsed_by,
        "dims": rows,
        "derived": derived_out,
        "data_identity": data_identity,
        "dataset_basename_run": run_basename,
        "reachability": reachability,
        "inputs_digest": sha256_bytes((cfg_sha + "|" + sha256_file(base_path)
                                       + "|world=%s" % (args.world_size or "-")).encode("utf-8")),
    }
    return out


def main():
    ap = argparse.ArgumentParser(description="R2-SK04 配置指纹门（7+2 维偏离 + 可达性谓词）",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="示例见文件头 docstring")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--config", help="生效 config yaml（A2: runs/run_<tag>/config.yaml）")
    g.add_argument("--config-from-log", help="从训练日志内嵌 Configuration Details 取生效配置")
    ap.add_argument("--baseline", default="officialB", help="officialA | officialB | <yaml 路径>（默认 officialB）")
    ap.add_argument("--baseline-list", action="store_true", help="列出可用基线并退出")
    ap.add_argument("--world-size", type=int, default=None, help="world_size（A2 实卡数；--config-from-log 时自动取）")
    ap.add_argument("--data-json", default=None, help="run 侧数据集 json 绝对路径（核对字节/顺序身份）")
    ap.add_argument("--out", default=None, help="JSON 输出路径（默认 stdout）")
    ap.add_argument("--gate", action="store_true",
                    help="按门色退出：0=绿(逐点)/3=黄(仅窗口)/4=红(不喂 S-M3)；默认一律 0")
    args = ap.parse_args()

    if args.baseline_list:
        for b in baseline_list():
            print("BASELINE id=%s officiality=%s file=%s" % (b["id"], b["officiality"], b["file"]))
        return 0

    if not (args.config or args.config_from_log):
        ap.error("必须提供 --config 或 --config-from-log（或单独用 --baseline-list）")

    try:
        out = run_fingerprint(args)
    except FileNotFoundError as e:
        print("FATAL %s" % e, file=sys.stderr)
        return 2
    except ValueError as e:
        print("FATAL %s" % e, file=sys.stderr)
        return 2
    except RuntimeError as e:
        print("FATAL %s" % e, file=sys.stderr)
        return 2

    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n",
                                  encoding="utf-8")
    else:
        print(json.dumps(out, ensure_ascii=False, indent=1))

    # ---- 一行机器可读摘要
    neff = out["derived"]["N_eff"]
    mism = [r["dim"] for r in out["dims"] if r["status"] == "mismatch"]
    unk = [r["dim"] for r in out["dims"] if r["status"] == "unverified"]
    print("FINGERPRINT_OK baseline=%s officiality=%s level=%s color=%s "
          "n_eff_run=%s n_eff_base=%s gbs_run=%s mismatch=[%s] unverified=[%s] "
          "open_gates=[%s]" % (out["baseline"]["id"], out["baseline"]["officiality"],
                               out["reachability"]["level"], out["reachability"]["color_hint"],
                               neff["run"], neff["baseline"], out["derived"]["GBS_eff"]["run"],
                               ",".join(mism), ",".join(unk),
                               ",".join(out["reachability"]["open_gates"])))
    if args.gate:
        code = {"green": 0, "yellow": 3, "red": 4}[out["reachability"]["color_hint"]]
        return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
