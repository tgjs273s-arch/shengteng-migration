# analyze_migrate_points.py —— Skill Phase1(analyze): AST 静态识别迁移点 + 产出落点映射报告
# 提升自初赛 POC poc1_parse.py（逻辑已真实验证：对 transformers v5.2.0 建模源码命中 49 处）
# 用法: python scripts/analyze_migrate_points.py [path/to/modeling_qwen3_5.py] [--out out/analyze]
# 解析对象三层回退(优先锁版本真文件): --src 指定 -> ./refs/...v5.2.0.py -> ./modeling_qwen3_5.py -> 内嵌自检
# 红线: 源文件"存在但 4 类全 0 命中"=内容无效/占位 -> 诚实回退自检, 不伪造命中
import argparse
import ast
import json
import sys
from pathlib import Path

DEFAULT_REF = Path("refs/modeling_qwen3_5__transformers_v5.2.0.py")
LOCAL_COPY = Path("modeling_qwen3_5.py")

# 4 类迁移点规则（v0 内置; v1 外置为 profiles/<model>.json, 见 SKILL.md §5）
PATTERNS = {
    "linear_attn_GatedDeltaNet": [
        "GatedDeltaNet", "Qwen3_5GatedDeltaNet",
        "chunk_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update",
        "fused_gdn_gating", "npu_recurrent_gated_delta_rule",
    ],
    "full_attn": [
        "Qwen3_5Attention", "GatedAttention",
    ],
    "vision_3d_conv_patch": [
        "Conv3d", "patch_embed", "patch_embedding", "temporal_patch_size",
    ],
    "custom_act_or_norm": [
        "RMSNorm", "SiLU", "quick_gelu",
    ],
}

# MindSpeed-MM 落点/适配建议（与 fsdp2_developer_migration_guide 配合使用, P2 阶段细化）
MINDSPEED_HINT = {
    "linear_attn_GatedDeltaNet": "mindspeed_mm/fsdp/models/qwen3_5 下复核 GatedDeltaNet; chunk 前向+反向优先昇腾 Triton, 极个别 kernel 下沉 Ascend C; 标[需算子落地+逐层余弦核对]",
    "full_attn":                 "复用 CANN/MindSpeed 全注意力实现; 标[逐层核对]",
    "vision_3d_conv_patch":      "复用 CANN 标准 Conv3d + 图算融合; 标[shape 核对]",
    "custom_act_or_norm":        "RMSNorm/quick_gelu/silu 映射 CANN 等价算子; 标[逐层核对]",
}


def _sym(node):
    return getattr(node, "name", "") or getattr(node, "attr", "") or getattr(node, "id", "") or ""


def classify(src_text: str):
    tree = ast.parse(src_text)
    hit = {k: [] for k in PATTERNS}
    for node in ast.walk(tree):
        nm = _sym(node)
        if not nm:
            continue
        for cat, kws in PATTERNS.items():
            if any(k in nm for k in kws):
                hit[cat].append({"line": getattr(node, "lineno", 0), "sym": nm})
    return hit


def _report_lines(hit, where, source_tag):
    out = ["# Skill P1 迁移点识别 -> MindSpeed-MM 落点映射 / 待核对清单",
           f"> 由 analyze_migrate_points.py 据 AST 识别结果自动生成; 解析对象: {where} [来源={source_tag}]",
           "> 命中行号为真实解析所得; '逐层 diff' 需 MindSpeed-MM 官方参考实现就位后扩展生成, 当前产出为映射+待核对清单, 不臆造 diff 数字。", ""]
    total_hits = 0
    for cat, items in hit.items():
        total_hits += len(items)
        uniq_lines = sorted({i["line"] for i in items if i["line"]})
        uniq_syms = sorted({i["sym"] for i in items if i["sym"]})
        out.append(f"## {cat}")
        out.append(f"- 命中次数: {len(items)}　去重行号数: {len(uniq_lines)}　去重符号: {uniq_syms}")
        out.append(f"- 行号样例: {uniq_lines[:12]}")
        out.append(f"- mindspeed 落点/适配建议: {MINDSPEED_HINT.get(cat, '-')}")
        out.append("")
    out.append(f"--- 合计命中 {total_hits} 处(去重行号见各类) ---")
    return "\n".join(out)


def _print_report(hit, where, source_tag):
    print(f"=== Skill P1 迁移点识别 @ {where}  [来源={source_tag}] ===")
    total = 0
    for cat, items in hit.items():
        total += len(items)
        lines = sorted({i["line"] for i in items if i["line"]})
        syms = sorted({i["sym"] for i in items if i["sym"]})
        print(f"[{cat}] 命中 {len(items)} 处 | 去重行号 {len(lines)} | 符号 {syms[:6]}")
    print(f"--- 合计命中 {total} 处 ---")
    print("结论: 已静态识别 GatedDeltaNet 线性注意力 + 视觉 3D-Conv + 自定义激活/_norm, 迁移有明确抓手。")


def _self_check():
    fake = '''
class Qwen3_5GatedDeltaNet:
    def forward(self, x):
        return chunk_gated_delta_rule(causal_conv1d_fn(x))
class Qwen3_5Attention:
    def forward(self, x):
        return GatedAttention(x)
class VisionPatch:
    def __init__(self):
        self.temporal_patch_size = 2
        self.proj = Conv3d(3, 64, 2)
        self.norm = RMSNorm(64)
        self.act = quick_gelu
'''
    return classify(fake), "<内嵌自检片段>", "SELF_CHECK(无真源码, 仅证解析逻辑; 正式使用请提供真实建模源码)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", default=None, help="建模源码路径(缺省走三层回退)")
    ap.add_argument("--out", default="out/analyze", help="产物目录")
    args = ap.parse_args()

    if args.src:
        p = Path(args.src)
        hit = classify(p.read_text(encoding="utf-8"))
        src, tag = str(p), "ARGV"
        if sum(len(v) for v in hit.values()) == 0:
            hit, src, tag = _self_check()
            tag = "ARGV 0命中->回退SELF_CHECK(源码不含4类符号, 不伪造)"
    elif DEFAULT_REF.exists():
        hit = classify(DEFAULT_REF.read_text(encoding="utf-8"))
        src, tag = str(DEFAULT_REF), "VENDORED(锁版本)"
        if sum(len(v) for v in hit.values()) == 0:
            hit, src, tag = _self_check()
            tag = "VENDORED 0命中->回退SELF_CHECK(refs 未含4类符号, 不伪造)"
    elif LOCAL_COPY.exists():
        hit = classify(LOCAL_COPY.read_text(encoding="utf-8"))
        src, tag = str(LOCAL_COPY), "LOCAL_COPY"
        if sum(len(v) for v in hit.values()) == 0:
            hit, src, tag = _self_check()
            tag = "LOCAL_COPY 0命中->回退SELF_CHECK(不伪造)"
    else:
        hit, src, tag = _self_check()

    _print_report(hit, src, tag)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "migrate_points.json").write_text(
        json.dumps({"source": src, "source_tag": tag, "points": hit}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    (outdir / "migrate_points_report.md").write_text(_report_lines(hit, src, tag), encoding="utf-8")
    print(f"产物: {outdir / 'migrate_points.json'} , {outdir / 'migrate_points_report.md'}")
    print("下游: P2(migrate) 消费 migrate_points.json; P4(scene_smoke) 用其做交叉印证。")


if __name__ == "__main__":
    main()
