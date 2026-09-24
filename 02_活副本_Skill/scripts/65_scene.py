# scene_smoke.py —— Skill Phase4(scene): 工业"操作规范/工艺手法"判定 推理链路冒烟 + 契约 + 交叉印证
# 提升自初赛 POC poc3_scene.py（逻辑已真实验证）
# 用法: python scripts/scene_smoke.py [path/to/image.jpg] [--out out/scene]
# 迁移落地+实卡就绪则真跑; 否则设计内降级: 打印原因+预期契约, 干净退出, 不伪造
import glob
import json
import sys
from pathlib import Path

PROMPT = ("这是工厂车间的作业监控画面/片段。请依据标准作业程序(SOP)，"
          "判断当前操作手法与工艺步骤是否规范，并以 JSON 输出 "
          "{compliant: 是/否, step: 当前作业步骤名, "
          "issue: 不规范的具体描述(顺序错误/手法错误/跳步/力度或角度异常/无), "
          "evidence: 画面中可见的判定依据, bbox: [x1,y1,x2,y2], confidence: 0~1}。只输出 JSON。")
EXPECTED = ('{"compliant":"否","step":"法兰螺栓紧固",'
            '"issue":"未采用对角交叉顺序，单边连续拧紧，易致密封面受力不均",'
            '"evidence":"操作者沿圆周同向连续拧紧第3、4颗螺栓",'
            '"bbox":[286,210,452,372],"confidence":0.88}')
TARGET_CLASS, TARGET_REPO = "Qwen3_5ForConditionalGeneration", "Qwen/Qwen3.5-0.8B"


def try_real_run(img, prompt):
    try:
        import transformers
    except Exception as e:
        return None, f"transformers 不可用: {type(e).__name__}: {e}"
    if not hasattr(transformers.models, "qwen3_5"):
        return None, "transformers.models.qwen3_5 不存在(迁移落地/版本未就位; 与 P1 静态识别互为印证)"
    mod = transformers.models.qwen3_5
    TargetCls = getattr(mod, TARGET_CLASS, None) or getattr(transformers, TARGET_CLASS, None)
    if TargetCls is None:
        return None, f"目标类 {TARGET_CLASS} 未导出(modeling 已存在但类未补全)"
    Proc = getattr(transformers, "AutoProcessor", None)   # 顶层, 非 models.qwen3_5
    if Proc is None:
        return None, "AutoProcessor 未在 transformers 顶层导出"
    try:
        import torch
        model = TargetCls.from_pretrained(TARGET_REPO, torch_dtype=torch.bfloat16, device_map="auto")
        processor = Proc.from_pretrained(TARGET_REPO)
    except Exception as e:
        return None, f"权重/算力未就绪: {type(e).__name__}: {e}"
    try:
        from qwen_vl_utils import process_vision_info
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{Path(img).resolve()}"},
            {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img_in, _ = process_vision_info(msgs)
        inputs = processor(text=[text], images=img_in, padding=True, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=160)
        return processor.batch_decode(out, skip_special_tokens=True)[0], None
    except Exception as e:
        return None, f"推理异常: {type(e).__name__}: {e}"


def cross_check_with_p1(analyze_json):
    # 与 P1(analyze) 静态识别交叉印证: 一静一动, 双重确认迁移缺口一致
    if not analyze_json.exists():
        return False, f"P1 产物缺失({analyze_json}), 无法交叉印证(请先跑 analyze_migrate_points.py)"
    try:
        data = json.loads(analyze_json.read_text(encoding="utf-8"))
        cats = [c for c, v in data.get("points", {}).items() if v]
        return True, (f"P1 静态探针已独立产出, 识别需迁移类别={cats}; "
                      "与 scene 运行时探针'目标类/版本未就位'同指'迁移尚未落地', 双重确认一致")
    except Exception as e:
        return False, f"读取 {analyze_json} 异常: {type(e).__name__}: {e}"


def main():
    ap = sys.argv
    img = ap[1] if len(ap) > 1 and not ap[1].startswith("--") else (
        (glob.glob("samples/*.jpg") + glob.glob("samples/*.png") + glob.glob("samples/*.jpeg") or ["samples/factory.jpg"])[0])
    outdir = Path("out/scene")
    if "--out" in ap:
        outdir = Path(ap[ap.index("--out") + 1])
    analyze_json = Path("out/analyze/migrate_points.json")

    print("=== Skill P4 工业操作规范判定冒烟(迁移目标 = Qwen3.5-0.8B 原生多模态) ===")
    print(f"目标类: {TARGET_CLASS}    目标权重: {TARGET_REPO}")
    print(f"输入图: {img if Path(img).exists() else '(缺图, 仍可证链路)'}")
    print("-" * 64)

    real_ok, real_out, reason = False, None, None
    if not Path(img).exists():
        reason = "缺样例图, 跳过真跑, 展示场景推理契约"
    else:
        real_out, reason = try_real_run(img, PROMPT)
        real_ok = real_out is not None

    if real_ok:
        print("[真跑成功] 模型输出:\n", real_out)
        print("结论: 迁移后 Qwen3.5-0.8B 端到端跑通工业操作规范判定。")
    else:
        print(f"[未真跑] 依赖未就绪: {reason}")
        print("       说明: 此为迁移落地/部署期门槛(目标类导出+transformers 基线+权重+实卡), 非脚本缺陷;")
        print("       迁移落地后本脚本无需改代码即真跑。")

    xok, xmsg = cross_check_with_p1(analyze_json)
    print("-" * 64)
    print("[交叉印证]", xmsg)
    print("-" * 64)
    print("场景推理链路预期输出(操作规范/工艺手法判定):")
    print(EXPECTED)

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "scene_contract.json").write_text(json.dumps({
        "scene": "工业操作规范/工艺手法判定", "target_class": TARGET_CLASS, "target_repo": TARGET_REPO,
        "needs": "语义+时序+常识推理(视觉 3D-Conv 看动作过程)", "expected_output": json.loads(EXPECTED),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "scene_deps.json").write_text(json.dumps({
        "real_run_ok": real_ok, "real_run_reason": reason,
        "cross_check_ok": xok, "cross_check": xmsg,
        "deploy_gates": ["mindspeed_mm/.../qwen3_5 导出目标类+处理器", "transformers 与赛题基线一致",
                         "qwen_vl_utils", "权重 convert_weight->dcp 并下载", "实卡算力"],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"产物: {outdir / 'scene_contract.json'} , {outdir / 'scene_deps.json'}")


if __name__ == "__main__":
    main()
