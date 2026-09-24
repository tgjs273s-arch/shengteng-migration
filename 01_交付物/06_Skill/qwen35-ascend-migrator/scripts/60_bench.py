# bench_run.py —— Skill Phase4(bench): 逐轮性能数据落盘(算子级微基准 v0; 端到端基准由 MindSpeed-MM 侧执行后回填)
# 用法: python scripts/bench_run.py --round 1 --tag baseline [--out out/bench]
# 红线: 没有基线轮(round=1, tag=baseline)就不许出对比结论; 每轮必须落盘 JSON + 对应 log 编号
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from verify_ops import probe_backend  # 复用 P0 双轨探测


def _time_op(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available() or _has_npu():
        torch.cuda.synchronize() if torch.cuda.is_available() else torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available() or _has_npu():
        torch.cuda.synchronize() if torch.cuda.is_available() else torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms/iter


def _has_npu():
    try:
        return torch.npu.is_available()
    except Exception:
        return False


def run_micro(device):
    import torch.nn.functional as F
    rows = []
    # 微基准1: 全注意力 sdpa
    q = torch.randn(1, 8, 128, 64, device=device)
    k = torch.randn(1, 8, 128, 64, device=device)
    v = torch.randn(1, 8, 128, 64, device=device)
    rows.append({"op": "sdpa(B=1,H=8,T=128,D=64)",
                 "ms_per_iter": round(_time_op(lambda: F.scaled_dot_product_attention(q, k, v)), 3)})
    # 微基准2: RMSNorm(手写, 同 P3)
    h = torch.randn(8, 128, 512, device=device)
    w = torch.ones(512, device=device)
    rows.append({"op": "rmsnorm(8,128,512)",
                 "ms_per_iter": round(_time_op(lambda: h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * w), 3)})
    # 微基准3: Conv3d patch(视觉侧)
    import torch.nn as nn
    conv = nn.Conv3d(3, 64, kernel_size=(2, 14, 14), stride=(2, 14, 14)).to(device)
    x = torch.randn(2, 3, 2, 224, 224, device=device)
    rows.append({"op": "conv3d_patch(B=2,224x224)",
                 "ms_per_iter": round(_time_op(lambda: conv(x)), 3)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True, help="轮次(1=基线轮)")
    ap.add_argument("--tag", required=True, help="本轮优化项标签, 如 baseline / triton_gdn_chunk")
    ap.add_argument("--out", default="out/bench")
    ap.add_argument("--note", default="", help="备注: 关联的 log 编号/优化说明")
    args = ap.parse_args()

    kind, backend = probe_backend()
    device = torch.device("npu:0" if kind == "npu" else "cpu")
    if kind == "cpu":
        print("[轨B] 无 NPU: 算子级微基准仅作本地相对参考, 端到端性能数据必须来自实卡 MindSpeed-MM 运行, 如实标注 scope。")

    rec = {
        "round": args.round, "tag": args.tag,
        "date": datetime.now().isoformat(timespec="seconds"),
        "backend": backend, "device": str(device),
        "scope": "operator-micro(v0); end-to-end 由 MindSpeed-MM 侧执行, 结果回填到 04_性能测试报告 登记表",
        "note": args.note,
        "micro_rows": run_micro(device),
    }
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"round_{args.round}_{args.tag}.json"
    fp.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in rec["micro_rows"]:
        print(f"[{r['op']}] {r['ms_per_iter']} ms/iter")
    print(f"产物: {fp}")
    print("提醒: 本轮对应的全量终端输出请 tee 进 05_原始日志/logs/, 并把日志编号写进 --note 或 04 登记表。")


if __name__ == "__main__":
    main()
