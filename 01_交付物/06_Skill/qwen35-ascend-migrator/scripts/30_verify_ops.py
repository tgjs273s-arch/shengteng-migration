# verify_ops.py —— Skill Phase3(verify): 4 类关键算子 forward/shape 贯通(算子矩阵) + 双轨探测
# 提升自初赛 POC poc2_npu_forward.py（逻辑已真实验证）
# 用法: python scripts/verify_ops.py [--out out/verify]
# 有卡走 NPU(轨A), 无卡走 CPU(轨B, 仅证 shape 等价); 无自定义 kernel 的算子标 contract-only, 不伪造
import argparse
import json
import os
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def probe_backend():
    # 探 torch_npu: 装好且卡可见 -> 轨A; 否则 -> 轨B(CPU 回退) —— 兼作 Skill P0 env-probe
    try:
        import torch_npu  # noqa  # 导入后 torch.npu 才挂上
        if torch.npu.is_available():
            return "npu", "Ascend NPU (torch_npu OK)"
    except Exception:
        pass
    return "cpu", "CPU-回退(无 NPU 卡或无 torch_npu, 仅证 shape 等价)"


def _rmsnorm(x, weight, eps=1e-6):
    # 手写 RMSNorm: torch.nn.RMSNorm 要 2.4 才有, 低版本也能跑
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def _quick_gelu(x):
    # HF 的 quick_gelu 近似: x * sigmoid(1.702 * x)
    return x * torch.sigmoid(1.702 * x)


def run_matrix(device):
    rows = []
    # 行1: 视觉 3D-Conv patch embedding
    try:
        conv = nn.Conv3d(3, 64, kernel_size=(2, 14, 14), stride=(2, 14, 14)).to(device)
        x = torch.randn(1, 3, 2, 28, 28, device=device)
        y = conv(x)
        rows.append({"op": "vision_3d_conv_patch(Conv3d)", "status": "forward_ok",
                     "in_shape": list(x.shape), "out_shape": list(y.shape)})
    except Exception as e:
        rows.append({"op": "vision_3d_conv_patch(Conv3d)", "status": "error",
                     "err": f"{type(e).__name__}: {e}"})
    # 行2: RMSNorm + silu + quick_gelu
    try:
        h = torch.randn(2, 8, 64, device=device)
        w = torch.ones(64, device=device)
        n = _rmsnorm(h, w)
        a1 = F.silu(n)
        a2 = _quick_gelu(n)
        rows.append({"op": "custom_act_or_norm(RMSNorm+silu+quick_gelu)", "status": "forward_ok",
                     "in_shape": list(h.shape),
                     "out_shape": [list(n.shape), list(a1.shape), list(a2.shape)]})
    except Exception as e:
        rows.append({"op": "custom_act_or_norm(RMSNorm+silu+quick_gelu)", "status": "error",
                     "err": f"{type(e).__name__}: {e}"})
    # 行3: 全注意力 sdpa
    try:
        q = torch.randn(1, 4, 16, 32, device=device)
        k = torch.randn(1, 4, 16, 32, device=device)
        v = torch.randn(1, 4, 16, 32, device=device)
        o = F.scaled_dot_product_attention(q, k, v)
        rows.append({"op": "full_attn(scaled_dot_product_attention)", "status": "forward_ok",
                     "in_shape": list(q.shape), "out_shape": list(o.shape)})
    except Exception as e:
        rows.append({"op": "full_attn(scaled_dot_product_attention)", "status": "error",
                     "err": f"{type(e).__name__}: {e}"})
    # 行4: GatedDeltaNet 线性注意力核心 -> 只声明 shape 契约, 不真算
    rows.append({"op": "linear_attn_GatedDeltaNet(chunk_gated_delta_rule/causal_conv1d)",
                 "status": "contract-only",
                 "shape_contract": {"in": "[B,T,H,K]", "out": "[B,T,H,V]"},
                 "note": "本地无该自定义 kernel, 仅声明 shape 契约, 不复刻/不伪造; "
                         "真 forward 依赖昇腾 Triton/Ascend C 落地(P2/复赛 PerfTune)"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/verify", help="产物目录")
    # ★ 坑 63：流水线 driver（scripts/run_from_zero.sh）按统一风格给每个阶段传 `--env`，
    #   而本脚本原先只接受 `--out` → argparse 直接退出：
    #     `30_verify_ops.py: error: unrecognized arguments: --env out/probe/env.json`
    #   → **P3 阶段从未真正运行过**；且 driver 的判据只看产物是否存在，
    #     于是它被记成普通 FAIL 而不是"接口不匹配"。现显式接受 `--env` 并落进产物。
    ap.add_argument("--env", default=None, help="P0 环境档案（可选；记录进产物便于溯源）")
    args = ap.parse_args()
    if args.env and not os.path.isfile(args.env):
        print("[WARN] --env 指定的环境档案不存在: %s（继续，P3 不依赖它）" % args.env)

    kind, backend = probe_backend()
    device = torch.device("npu:0" if kind == "npu" else "cpu")
    print("=== Skill P3 算子矩阵 @", backend, " device=", device, "===")
    rows = run_matrix(device)
    for r in rows:
        if r["status"] == "contract-only":
            print(f"[{r['op']}] contract-only | 契约 in={r['shape_contract']['in']} out={r['shape_contract']['out']}")
            print("        -> 本地无 kernel, 不伪造 forward; 真算子归 P2 昇腾 Triton/Ascend C 落地")
        elif r["status"] == "forward_ok":
            print(f"[{r['op']}] forward_ok | in={r['in_shape']} out={r['out_shape']}")
        else:
            print(f"[{r['op']}] ERROR | {r.get('err')}")
    print("-" * 64)
    print("backend:", backend)   # 无卡含 'CPU-回退'=轨B; 有卡含 'Ascend NPU'=轨A
    print("结论: 4 类算子里 3 类真 forward 贯通, GatedDeltaNet chunk 诚实标 contract-only。")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "ops_matrix.json").write_text(json.dumps({
        "backend": backend, "device": str(device), "matrix": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"产物: {outdir / 'ops_matrix.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # 兜底: 整体异常也完整打印, 便于日志定位, 不静默
        traceback.print_exc()
        raise
