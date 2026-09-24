#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""compare_ckpt.py —— 逐张量比对两次运行的落盘权重（不需要框架内部接口）

为什么这样做（而不是自己 hook 前向/反向）
--------------------------------------
"固定完整状态的一步回放"（复核建议的 e）需要拿到同一状态并分别比前向、梯度、更新 ——
那要求接入训练框架内部，而**猜接口是被本项目明令禁止的**（坑 163/179）。
落盘权重是**不需要任何内部接口**的客观量：它是整条"前向→反向→更新"路径的累积结果。
于是：
  · 两份**同配置**运行的权重**逐张量完全相同** ⇒ 该路径在本负载下可复现；
  · 不完全相同 ⇒ 差异的**分布形态**能区分两类原因：
      - 差异遍布所有张量、量级均匀 ⇒ 与"全局归约/累加顺序"一类的非确定性一致；
      - 差异集中在少数张量 ⇒ 指向局部问题（某个算子/某个模块的路径）。
  ★ 本工具**只报事实**（相等与否、差异量级、分布），**不做归因结论**。

判据：全部张量逐位相等 ⇒ CKPT_IDENTICAL；否则 CKPT_DIFFERS + 差异摘要。

用法：python3 compare_ckpt.py <ckptA 目录或 safetensors 文件> <ckptB ...> [--top 10]
"""
import argparse
import os
import sys


def find_weights(p):
    """给定目录或文件，找到权重文件（safetensors 优先）。"""
    if os.path.isfile(p):
        return p
    cands = []
    for root, _d, files in os.walk(p):
        for f in files:
            if f.endswith(".safetensors"):
                cands.append(os.path.join(root, f))
    if not cands:
        return None
    cands.sort(key=lambda x: -os.path.getsize(x))
    return cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    wa, wb = find_weights(args.a), find_weights(args.b)
    if not wa or not wb:
        print("CKPT_FAIL 找不到权重文件：a=%s b=%s" % (wa, wb))
        return 2
    print("CKPT_A %s (%d B)" % (wa, os.path.getsize(wa)))
    print("CKPT_B %s (%d B)" % (wb, os.path.getsize(wb)))

    try:
        from safetensors import safe_open
        import torch
    except Exception as exc:
        print("CKPT_FAIL 缺依赖：%s: %s" % (type(exc).__name__, exc))
        return 2

    with safe_open(wa, framework="pt", device="cpu") as fa, \
            safe_open(wb, framework="pt", device="cpu") as fb:
        ka, kb = set(fa.keys()), set(fb.keys())
        if ka != kb:
            print("CKPT_KEYS_DIFFER only_a=%d only_b=%d（例：%s / %s）"
                  % (len(ka - kb), len(kb - ka), sorted(ka - kb)[:3], sorted(kb - ka)[:3]))
        common = sorted(ka & kb)
        n_ident, diffs, n_nan = 0, [], 0
        for k in common:
            ta, tb = fa.get_tensor(k), fb.get_tensor(k)
            if ta.shape != tb.shape:
                diffs.append((k, float("inf"), float("inf"), "shape"))
                continue
            if torch.equal(ta, tb):
                n_ident += 1
                continue
            d = (ta.float() - tb.float()).abs()
            if not torch.isfinite(d).all():
                n_nan += 1
            mx, mn = float(d.max()), float(d.mean())
            denom = ta.float().abs().mean().item()
            rel = (mn / denom) if denom else float("inf")
            diffs.append((k, mx, rel, ""))
    print("CKPT_SUM 共同张量=%d **逐位相同=%d** 不同=%d 含非有限=%d"
          % (len(common), n_ident, len(diffs), n_nan))
    diffs.sort(key=lambda x: -x[1])
    for k, mx, rel, why in diffs[:args.top]:
        print("   %-60s max_abs=%.6g mean_rel=%s %s" % (k[:60], mx, ("%.3g" % rel) if rel == rel else "n/a", why))
    if not diffs and n_ident:
        print("CKPT_IDENTICAL 全部 %d 个张量逐位相同" % n_ident)
        return 0
    print("CKPT_DIFFERS 有 %d/%d 个张量不同（形态见上；**本工具不做归因**）" % (len(diffs), len(common)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
