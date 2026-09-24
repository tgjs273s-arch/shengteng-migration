#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_ops.py —— 兼容 shim

★ 坑 64：流水线脚本用**数字前缀**命名（`30_verify_ops.py`）以便人读阶段顺序，
  但 Python **无法 `import 30_verify_ops`**（标识符不能以数字开头），
  于是 `60_bench.py` 里的 `from verify_ops import probe_backend` 必然
  `ModuleNotFoundError` —— 结果是 **P6 阶段从未成功运行过**。

  修法（保持编号命名的可读性，同时让导入可用）：
    · 本文件按**路径**加载 `30_verify_ops.py` 并把它的公开名字转发到本模块命名空间
    · 于是 `from verify_ops import probe_backend` 与 `import verify_ops` 均可用

  为什么不用「重命名脚本去掉数字」：流水线各脚本、文档、SKILL.md 与交付材料中
  大量按 `NN_名称.py` 引用，改动面大且会破坏"阶段顺序即文件名顺序"的可读性。
  用显式 shim 把「为什么需要这个文件」写在代码里，比隐式约定更可靠。
"""
import importlib.util
import os

_IMPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "30_verify_ops.py")

if not os.path.isfile(_IMPL):
    raise ImportError("verify_ops shim: 找不到实现文件 %s" % _IMPL)

_spec = importlib.util.spec_from_file_location("_verify_ops_impl", _IMPL)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# 只转发公开名字（不转发 __name__/__file__ 等，避免误导调用方）
for _k, _v in vars(_mod).items():
    if not _k.startswith("__"):
        globals()[_k] = _v

# 明确暴露被 60_bench.py 使用的接口，缺失则提前报错而不是运行到一半才炸
if "probe_backend" not in globals():
    raise ImportError("verify_ops shim: 30_verify_ops.py 未提供 probe_backend"
                      "（60_bench.py 依赖它）")

__all__ = [k for k in vars(_mod) if not k.startswith("__")]
