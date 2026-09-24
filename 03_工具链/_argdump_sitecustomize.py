"""捕获 `prepare_wy_repr_bwd` 的**真实入参**(shape/dtype/stride)—— 只读观测，不改计算。

用法：把它所在目录放到 PYTHONPATH **最前**，并设 `ARGDUMP_OUT=<目录>`；
解释器启动时会 import 目标模块并**替换其函数属性**，此后任何
`from ...wy_fast import prepare_wy_repr_bwd` 取到的都是打过补丁的版本。

★ 为什么用"启动时 import + 改属性"而不是事后 patch：
  若调用方写的是 `from x import f`，事后改 `x.f` **不会**影响已绑定的名字。
  在**任何**调用方 import 之前就把属性换掉，才能保证拦得住（这是本文件存在的唯一理由）。
★ 只记录前 `ARGDUMP_MAX`(默认 2) 次调用的**参数元信息**，不拷贝张量数据、不改返回值。
"""
import json
import os

_OUT = os.environ.get("ARGDUMP_OUT", "/tmp/argdump")
_MAX = int(os.environ.get("ARGDUMP_MAX", "2"))
_TARGETS = [("mindspeed_mm.fsdp.ops.gdn.triton.wy_fast", "prepare_wy_repr_bwd")]


def _desc(v):
    d = {"py_type": type(v).__name__}
    try:
        import torch
        if isinstance(v, torch.Tensor):
            d.update({"shape": list(v.shape), "dtype": str(v.dtype),
                      "stride": list(v.stride()), "device": str(v.device),
                      "is_contiguous": bool(v.is_contiguous()),
                      "requires_grad": bool(v.requires_grad)})
        elif isinstance(v, (list, tuple)):
            d["len"] = len(v)
            d["elems"] = [_desc(x) for x in list(v)[:4]]
        elif isinstance(v, (int, float, str, bool)) or v is None:
            d["value"] = v
    except Exception as exc:            # noqa: BLE001
        d["desc_error"] = "%s: %s" % (type(exc).__name__, exc)
    return d


def _install():
    try:
        os.makedirs(_OUT, exist_ok=True)
    except OSError:
        pass
    import importlib
    for mod, fn in _TARGETS:
        try:
            m = importlib.import_module(mod)
        except Exception as exc:                        # noqa: BLE001
            print("ARGDUMP_IMPORT_FAIL %s: %s: %s" % (mod, type(exc).__name__, exc), flush=True)
            continue
        orig = getattr(m, fn, None)
        if orig is None:
            print("ARGDUMP_NO_FUNC %s.%s" % (mod, fn), flush=True)
            continue
        state = {"n": 0}

        def _make(orig_fn, name, st):
            def _wrapper(*a, **k):
                if st["n"] < _MAX:
                    rec = {"fn": name, "call_index": st["n"],
                           "args": [_desc(x) for x in a],
                           "kwargs": {kk: _desc(vv) for kk, vv in k.items()}}
                    p = os.path.join(_OUT, "%s_call%d.json" % (name, st["n"]))
                    try:
                        with open(p, "w", encoding="utf-8") as fh:
                            json.dump(rec, fh, ensure_ascii=False, indent=1)
                        print("ARGDUMP_WROTE %s" % p, flush=True)
                    except OSError as exc:
                        print("ARGDUMP_WRITE_FAIL %s" % exc, flush=True)
                    st["n"] += 1
                return orig_fn(*a, **k)
            return _wrapper

        setattr(m, fn, _make(orig, fn, state))
        print("ARGDUMP_PATCHED %s.%s (out=%s max=%d)" % (mod, fn, _OUT, _MAX), flush=True)


_install()
