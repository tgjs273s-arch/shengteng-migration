#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_batchfp.py — 逐步 batch 只读指纹代理（诊断专用，**不进交付主路径**）

=== 它为什么存在 ===
回答一个此前**从未被直接观测**过的问题：「两次运行，每一步喂进模型的是不是同一批数据」。
已有证据链里，"deter=false 下两次运行从第几步开始分岔"只能从 loss/grad_norm 的**打印值**去猜，
而打印只有 4~6 位有效数字 —— 分岔点落在打印精度之下就看不出来（坑 170 同族：判据的分辨率低于现象的尺度）。
本代理把"数据侧是否相同"从"推断"变成**逐字可比**的 sha256 行。

=== 它**不能**证明什么（写使用报告时必须一起写）===
  1. 只覆盖"进入模型**之前**"的 batch 张量。模型内部算子、归约顺序、并行/通信**都不在覆盖内** ——
     所以 BATCHFP 逐行相同**不能**推出"两次运行结果应逐位相同"。
  2. 不能替代 deter=true 正对照（那条证明的是"确定性路径下可逐位相同"，与本指纹互补，不是同一件事）。
  3. 不能证明 RNG 状态相同：本代理**刻意不碰** RNG（只读），因此它看不见种子 / 随机增强带来的差异。
     也正因如此，**用"改 seed"当负向对照是无效的**（见 _batchfp_selftest.py 用例 b 的备注）。
  4. 不能证明 collator / DataLoader worker 的内部行为：只看 worker 交出来的成品。
  5. 对**无法摘要**的对象本文件**如实标注**（unstable_repr:… / unhashable:… / undigestible:… /
     cycle:… / depth_limit:…），此时该字段**没有**可信指纹 —— 标注**不等于**"相同"，不许当通过。

=== ★ 纪律（写进 protocol，先冻结后执行）===
  · 记录本身（逐字段遍历 + sha256）会改变时序，而"最外层多套一个对象"也改变了数据流的对象身份；
    **正式计时 / 性能收益的那次运行必须全关**。默认就是关的：不设 BATCHFP 即零代理、零输出、零开销。
  · 用它跑出来的任何运行，`plan_consistent` 必须声明为 **false**（由 51_train_fp.py 打印该声明）。

=== 本文件不含任何"需要查证才能写"的第三方接口 ===
它只依赖三样东西：① Python 迭代协议（__iter__ / __next__ / __len__ / __getattr__ 转发）；
② torch / numpy 的**可选**导入；③ 本文件自己的函数。
它**不** import、也**不**猜 mindspeed_mm 的任何模块、键名或返回值形状
（框架侧的注入点由 51_train_fp.py 按设计文档 §3bis 用 `super().get_dataloader()` 拿到，
 本文件只管"拿到一个可迭代 loader 之后怎么只读地看它"）。故本文件**没有** `# 待查证：` 条目。

=== 日志格式（固定、可 grep；**每微批**一组行，只打摘要不打张量内容正文）===
    BATCHFP_HEADER rank=? tag=train enabled=1 max_micro_batches=3 ... note=noncomparable_diagnostic_run
    BATCHFP        rank=? tag=train yield=1 micro_hint=1 step_hint=1 step_hint_src=derived_from_yield_index_gas1
                   step_hint_framework=false field=input_ids dtype=numpy.int64 shape=(4,1024) nbytes=32768 sha256=<64hex>
    ...（每个字段一行，字段名**排序后**遍历，保证同一批数据每次顺序一致）
    BATCHFP_MICRO  rank=? tag=train yield=1 micro_hint=1 step_hint=1 step_hint_framework=false
                   nfields=6 batch_dtype=dict batch_shape=(n=6) sha256=<64hex>
    BATCHFP_TOTAL  rank=? tag=train pass=1 yielded=100 logged=3 skipped=97
  · `step_hint` / `micro_hint` 是**从 yield_index 推导**的值，**不是框架提供的步号** ——
    所以每行都硬带 `step_hint_framework=false`，并给出推导依据 `step_hint_src=…`。
  · `yield_index` 是"**本次迭代内**第几次产出"，每次 `__iter__` 重置（与 DataLoader 每 epoch 重新迭代
    的语义一致）。字段行刻意**不带** pass 计数，这样"同一批数据过两遍"的行可以**逐字 diff**；
    pass 计数只出现在 BATCHFP_TOTAL 行。
  · 每微批既有"每字段一行"（用来定位**是哪个字段**变了）又有"一行 BATCHFP_MICRO"（用来一行 diff 判定）。

=== 开关 ===
    BATCHFP=1            启用（默认关闭；关闭时 wrap() 原样返回 loader：零代理、零输出、零开销）
    BATCHFP_MAX=N        只记前 N 个微批，默认 3
    BATCHFP_LOG=<path>   额外把行追加到文件，便于两次运行 diff：
                         路径含 {rank} 则替换；否则 rank!=0 时追加 .rank<N> 后缀
                         （多进程写同一文件会互相撕裂，所以刻意分文件）
"""

import hashlib
import os
import re
import sys

_DEFAULT_MAX_MICRO_BATCHES = 3
_TRUTHY = ("1", "true", "yes", "on", "y", "t")
# ★ 只认"默认 repr 带内存地址"的经典形态 `<... at 0x7f...>`。
#   为什么不用宽泛的 `0x[0-9a-f]{4,}`：那会把**内容里恰好出现的** 16 进制串也误判成不稳定。
_ADDR_IN_REPR = re.compile(r"<[^<>]*\bat 0x[0-9a-fA-F]+>")
_MAX_DEPTH = 64

# ---- 可选依赖：必须**延迟 + 容错**导入 ------------------------------------------------
# 本机（Windows 自检机）**没有 torch**：若在模块顶层 `import torch`，本文件连 import 都过不去，
# 自检就无从运行。而在训练机上它又必须能拿到真的 Tensor 类型来做 isinstance。故用懒加载 + 缓存。
_np = None
_np_tried = False
_torch = None
_torch_tried = False

_SINK = None                       # 日志出口；None = 打印到 stdout（自检会临时替换它）
_GRAD_ACC_STEPS = None             # 梯度累积步数：只用于**推导** step_hint，不参与数据通路
_GRAD_ACC_SRC = "assumed_1"


def _get_numpy():
    global _np, _np_tried
    if not _np_tried:
        _np_tried = True
        try:
            import numpy as _m
            _np = _m
        except Exception:
            _np = None
    return _np


def _get_torch():
    global _torch, _torch_tried
    if not _torch_tried:
        _torch_tried = True
        try:
            import torch as _m
            _torch = _m
        except Exception:
            _torch = None
    return _torch


def _reset_torch_cache():
    """仅供自检使用：清空 torch 懒加载缓存，以便注入 / 移除鸭子替身。"""
    global _torch, _torch_tried
    _torch = None
    _torch_tried = False


def set_grad_accum_steps(steps, src="caller"):
    """设置用于**推导** step_hint 的梯度累积步数。
    为什么不自己从框架里读：那是猜接口。调用方（51_train_fp.py）只从**设计文档已查证**的
    键名 `args.training.gradient_accumulation_steps`（文档 §3 表）或配置 yaml 的
    `training.gradient_accumulation_steps`（50_train.py 读的同一个键）取值，并在 src 里交代来源。
    取不到就保持 None → step_hint 退化为 "假定 gas=1"，且 src 会如实写明。"""
    global _GRAD_ACC_STEPS, _GRAD_ACC_SRC
    try:
        n = int(steps)
        _GRAD_ACC_STEPS = n if n >= 1 else None
    except Exception:
        _GRAD_ACC_STEPS = None
    _GRAD_ACC_SRC = str(src)


def get_grad_accum_steps():
    return _GRAD_ACC_STEPS, _GRAD_ACC_SRC


def default_max_micro_batches():
    """默认只记多少个微批（=3）。做成公开函数，免得调用方去戳私有常量。"""
    return _DEFAULT_MAX_MICRO_BATCHES


# ---------------------------------------------------------------- 日志出口
def _default_sink(line):
    print(line, flush=True)


def set_sink(fn):
    """替换日志出口（自检用它捕获行）。fn=None 表示回到 stdout。返回旧的 sink。"""
    global _SINK
    old = _SINK
    _SINK = fn
    return old


def get_sink():
    return _SINK if _SINK is not None else _default_sink


# ---------------------------------------------------------------- 开关解析
def _resolve_enabled(enabled):
    """参数优先，其次环境变量 BATCHFP / BATCHFP_ENABLE；**缺省关闭**。
    为什么缺省必须能全关：记录本身会改变时序，"正式计时"与"取证"必须是两次不同的运行。"""
    if enabled is not None:
        return bool(enabled)
    v = os.environ.get("BATCHFP")
    if v is None:
        v = os.environ.get("BATCHFP_ENABLE", "")
    return str(v).strip().lower() in _TRUTHY


def _resolve_max(max_micro_batches):
    """前 N 个微批。返回 (n, 来源标注) —— 来源要进 HEADER，避免"以为设了其实没生效"。"""
    if max_micro_batches is not None:
        try:
            return max(0, int(max_micro_batches)), "param"
        except Exception:
            return _DEFAULT_MAX_MICRO_BATCHES, "param_invalid_fallback_default"
    raw = os.environ.get("BATCHFP_MAX")
    if raw is None or str(raw).strip() == "":
        return _DEFAULT_MAX_MICRO_BATCHES, "default"
    try:
        return max(0, int(str(raw).strip())), "env"
    except Exception:
        # ★ 坏值不静默吞掉：退回默认，但把"来源"标注成 fallback，HEADER 里看得见。
        return _DEFAULT_MAX_MICRO_BATCHES, "env_invalid_fallback_default"


def _resolve_gas(grad_accum_steps):
    if grad_accum_steps is not None:
        try:
            n = int(grad_accum_steps)
            if n >= 1:
                return n, "param"
        except Exception:
            pass
        return None, "param_invalid_assumed_1"
    n, src = get_grad_accum_steps()
    if n:
        return n, src
    return None, "assumed_1"


# ---------------------------------------------------------------- 摘要
def _type_name(obj):
    try:
        return type(obj).__name__
    except Exception:
        return "?"


def _shape_str(obj):
    """shape 里**不留空格**（(4,1024) 而不是 (4, 1024)）——否则按空白分词 grep/awk 会被切开。"""
    try:
        sh = getattr(obj, "shape", None)
        if sh is None:
            return "()"
        return str(tuple(sh)).replace(" ", "")
    except Exception:
        return "?"


def _hash_bytes(prefix, raw):
    h = hashlib.sha256()
    h.update(prefix.encode("utf-8", "replace"))
    h.update(raw)
    return h.hexdigest()


def _hash_text(prefix, text):
    h = hashlib.sha256()
    h.update(prefix.encode("utf-8", "replace"))
    h.update(text.encode("utf-8", "replace"))
    return h.hexdigest()


def _ndarray_bytes(arr):
    """numpy 数组 → 连续内存的原始字节。`ascontiguousarray` 是**刻意**的：
    我们要比较的是**值**，不是内存布局（同一份数据的不同 stride 不应被算成不同）。"""
    np = _get_numpy()
    if np is not None and isinstance(arr, np.ndarray):
        return np.ascontiguousarray(arr).tobytes()
    tb = getattr(arr, "tobytes", None)
    if callable(tb):
        return tb()
    raise TypeError("no tobytes")


def _is_container(obj):
    if isinstance(obj, (dict, list, tuple, set, frozenset)):
        return True
    return hasattr(obj, "keys") and hasattr(obj, "__getitem__")


def _looks_like_torch_tensor(obj):
    """torch **不可导入**时的兜底识别：模块名以 torch 开头且能自述字节。
    只在 import 失败时启用，避免与真张量语义分叉。"""
    mod = getattr(type(obj), "__module__", "") or ""
    return mod == "torch" or mod.startswith("torch.")


def _torch_digest(t, torch):
    prefix = "torch|%s|%s|" % (getattr(t, "dtype", "?"), _shape_str(t))
    try:
        det = getattr(t, "detach", None)
        cpu = det() if callable(det) else t
        c = getattr(cpu, "cpu", None)
        if callable(c):
            cpu = cpu.cpu()
        contig = getattr(cpu, "contiguous", None)
        if callable(contig):
            cpu = cpu.contiguous()
    except Exception as e:
        return "undigestible:torch.Tensor:%s" % type(e).__name__
    try:
        return _hash_bytes(prefix, _ndarray_bytes(cpu.numpy()))
    except Exception:
        pass
    # bfloat16 / 复数等 numpy 不支持的 dtype：退到 uint8 视图取原始字节。
    u8 = getattr(torch, "uint8", None) if torch is not None else None
    if u8 is None:
        # ★ 没有 torch 就拿不到 uint8 这个 dtype 对象 → **如实标注**，绝不猜一个等价物。
        return "undigestible:torch_like:no_torch_for_uint8_view"
    try:
        view = cpu.view(u8)
        return _hash_bytes(prefix, _ndarray_bytes(view.numpy()))
    except Exception as e:
        return "undigestible:torch.Tensor:%s" % type(e).__name__


def _numpy_digest(arr, seen, depth):
    prefix = "numpy|%s|%s|" % (getattr(arr, "dtype", "?"), _shape_str(arr))
    try:
        dt = getattr(arr, "dtype", None)
        if dt == object or getattr(dt, "hasobject", False):
            # ★ object 数组的 tobytes() 给的是**指针值**：跨进程/跨运行必然不同，
            #   拿它当指纹会产生"同一份数据却指纹不同"的**假警报**（比漏报更坏）。
            #   这里如实降级为逐元素结构摘要（可能很慢，但不会说谎）。
            return _sequence_digest(list(arr.ravel()), "numpy_object_array", seen, depth)
        return _hash_bytes(prefix, _ndarray_bytes(arr))
    except Exception as e:
        return "undigestible:numpy.ndarray:%s" % type(e).__name__


def _sorted_items(d):
    """字典按 key 排序遍历。key 类型混杂时不能直接 sorted（会 TypeError）→
    退化为按 (类型名, repr) 排序，仍然是**确定性**顺序。"""
    items = None
    try:
        items = list(d.items())
    except Exception:
        try:
            items = [(k, d[k]) for k in d.keys()]
        except Exception:
            return None
    try:
        items.sort(key=lambda kv: kv[0])
    except Exception:
        items.sort(key=lambda kv: (_type_name(kv[0]), repr(kv[0])))
    return items


def _sequence_digest(seq, tag, seen, depth):
    """列表/元组：**下标必须参与摘要**，否则 [a,b] 与 [b,a] 会得到同一个指纹（顺序就被看不见了）。"""
    parts = []
    for i, v in enumerate(seq):
        parts.append("%d:%s" % (i, digest(v, seen, depth + 1)))
    return _hash_text("%s|len=%d|" % (tag, len(parts)), "\n".join(parts))


def digest(obj, _seen=None, _depth=0):
    """把任意对象摘要成一个**可逐字比较**的字符串。

    返回值只有两种形态，都是刻意设计成"看得见"的：
      · 64 位小写 hex —— 正常的 sha256 指纹。对张量/数组是**逐字节**摘要
        （detach → cpu → contiguous → raw bytes），所以"改一个元素"必然改变它；
        绝不做"只取 shape""只取 sum"这类会碰撞的聚合。
      · "<标记>:<类型名>" —— **如实标注无法摘要 / 跨进程不稳定**，绝不静默跳过、绝不假装算出了指纹：
          unhashable:<type>                      repr 本身失败
          unstable_repr:<type>                   repr 里含内存地址 ⇒ 跨进程不可复现
          undigestible:<类型>:<异常名>            取字节失败
          cycle:<type> / depth_limit:<type>      循环引用 / 嵌套过深
    """
    if _depth > _MAX_DEPTH:
        return "depth_limit:%s" % _type_name(obj)
    if _seen is None:
        _seen = set()
    is_cont = _is_container(obj)
    if is_cont:
        oid = id(obj)
        if oid in _seen:
            return "cycle:%s" % _type_name(obj)
        # 用**副本**扩展：兄弟节点各自独立判断，不会互相误判成环。
        _seen = _seen | {oid}

    torch = _get_torch()
    if torch is not None:
        tcls = getattr(torch, "Tensor", None)
        if tcls is not None and isinstance(obj, tcls):
            return _torch_digest(obj, torch)
    elif _looks_like_torch_tensor(obj):
        return _torch_digest(obj, None)

    np = _get_numpy()
    if np is not None and isinstance(obj, np.ndarray):
        return _numpy_digest(obj, _seen, _depth)

    if isinstance(obj, dict) or (hasattr(obj, "items") and hasattr(obj, "keys")):
        items = _sorted_items(obj)
        if items is None:
            return "undigestible:mapping:%s" % _type_name(obj)
        parts = []
        for k, v in items:
            parts.append("%s=%s" % (k, digest(v, _seen, _depth + 1)))
        return _hash_text("mapping|n=%d|" % len(parts), "\n".join(parts))
    if isinstance(obj, (list, tuple)):
        return _sequence_digest(obj, "tuple" if isinstance(obj, tuple) else "list", _seen, _depth)
    if isinstance(obj, (set, frozenset)):
        # 集合无序 → 按下标排序子摘要，保证确定性。
        subs = sorted(digest(v, _seen, _depth + 1) for v in obj)
        return _hash_text("set|n=%d|" % len(subs), "\n".join(subs))
    if isinstance(obj, (bytes, bytearray)):
        return _hash_bytes("%s|len=%d|" % (_type_name(obj), len(obj)), bytes(obj))
    if isinstance(obj, str):
        return _hash_text("str|len=%d|" % len(obj), obj)
    if obj is None or isinstance(obj, (bool, int, float, complex)):
        return _hash_text("%s|" % _type_name(obj), repr(obj))

    try:
        r = repr(obj)
    except Exception:
        return "unhashable:%s" % _type_name(obj)
    if _ADDR_IN_REPR.search(r):
        # 默认 repr 形如 `<Foo object at 0x7f...>`：里面含**内存地址**，跨进程必然不同。
        # 直接哈希它 = 制造"两次运行同一份数据却指纹不同"的假警报，所以如实标注为不稳定。
        return "unstable_repr:%s" % _type_name(obj)
    return _hash_text("repr|%s|" % _type_name(obj), r)


def describe(obj):
    """返回 (dtype, shape, nbytes) 三个**元信息**字符串；只描述元信息，不含任何内容正文。"""
    torch = _get_torch()
    if torch is not None:
        tcls = getattr(torch, "Tensor", None)
        if tcls is not None and isinstance(obj, tcls):
            nb = "?"
            try:
                nb = str(int(obj.element_size()) * int(obj.nelement()))
            except Exception:
                nb = "?"
            return (str(getattr(obj, "dtype", "torch.?")), _shape_str(obj), nb)
    np = _get_numpy()
    if np is not None and isinstance(obj, np.ndarray):
        nb = "?"
        try:
            nb = str(int(obj.nbytes))
        except Exception:
            nb = "?"
        return ("numpy.%s" % getattr(obj, "dtype", "?"), _shape_str(obj), nb)
    if hasattr(obj, "dtype") and hasattr(obj, "shape"):
        # 鸭子张量（既不是真 torch 也不是 numpy）：如实标出 dtype/shape，不假装是 torch。
        # nbytes 只在**对象自己能报出来**时才算（element_size×nelement / nbytes），
        # 拿不到就写 "?" —— 不猜元素大小（猜错会让"字节数"这个判据变成假信息）。
        nb = "?"
        try:
            es = getattr(obj, "element_size", None)
            ne = getattr(obj, "nelement", None)
            if callable(es) and callable(ne):
                nb = str(int(es()) * int(ne()))
            elif hasattr(obj, "nbytes"):
                nb = str(int(obj.nbytes))
        except Exception:
            nb = "?"
        return (str(getattr(obj, "dtype")), _shape_str(obj), nb)
    if isinstance(obj, dict) or (hasattr(obj, "items") and hasattr(obj, "keys")):
        n = "?"
        try:
            n = str(len(obj))
        except Exception:
            n = "?"
        return ("dict", "(n=%s)" % n, "")
    if isinstance(obj, (list, tuple)):
        return (_type_name(obj), "(len=%d)" % len(obj), "")
    if isinstance(obj, (set, frozenset)):
        return (_type_name(obj), "(n=%d)" % len(obj), "")
    if isinstance(obj, str):
        return ("str", "(len=%d)" % len(obj), str(len(obj.encode("utf-8", "replace"))))
    if isinstance(obj, (bytes, bytearray)):
        return (_type_name(obj), "(len=%d)" % len(obj), str(len(obj)))
    if obj is None or isinstance(obj, (bool, int, float, complex)):
        return (_type_name(obj), "()", "")
    return (_type_name(obj), "()", "")


def iter_fields(batch):
    """把 batch 拆成 (字段名, 值) 列表，**按下标/键名排序**，保证同一批数据每次顺序一致。

    · Mapping（dict 及 keys/items 鸭子类型）→ 按 key 排序（混杂类型退化为 (类型名, repr) 排序）
    · list / tuple → 位置名 `[0]` `[1]` …（位置本身就是信息，改名会掩盖顺序变化）
    · 其它 → 单字段 `<batch>`，**不猜**它有什么属性（不编造 input_ids/labels 之类的键名）
    """
    if isinstance(batch, dict) or (hasattr(batch, "items") and hasattr(batch, "keys")):
        items = _sorted_items(batch)
        if items is not None:
            return [(str(k), v) for k, v in items]
        return [("<mapping>", batch)]
    if isinstance(batch, (list, tuple)):
        return [("[%d]" % i, v) for i, v in enumerate(batch)]
    return [("<batch>", batch)]


# ---------------------------------------------------------------- 日志文件（可选）
def _open_log(path, rank=None):
    """打开逐行追加的指纹文件。返回 (fh, err)。

    ★ 多 rank 写**同一个**文件会互相撕裂（行交错），所以刻意分文件：
      · 路径里含 `{rank}` → 替换；
      · 否则 rank==0（或 RANK 缺失）写原路径，其它 rank 写 `<path>.rank<N>`。
    """
    if not path:
        return None, None
    if rank is None:
        rank = os.environ.get("RANK", "?")
    target = str(path)
    if "{rank}" in target:
        target = target.replace("{rank}", str(rank))
    elif str(rank) not in ("0", "?"):
        target = "%s.rank%s" % (target, rank)
    try:
        d = os.path.dirname(os.path.abspath(target))
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        return open(target, "a", encoding="utf-8"), None
    except Exception as e:
        return None, "%s:%s" % (type(e).__name__, e)


# ---------------------------------------------------------------- 只读代理
class _FingerprintIterator(object):
    """把底层 loader 的迭代器包一层，**唯一**的取样点就是这里的 `next(self._it)`。"""

    def __init__(self, it, proxy):
        self._it = it
        self._proxy = proxy

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self._it)          # ← 全文**唯一**一次取样：不 peek、不预读、不重排
        except StopIteration:
            self._proxy._on_pass_end()      # 正常结束：记一行 TOTAL
            raise
        self._proxy._on_batch(batch)
        return batch


class BatchFingerprintProxy(object):
    """只读指纹代理：**逐字转发**底层 loader。

    保证（全部在 _batchfp_selftest.py 里有正/负对照）：
      · 顺序不变：产出顺序与底层**逐元素相同**（同一个对象，不复制、不排序）；
      · 不额外取样：底层 `__next__` 的调用次数**恰好等于**被消费的微批数（无 peek / 无预读）；
      · 不消耗随机数：只读，绝不调用任何会推进 RNG 的东西；
      · 对训练循环**透明**：`__getattr__` 把 sampler / dataset / set_epoch 等属性转发给底层
        （否则 `for epoch: loader.sampler.set_epoch(e)` 之类会 AttributeError 直接炸训练）。
    """

    LINE_PREFIX = "BATCHFP"

    def __init__(self, loader, tag="train", enabled=None, max_micro_batches=None,
                 grad_accum_steps=None, sink=None, log_path=None):
        self._loader = loader
        self._tag = str(tag)
        self._enabled = _resolve_enabled(enabled)
        self._max, self._max_src = _resolve_max(max_micro_batches)
        self._gas, self._gas_src = _resolve_gas(grad_accum_steps)
        self._sink = sink if sink is not None else get_sink()
        self._rank = os.environ.get("RANK", "?")
        self._pass_no = 0
        self._yield_index = 0
        self._passed = 0
        self._logged = 0
        self._skipped = 0
        self._direct_iter = None
        self._fh = None
        if self._enabled:
            if log_path is None:
                log_path = os.environ.get("BATCHFP_LOG")
            self._fh, err = _open_log(log_path, self._rank)
            self._emit("%s_HEADER rank=%s tag=%s enabled=1 max_micro_batches=%d max_src=%s "
                       "grad_accum_steps=%s gas_src=%s yield_index_resets_each_iter=true "
                       "digest=sha256_bytes step_hint_is_framework_step=false "
                       "note=noncomparable_diagnostic_run"
                       % (self.LINE_PREFIX, self._rank, self._tag, self._max, self._max_src,
                          self._gas if self._gas else "none", self._gas_src))
            if err:
                # ★ 日志文件打不开**不杀训练**（它只是取证），但必须**大声**说出来，
                #   否则"我以为记下来了"= 最坏的静默失效。
                self._emit("%s_WARN log_open_failed path=%s err=%s (仅 stdout 记录，训练继续)"
                           % (self.LINE_PREFIX, log_path, err))

    # ---- 透明转发 ------------------------------------------------------------
    def __getattr__(self, name):
        # ★ 只转发"我们自己没定义"的属性。训练循环可能访问 loader.sampler / loader.dataset /
        #   loader.set_epoch(...)，代理若不转发就会 AttributeError —— 那是**引入故障**，不是取证。
        if name.startswith("__") and name.endswith("__"):
            # 避免 pickle/拷贝/特殊方法查找时触发无限递归
            raise AttributeError(name)
        loader = self.__dict__.get("_loader", None)
        if loader is None:
            raise AttributeError(name)
        return getattr(loader, name)

    def __len__(self):
        # ★ 底层有 __len__ 就转发；没有就**照样抛 TypeError** —— 绝不在代理上伪造长度，
        #   伪造会让框架的 steps_per_epoch / 调度计算基于一个假数字。
        return len(self._loader)

    def __iter__(self):
        self._pass_no += 1
        self._yield_index = 0        # 每次 __iter__ 重置：yield_index 是"本次迭代内第几次产出"
        it = iter(self._loader)
        if self._enabled:
            self._emit("%s_PASS rank=%s tag=%s pass=%d note=yield_index_reset"
                       % (self.LINE_PREFIX, self._rank, self._tag, self._pass_no))
        return _FingerprintIterator(it, self)

    def __next__(self):
        # 支持直接 `next(proxy)`（迭代器协议）。`for batch in proxy` 每次拿的是**新的**迭代器
        # （与 DataLoader 一致）；这里的 `_direct_iter` 只为 next() 单步用法保留状态。
        it = self.__dict__.get("_direct_iter", None)
        if it is None:
            it = self.__iter__()
            self._direct_iter = it
        return it.__next__()

    # ---- 记账 ---------------------------------------------------------------
    def _step_hint(self, idx):
        """从 yield_index **推导**步号（并如实标注推导依据）。
        为什么是推导：框架打印的 iteration 号本代理看不到，也不许猜；
        gas 已知时按 (idx-1)//gas 换算，未知时假定 gas=1 并写进 step_hint_src。"""
        if self._gas and self._gas > 1:
            return ((idx - 1) // self._gas + 1, (idx - 1) % self._gas + 1,
                    "derived_from_yield_index_gas%d" % self._gas)
        return (idx, 1, "derived_from_yield_index_gas1")

    def _emit(self, line):
        try:
            self._sink(line)
        except Exception:
            pass
        if self._fh is not None:
            try:
                self._fh.write(line + "\n")
                self._fh.flush()
            except Exception:
                pass

    def _on_batch(self, batch):
        self._yield_index += 1
        self._passed += 1
        idx = self._yield_index
        if not self._enabled:
            return
        if idx > self._max:
            self._skipped += 1
            return
        self._logged += 1
        step, micro, src = self._step_hint(idx)
        hint = ("yield=%d micro_hint=%d step_hint=%d step_hint_src=%s step_hint_framework=false"
                % (idx, micro, step, src))
        fields = iter_fields(batch)          # ★ 只拆一次：拆两次等于把 batch 遍历两遍（无谓开销）
        for name, value in fields:
            dt, shape, nb = describe(value)
            self._emit("%s rank=%s tag=%s %s field=%s dtype=%s shape=%s nbytes=%s sha256=%s"
                       % (self.LINE_PREFIX, self._rank, self._tag, hint, name, dt, shape, nb,
                          digest(value)))
        bdt, bshape, _ = describe(batch)
        # `keys=` 给出**排序后**的字段名集合（设计文档 §4 的格式里就有这一项）：
        # 一行就能看出"这一步的字段集合有没有变"（字段增删/改名属于数据流变化，必须看得见）。
        self._emit("%s_MICRO rank=%s tag=%s %s nfields=%d keys=%s batch_dtype=%s batch_shape=%s sha256=%s"
                   % (self.LINE_PREFIX, self._rank, self._tag, hint, len(fields),
                      ",".join(n for n, _v in fields), bdt, bshape, digest(batch)))

    def _on_pass_end(self):
        if not self._enabled:
            return
        self._emit("%s_TOTAL rank=%s tag=%s pass=%d yielded=%d logged=%d skipped=%d"
                   % (self.LINE_PREFIX, self._rank, self._tag, self._pass_no,
                      self._passed, self._logged, self._skipped))

    def close(self):
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None


# ---------------------------------------------------------------- 入口 API
def wrap(loader, tag="train", enabled=None, max_micro_batches=None,
         grad_accum_steps=None, sink=None, log_path=None):
    """按需包装 loader。

    · **关闭时原样返回 loader 本身**（零代理、零输出、零开销）—— 正式计时的那次运行靠这条。
    · loader 为 None 时返回 None（框架在 val_interval=0 / 无验证集时可能给 None，
      给它套代理会立刻炸；这里明确放行，并在语义上保持"没变"）。
    """
    if loader is None:
        return None
    if not _resolve_enabled(enabled):
        return loader
    return BatchFingerprintProxy(loader, tag=tag, enabled=True,
                                 max_micro_batches=max_micro_batches,
                                 grad_accum_steps=grad_accum_steps,
                                 sink=sink, log_path=log_path)


def wrap_dataloader_result(res, tag="train", **kw):
    """把 `Trainer.get_dataloader()` 的返回值**只在最外层**套代理。

    ★ 两种返回形状都必须正确（设计文档 §3bis）：`(train_loader, val_loader)` 二元组 **或** 单个 loader。
      · 元组/列表：只换下标 0（train），其余元素**按原对象原样返回**（val 一字不动）；
      · 单个 loader：直接包。
    序列类型保持（tuple 进 tuple 出）—— 改变返回类型本身就是改接口，框架下游可能立刻炸。
    """
    if res is None:
        return None
    if isinstance(res, (tuple, list)):
        if len(res) == 0:
            return res
        head = wrap(res[0], tag=tag, **kw)
        tail = list(res[1:])
        return tuple([head] + tail) if isinstance(res, tuple) else ([head] + tail)
    return wrap(res, tag=tag, **kw)
