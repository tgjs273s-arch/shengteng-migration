#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_batchfp_selftest.py — `_batchfp.py` 的**本机**自检（无 torch / 无 torch_npu / 无 NPU）

=== 它为什么存在 ===
本项目最贵的错误不是"没测出来"，而是"测了个恒真的判据"：
  · 坑 107：诊断规则的匹配串在**正常日志**里也出现 ⇒ 每次都误报；
  · 坑 187：判据只写正对照、没有负对照 ⇒ "相同"到底意味着什么没人知道。
所以本文件对**每一条**性质都做**双向**检验：好样本必须通过，**坏样本必须被判不符**。
任何一条不符即打印 `FP_SELFTEST_FAIL` 并以非 0 退出（不许把失败说成"环境问题"）。

=== 本机限制（必须写在结论里）===
本机 **没有 torch、没有 torch_npu、没有 NPU**，所以：
  · 数据一律用 numpy 数组 / 鸭子类型假对象构造；
  · digest() 的 torch 分支只用**鸭子替身**做过**结构性**验证（用例 g）——
    它证明"代码路径通、逐字节敏感、退路如实标注"，**不**证明真 torch 上的 dtype/元素大小语义。
    真机（有 torch）需另跑一次同样的断言（把用例 g 的替身换成真张量即可）。
  · "不消耗随机数"用 `numpy.random` 的固定种子流验证（本机可复现）；NPU 侧 RNG 不在本文件覆盖内。

用法：python _batchfp_selftest.py      （退出码 0 = 全部 OK / 1 = 有[**不符**]）
"""

import contextlib
import copy
import io
import os
import sys
import tempfile

# ★ 本机控制台编码可能是 cp936：先把 stdout/stderr 设成 utf-8 + replace，
#   否则中文/符号会 UnicodeEncodeError 把自检自己搞崩（那属于测试脚手架故障，不是被测对象故障）。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import _batchfp  # noqa: E402

_RESULTS = []


def check(name, ok, detail):
    """记录并打印一条判据。ok=False 即 [**不符**]。"""
    _RESULTS.append((name, bool(ok), detail))
    print("  [%s] %s\n        实测: %s" % ("OK" if ok else "**不符**", name, detail))
    return bool(ok)


# ---------------------------------------------------------------- 测试脚手架（全部是鸭子类型假对象）
class _CountingLoader(object):
    """鸭子类型假 loader：只实现 __iter__/__next__/__len__。

    ★ 计数器放在**底层** loader 上：代理若偷偷多取一次（peek / 预读 / 校验形状），
      这里的 calls 就会 > 被消费的微批数 —— 这就是用例 d) 的测量点。
    """

    def __init__(self, batches):
        self._batches = list(batches)
        self.calls = 0        # 底层 __next__ 被调用的次数
        self.iters = 0        # 底层 __iter__ 被调用的次数
        self._i = 0

    def __iter__(self):
        self.iters += 1
        self._i = 0
        return self

    def __next__(self):
        if self._i >= len(self._batches):
            raise StopIteration
        b = self._batches[self._i]
        self._i += 1
        self.calls += 1
        return b

    def __len__(self):
        return len(self._batches)


class _NoLenLoader(object):
    """没有 __len__ 的 loader：用来证明代理**不伪造长度**（len(proxy) 必须照样抛 TypeError）。"""

    def __init__(self, batches):
        self._batches = list(batches)
        self._it = iter(self._batches)

    def __iter__(self):
        self._it = iter(self._batches)
        return self

    def __next__(self):
        return next(self._it)


class _RichLoader(_CountingLoader):
    """带 sampler / dataset / set_epoch 的假 loader：证明代理对训练循环**透明**。"""

    def __init__(self, batches):
        _CountingLoader.__init__(self, batches)
        self.sampler = "SAMPLER_OBJ"
        self.dataset = ["D"]
        self.set_epoch_calls = []

    def set_epoch(self, e):
        self.set_epoch_calls.append(e)


class _BadOrderProxy(object):
    """故意写错的代理：先全部取出再**逆序**产出。用来证明 c) 的判据真能发现重排。"""

    def __init__(self, loader):
        self._loader = loader

    def __iter__(self):
        return iter(list(reversed(list(self._loader))))


class _ExtraSampleProxy(object):
    """故意写错的代理：迭代开始时**多取一个**（"预读/校验形状"）。用来证明 d) 的判据能发现多取样。"""

    def __init__(self, loader):
        self._loader = loader

    def __iter__(self):
        it = iter(self._loader)
        try:
            next(it)                     # ← 多出来的那一次取样（顺序也因此错位）
        except StopIteration:
            return
        for b in it:
            yield b


class _RngHungryProxy(object):
    """故意写错的代理：每产出一个 batch 就抽一个随机数。用来证明 e) 的判据能发现 RNG 被动过。"""

    def __init__(self, loader):
        self._loader = loader

    def __iter__(self):
        for b in self._loader:
            np.random.rand()
            yield b


class _NaiveProxy(object):
    """故意写错的代理：只包了 __iter__，**不转发**其它属性。用来证明 i) 的判据能发现"不透明"。"""

    def __init__(self, loader):
        self._loader = loader

    def __iter__(self):
        return iter(self._loader)


class _Cap(object):
    """捕获 _batchfp 的日志行。

    capture_sink=True  → 替换 sink（结构化捕获，不污染 stdout，用于内容比较）；
    capture_sink=False → **不**替换 sink，走真实 stdout 路径（用于用例 f) 的"零输出"检查：
                         若代理真的打了行，它必然出现在 stdout 里，跑不掉）。
    """

    def __init__(self, capture_sink=True):
        self._capture_sink = capture_sink

    def __enter__(self):
        self.lines = []
        self.out = io.StringIO()
        prev = _batchfp.set_sink(self.lines.append if self._capture_sink else None)
        self._prev = prev
        self._redir = contextlib.redirect_stdout(self.out)
        self._redir.__enter__()
        return self

    def __exit__(self, *exc):
        self._redir.__exit__(*exc)
        _batchfp.set_sink(self._prev)
        return False


# ---------------------------------------------------------------- 假数据
def _make_batches(n=3):
    """一份"多模态训练 batch 形状"的假数据：嵌套 dict + 混合类型 + 三种张量语义字段
    （token / label / 像素），全部用 numpy（本机没有 torch）。"""
    out = []
    for k in range(n):
        out.append({
            "input_ids": np.arange(4 * 1024, dtype=np.int64).reshape(4, 1024) + 100 * k,
            "labels": np.arange(4 * 1024, dtype=np.int64).reshape(4, 1024),
            "attention_mask": np.ones((4, 1024), dtype=np.int64),
            "pixel_values": (np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8) / 255.0),
            "meta": {"sample_id": "s%d" % k, "path": "/root/mock/%d.jpg" % k},
        })
    return out


def _tamper(batches, field):
    """深拷贝后**只改一个元素**（一个 token / 一个 label / 一个像素）。

    ★ 刻意改**数据本身**，而不是改 seed / cutoff_len：
      后两者**不保证**实际喂进去的输入真的变了（坑 187 的教训）——
      负向对照必须真的动到"判据所测量的那个量"。
    """
    out = copy.deepcopy(batches)
    b = out[0]
    if field == "input_ids":
        b["input_ids"][0, 0] += 1                 # 一个 token
    elif field == "labels":
        b["labels"][0, 0] += 1                    # 一个 label
    elif field == "pixel_values":
        b["pixel_values"][0, 0, 0] += 1.0 / 255.0  # 一个像素
    else:
        raise ValueError(field)
    return out


def _lines_for(batches, enabled=True, **kw):
    """把一批假数据过一次代理，返回它产生的所有日志行。

    ★ 传进去的是 **deepcopy**：两次调用拿到的是**等值但不同对象**的数据 ⇒
      行逐字相同就证明指纹是**按值**比较，而不是靠对象身份。
    """
    with _Cap() as cap:
        ld = _CountingLoader([copy.deepcopy(b) for b in batches])
        for _ in _batchfp.wrap(ld, enabled=enabled, **kw):
            pass
    return cap.lines


def _diff(l0, l1):
    """逐行比对；长度不同返回 None（长度不同本身也是"不符"）。"""
    if len(l0) != len(l1):
        return None
    return [(a, b) for a, b in zip(l0, l1) if a != b]


# ---------------------------------------------------------------- 用例 a
def case_a():
    print("\n== a) 正常：同一份假数据过两遍，两次摘要行逐字相同 ==")
    base = _make_batches()
    l1 = _lines_for(base, tag="a")
    l2 = _lines_for(base, tag="a")          # 第二次是 deepcopy 出来的等值数据
    check("a) 两遍摘要行逐字相同（且行数 > 0，否则判据是空的）",
          len(l1) > 0 and l1 == l2,
          "行数=%d 完全相同=%s" % (len(l1), l1 == l2))
    if l1:
        print("        样例行: %s" % l1[0][:150])
    # 坏样本对照：数据变了，判据必须拒绝"相同"
    l3 = _lines_for(_tamper(base, "input_ids"), tag="a")
    check("a-坏样本对照）改一个元素后仍判成'相同' → 必须为否",
          _diff(l1, l3) != [] and _diff(l1, l3) is not None,
          "改一个元素后有 %s 处不同" % (len(_diff(l1, l3)) if _diff(l1, l3) is not None else "长度都不同"))


# ---------------------------------------------------------------- 用例 b
def case_b():
    print("\n== b) 负向（关键）：改一个 token / label / 像素 → 摘要必须变化 ==")
    base = _make_batches()
    l0 = _lines_for(base, tag="b")
    for field in ("input_ids", "labels", "pixel_values"):
        l1 = _lines_for(_tamper(base, field), tag="b")
        d = _diff(l0, l1)
        if d is None:
            check("b) 改一个 %s → 摘要必须变" % field, False, "行数都变了：%d vs %d" % (len(l0), len(l1)))
            continue
        field_hits = [x for x in d if " field=" in x[0]]
        only_that = (len(field_hits) == 1 and (" field=%s " % field) in field_hits[0][0])
        micro_hit = any("BATCHFP_MICRO" in y for _x, y in d)
        # 期望恰好 2 行变化：该字段行 + 微批汇总行（改一个元素**只**影响这两行 ⇒ 定位能力）
        check("b) 改一个 %s（1 个元素）→ 摘要必须变，且**只**变该字段行+微批行" % field,
              len(d) == 2 and only_that and micro_hit,
              "变化行数=%d 其中字段行=%d(命中 %s) 微批行=%s"
              % (len(d), len(field_hits), only_that, micro_hit))
        if d:
            print("        字段行 before: %s" % d[0][0][:140])
            print("        字段行 after : %s" % d[0][1][:140])
    # 正对照：不改数据必须 0 处不同（否则"变化"恒真，上面三条就没有意义）
    check("b-正对照）不改数据 → 0 处不同（证明上面的'变化'不是恒真）",
          _diff(l0, _lines_for(base, tag="b")) == [],
          "差异行数=%d" % len(_diff(l0, _lines_for(base, tag="b")) or []))
    # 备注：为什么不能用"改 seed / 改 cutoff_len"当负向对照
    np.random.seed(20260922)
    s1 = _lines_for(base, tag="b")
    np.random.seed(7)
    s2 = _lines_for(base, tag="b")
    check("b-备注）只改 seed 而数据没变 → 0 处不同（这正是 seed 不能当负向对照的原因）",
          _diff(s1, s2) == [],
          "差异行数=%d（数据是确定性的 arange，seed 不影响它 ⇒ 用 seed 当负向测试会'看起来通过'" % len(_diff(s1, s2) or []))


# ---------------------------------------------------------------- 用例 c
def case_c():
    print("\n== c) 顺序不变：代理产出的 batch 与底层逐元素相同 ==")
    objs = [{"i": i, "x": np.array([i], dtype=np.int64)} for i in range(5)]
    raw = list(_CountingLoader(objs))
    with _Cap():
        via = list(_batchfp.wrap(_CountingLoader(objs), tag="c", enabled=True))
    same = (len(raw) == len(via)) and all(a is b for a, b in zip(raw, via))
    check("c) 顺序逐元素相同（用 id 比对：同一批**对象**、同一顺序）",
          same, "底层=%d 代理=%d id序列相同=%s" % (len(raw), len(via), same))
    # 坏样本对照：故意逆序的代理必须被判不符
    with _Cap():
        bad = list(_BadOrderProxy(_CountingLoader(objs)))
    bad_same = (len(bad) == len(raw)) and all(a is b for a, b in zip(raw, bad))
    check("c-坏样本对照）逆序代理必须被判不符", not bad_same,
          "逆序代理的 id序列相同=%s（必须为 False）" % bad_same)
    # 补充：摘要本身对顺序敏感（否则"顺序被换掉"根本不会体现在指纹里）
    A = np.array([1, 2, 3], dtype=np.int64)
    B = np.array([9, 8, 7], dtype=np.int64)
    check("c-补充）digest 对顺序敏感：digest([A,B]) != digest([B,A])",
          _batchfp.digest([A, B]) != _batchfp.digest([B, A]),
          "digest([A,B])=%s… digest([B,A])=%s…"
          % (_batchfp.digest([A, B])[:16], _batchfp.digest([B, A])[:16]))


# ---------------------------------------------------------------- 用例 d
def case_d():
    print("\n== d) 不额外取样：底层 __next__ 调用次数 == 消费的微批数 ==")
    base = _make_batches()
    for en in (True, False):
        ld = _CountingLoader([copy.deepcopy(b) for b in base])
        n = 0
        with _Cap():
            for _ in _batchfp.wrap(ld, tag="d", enabled=en):
                n += 1
        check("d) enabled=%s：底层 __next__ 调用数 == 消费微批数" % en,
              ld.calls == n and n == len(base),
              "calls=%d consumed=%d len(loader)=%d" % (ld.calls, n, len(base)))
    # 坏样本对照：多取一次的代理必须被判不符
    ld = _CountingLoader([copy.deepcopy(b) for b in base])
    n = 0
    for _ in _ExtraSampleProxy(ld):
        n += 1
    check("d-坏样本对照）多取一次的代理必须被判不符", ld.calls != n,
          "calls=%d consumed=%d（必须不等）" % (ld.calls, n))


# ---------------------------------------------------------------- 用例 e
def case_e():
    print("\n== e) 不消耗随机数：固定 numpy 种子下，过不过代理的随机数序列相同 ==")
    base = _make_batches()
    N = 4

    def _draw(k):
        return [int(np.random.randint(0, 2 ** 31 - 1)) for _ in range(k)]

    def _protocol(proxy_factory):
        """固定种子 → 取 1 个数 → 过一遍代理 → 再取 3 个数。返回 (首个数, 后续 3 个数)。"""
        np.random.seed(1234)
        first = int(np.random.randint(0, 2 ** 31 - 1))
        with _Cap():
            for _ in proxy_factory():
                pass
        return first, _draw(3)

    np.random.seed(1234)
    control = _draw(N)

    first_ok, rest_ok = _protocol(
        lambda: _batchfp.wrap(_CountingLoader([copy.deepcopy(b) for b in base]), tag="e", enabled=True))
    check("e) 过代理（enabled=True）与对照的随机数序列**完全相同**",
          first_ok == control[0] and rest_ok == control[1:],
          "对照=%s / 过代理=(%d, %s)" % (control, first_ok, rest_ok))

    first_bad, rest_bad = _protocol(
        lambda: _RngHungryProxy(_CountingLoader([copy.deepcopy(b) for b in base])))
    check("e-坏样本对照）消耗 RNG 的代理必须被判不符", rest_bad != control[1:],
          "对照后续=%s / 坏代理后续=%s（必须不同）" % (control[1:], rest_bad))


# ---------------------------------------------------------------- 用例 f
def case_f():
    print("\n== f) 关掉开关时零输出（默认关闭）==")
    keys = ("BATCHFP", "BATCHFP_ENABLE", "BATCHFP_MAX", "BATCHFP_LOG")
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    try:
        base = _make_batches()
        # ★ 不替换 sink：若代理真的打了行，必然落在真实 stdout 上
        with _Cap(capture_sink=False) as cap:
            ld = _CountingLoader([copy.deepcopy(b) for b in base])
            p = _batchfp.wrap(ld, tag="f")          # 不传 enabled ⇒ 走环境变量（未设 ⇒ 关）
            n = sum(1 for _ in p)
        text = cap.out.getvalue()
        check("f) 默认关闭：stdout 里没有任何 BATCHFP 行，且代理不介入（wrap 原样返回 loader）",
              ("BATCHFP" not in text) and len(cap.lines) == 0 and (p is ld) and n == len(base),
              "stdout含BATCHFP=%s sink行数=%d wrap返回原对象=%s 消费=%d"
              % ("BATCHFP" in text, len(cap.lines), p is ld, n))
        # 坏样本对照：打开开关时必须**真的**有输出（否则上面的"零输出"是假通过）
        os.environ["BATCHFP"] = "1"
        with _Cap(capture_sink=False) as cap2:
            for _ in _batchfp.wrap(_CountingLoader([copy.deepcopy(b) for b in base]), tag="f"):
                pass
        check("f-坏样本对照）打开开关（BATCHFP=1）后必须输出 BATCHFP 行",
              "BATCHFP" in cap2.out.getvalue(),
              "stdout 行数=%d，含 BATCHFP=%s" % (len(cap2.out.getvalue().splitlines()),
                                                "BATCHFP" in cap2.out.getvalue()))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------- 用例 g
_FAKE_UINT8 = object()


class _FakeTorchTensor(object):
    """★ 鸭子替身（本机没有真 torch）：只为证明 digest() 的 torch 分支**结构上**跑通。

    它**不能**替代真机验证 —— 真 torch 的 dtype / 元素大小 / bfloat16 报错语义需在有 torch 的
    机器上另跑一次同样的断言。这里模拟两件事：
      ① `.detach().cpu().numpy().tobytes()`（正常 dtype 路径）；
      ② `.numpy()` 抛 TypeError（bfloat16 的真实行为）→ 退到 `.view(torch.uint8)` 取原始字节。
    """

    def __init__(self, raw, shape, dtype_name, numpy_ok=True):
        self._raw = raw                 # bytes：就是"原始字节"
        self.shape = shape
        self.dtype = dtype_name
        self._numpy_ok = numpy_ok

    @classmethod
    def from_int64(cls, arr):
        return cls(np.ascontiguousarray(arr, dtype=np.int64).tobytes(),
                   tuple(arr.shape), "torch.int64", True)

    @classmethod
    def bf16_like(cls, raw):
        # (len//2,) 是"元素个数"，raw 是它的原始字节 —— 与真 bfloat16 的存储一致
        return cls(raw, (len(raw) // 2,), "torch.bfloat16", False)

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def numel(self):
        return int(np.prod(self.shape)) if self.shape else 1

    def nelement(self):
        return self.numel()

    def element_size(self):
        return 8 if self.dtype == "torch.int64" else 2

    def view(self, dtype):
        if dtype is _FAKE_UINT8:
            return _FakeTorchTensor(self._raw, (len(self._raw),), "torch.uint8", True)
        return self

    def numpy(self):
        if not self._numpy_ok:
            # 与真 torch 的报错同族：numpy 不支持该 dtype
            raise TypeError("Got unsupported ScalarType BFloat16")
        return np.frombuffer(self._raw, dtype=np.uint8)


# ★ 让替身的模块名以 torch 开头：这样"torch **不可导入**"时也能被识别为 torch 张量（鸭子路径）。
_FakeTorchTensor.__module__ = "torch"


def _install_fake_torch():
    import types
    m = types.ModuleType("torch")
    m.Tensor = _FakeTorchTensor
    m.uint8 = _FAKE_UINT8
    sys.modules["torch"] = m
    _batchfp._reset_torch_cache()
    return m


def _uninstall_fake_torch():
    sys.modules.pop("torch", None)
    _batchfp._reset_torch_cache()


def case_g():
    print("\n== g) torch 分支（**鸭子替身**，本机无真 torch）==")
    try:
        _uninstall_fake_torch()
        t1 = _FakeTorchTensor.from_int64(np.arange(8, dtype=np.int64))
        t2 = _FakeTorchTensor.from_int64(np.arange(8, dtype=np.int64) + np.array([1] + [0] * 7))
        d1, d2 = _batchfp.digest(t1), _batchfp.digest(t2)
        check("g1) torch **不可导入**时的鸭子路径：改一个元素 → 摘要变（且不是'无法摘要'标记）",
              d1 != d2 and len(d1) == 64 and (not d1.startswith("undigestible")),
              "d1=%s… d2=%s…" % (d1[:16], d2[:16]))

        _install_fake_torch()
        d3, d4 = _batchfp.digest(t1), _batchfp.digest(t2)
        check("g2) torch 可导入时走 isinstance 路径：改一个元素 → 摘要变，且与鸭子路径**一致**",
              d3 != d4 and d3 == d1,
              "d3=%s… d4=%s… 与 g1 一致=%s" % (d3[:16], d4[:16], d3 == d1))

        raw1 = bytes(range(16))
        raw2 = bytes([255]) + bytes(range(1, 16))
        b1, b2 = _FakeTorchTensor.bf16_like(raw1), _FakeTorchTensor.bf16_like(raw2)
        e1, e2 = _batchfp.digest(b1), _batchfp.digest(b2)
        check("g3) bfloat16 式退路（.numpy() 抛错 → .view(uint8)）：逐字节敏感且不是标记",
              e1 != e2 and len(e1) == 64 and (not e1.startswith("undigestible")),
              "e1=%s… e2=%s…" % (e1[:16], e2[:16]))

        _uninstall_fake_torch()
        m1 = _batchfp.digest(b1)
        check("g4) 无 torch 时 bfloat16 退路**不可用** → 必须如实标注（不许编一个等价物）",
              m1 == "undigestible:torch_like:no_torch_for_uint8_view",
              "digest=%s" % m1)

        # 顺带验证 describe() 只给元信息（不打印内容正文）—— **两条路径都要验**
        t8 = _FakeTorchTensor.from_int64(np.arange(8, dtype=np.int64))
        _install_fake_torch()
        dt, sh, nb = _batchfp.describe(t8)
        _uninstall_fake_torch()
        dt2, sh2, nb2 = _batchfp.describe(t8)
        check("g5) describe 只给元信息 dtype/shape/nbytes（isinstance 路径与无 torch 的鸭子路径都给全）",
              dt == "torch.int64" and sh == "(8,)" and nb == "64"
              and dt2 == "torch.int64" and sh2 == "(8,)" and nb2 == "64",
              "isinstance路径: dtype=%s shape=%s nbytes=%s / 鸭子路径: dtype=%s shape=%s nbytes=%s"
              % (dt, sh, nb, dt2, sh2, nb2))
    finally:
        _uninstall_fake_torch()


# ---------------------------------------------------------------- 用例 h
def case_h():
    print("\n== h) get_dataloader 返回值两种形状：元组 (train,val) 与 单个 loader ==")
    base = _make_batches()
    tr = _CountingLoader([copy.deepcopy(b) for b in base])
    va = _CountingLoader([copy.deepcopy(b) for b in base])
    with _Cap():
        res = _batchfp.wrap_dataloader_result((tr, va), tag="h", enabled=True)
    ok_tuple = (isinstance(res, tuple) and len(res) == 2
                and isinstance(res[0], _batchfp.BatchFingerprintProxy) and (res[1] is va))
    check("h1) 元组：只包下标 0（train），val **按原对象原样**返回、类型仍是 tuple",
          ok_tuple, "类型=%s len=%s [0]=%s [1] is val=%s"
          % (type(res).__name__, len(res), type(res[0]).__name__, res[1] is va))
    with _Cap():
        single = _batchfp.wrap_dataloader_result(tr, tag="h", enabled=True)
    check("h2) 单个 loader：直接包成代理", isinstance(single, _batchfp.BatchFingerprintProxy),
          "类型=%s" % type(single).__name__)
    with _Cap():
        none_res = _batchfp.wrap_dataloader_result(None, tag="h", enabled=True)
    check("h3) None（无验证集等）：原样返回 None，不套代理（套了会立刻炸）", none_res is None,
          "返回=%r" % (none_res,))
    # 坏样本对照：改了返回类型 / 把 val 也包上的实现必须被判不符
    with _Cap():
        bad = [_batchfp.wrap(x, tag="h", enabled=True) for x in (tr, va)]
    bad_ok = (isinstance(bad, tuple) and len(bad) == 2 and (bad[1] is va))
    check("h-坏样本对照）返回 list 且把 val 也包掉的实现必须被判不符", not bad_ok,
          "坏实现判据成立=%s（必须为 False；类型=%s [1] is val=%s）"
          % (bad_ok, type(bad).__name__, bad[1] is va))
    for p in (res[0] if isinstance(res, tuple) else None, single, bad[0], bad[1]):
        if isinstance(p, _batchfp.BatchFingerprintProxy):
            p.close()


# ---------------------------------------------------------------- 用例 i
def case_i():
    print("\n== i) 迭代器协议与透明转发（__len__ / __next__ / __getattr__）==")
    base = _make_batches()
    ld = _RichLoader([copy.deepcopy(b) for b in base])
    with _Cap():
        p = _batchfp.wrap(ld, tag="i", enabled=True)
        len_ok = (len(p) == len(ld))
        b1 = next(p)                      # 迭代器协议：next(proxy) 必须可用
        b2 = next(p)
        next_ok = (b1 is base[0] or b1 is ld._batches[0]) and (b2 is ld._batches[1])
        attr_ok = (p.sampler == "SAMPLER_OBJ") and (p.dataset == ["D"])
        p.set_epoch(5)
        epoch_ok = (ld.set_epoch_calls == [5])
        p.close()
    check("i1) __len__ 转发（不伪造长度）", len_ok, "len(proxy)=%s len(loader)=%s" % (len(p), len(ld)))
    check("i2) __next__ 可用（迭代器协议），产出的是**同一个** batch 对象", next_ok,
          "next→同一对象=%s（b1 is loader[0]=%s, b2 is loader[1]=%s）"
          % (next_ok, b1 is ld._batches[0], b2 is ld._batches[1]))
    check("i3) __getattr__ 透明转发 sampler/dataset/set_epoch（否则训练循环会 AttributeError）",
          attr_ok and epoch_ok,
          "sampler=%r dataset=%r set_epoch_calls=%s" % (p.sampler, p.dataset, ld.set_epoch_calls))
    # 坏样本对照：不转发的实现必须被判不符
    naive = _NaiveProxy(ld)
    try:
        _ = naive.sampler
        naive_ok = True
    except AttributeError:
        naive_ok = False
    check("i-坏样本对照）不转发的代理必须被判不符（访问 .sampler 应 AttributeError）",
          not naive_ok, "naive.sampler 可访问=%s（必须为 False）" % naive_ok)
    # 底层没有 __len__ 时，代理也必须照样抛 TypeError（不伪造）
    with _Cap():   # 捕获 HEADER 行，避免自检自己的输出被混进"零输出"判据之外的地方
        nolen = _batchfp.wrap(_NoLenLoader([copy.deepcopy(b) for b in base]), tag="i", enabled=True)
    try:
        _ = len(nolen)
        nolen_ok = False
    except TypeError:
        nolen_ok = True
    check("i4) 底层无 __len__ → len(proxy) 照样抛 TypeError（不伪造长度）", nolen_ok,
          "抛 TypeError=%s" % nolen_ok)
    nolen.close()


# ---------------------------------------------------------------- 用例 j
def case_j():
    print("\n== j) BATCHFP_LOG 落盘 + 多 rank 分文件（避免多进程写同一文件互相撕裂）==")
    saved = {k: os.environ.get(k) for k in ("BATCHFP_LOG", "RANK", "BATCHFP", "BATCHFP_MAX")}
    tmp = tempfile.mkdtemp(prefix="batchfp_selftest_")
    try:
        base = _make_batches()
        path0 = os.path.join(tmp, "fp.log")
        os.environ["BATCHFP_LOG"] = path0
        os.environ["BATCHFP"] = "1"
        os.environ["RANK"] = "0"
        with _Cap():
            p0 = _batchfp.wrap(_CountingLoader([copy.deepcopy(b) for b in base]), tag="j")
            for _ in p0:
                pass
            p0.close()
        os.environ["RANK"] = "1"
        with _Cap():
            p1 = _batchfp.wrap(_CountingLoader([copy.deepcopy(b) for b in base]), tag="j")
            for _ in p1:
                pass
            p1.close()
        t0 = open(path0, encoding="utf-8").read() if os.path.isfile(path0) else ""
        t1p = path0 + ".rank1"
        t1 = open(t1p, encoding="utf-8").read() if os.path.isfile(t1p) else ""
        check("j) rank0 写原路径、rank1 写 <path>.rank1，两边都有 BATCHFP 行且互不混杂",
              ("BATCHFP" in t0) and ("BATCHFP" in t1) and ("rank=1" not in t0) and ("rank=0" not in t1),
              "rank0 行数=%d（含 rank=1 的行=%s） rank1 文件存在=%s 行数=%d"
              % (len(t0.splitlines()), "rank=1" in t0, os.path.isfile(t1p), len(t1.splitlines())))
        # 坏样本对照：路径模板 {rank} 必须被替换（否则两个 rank 会撞同一文件）
        path2 = os.path.join(tmp, "fp_{rank}.log")
        os.environ["BATCHFP_LOG"] = path2
        with _Cap():
            p2 = _batchfp.wrap(_CountingLoader([copy.deepcopy(b) for b in base]), tag="j")
            for _ in p2:
                pass
            p2.close()
        p2file = path2.replace("{rank}", "1")
        check("j-坏样本对照）{rank} 模板若未被替换就会撞文件名 → 必须已替换",
              os.path.isfile(p2file) and ("{rank}" not in p2file),
              "存在 %s = %s" % (os.path.basename(p2file), os.path.isfile(p2file)))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------- main
def main():
    print("== _batchfp 本机自检（无 torch / 无 torch_npu / 无 NPU）==")
    print("Python=%s numpy=%s torch=%s" % (sys.version.split()[0], np.__version__,
                                          "ABSENT" if _batchfp._get_torch() is None else "present"))
    try:
        import torch as _t  # noqa: F401
        print("!! 本机竟然能 import torch：用例 g 的'鸭子替身'结论将被真 torch 路径覆盖，" )
        print("   请在有 torch 的机器上删掉/替换替身后重跑 g)，并把结论按真 torch 重述。")
    except Exception:
        pass
    case_a()
    case_b()
    case_c()
    case_d()
    case_e()
    case_f()
    case_g()
    case_h()
    case_i()
    case_j()

    failed = [n for n, ok, _d in _RESULTS if not ok]
    print("\n== 汇总 ==")
    print("判据条数 = %d，不符 = %d" % (len(_RESULTS), len(failed)))
    for n in failed:
        print("  **不符** %s" % n)
    if not failed:
        print("说明：本机结论只覆盖 numpy/鸭子对象路径；digest 的真 torch 分支需在有 torch 的机器上重跑用例 g。")
    print("FP_SELFTEST_%s" % ("OK" if not failed else "FAIL"))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
