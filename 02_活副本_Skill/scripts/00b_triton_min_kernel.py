#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00b_triton_min_kernel.py —— Triton-on-NPU 最小可用性判据（唯一权威判据）

为什么独立成文件（不是 heredoc 内联）：
  · 坑 21：triton 的 inspect.getsourcelines 在 stdin/heredoc 下会失败 → **必须落盘**
  · 坑 46：旧判据用 `triton.program_id(0)` —— 该属性在 JIT 命名空间**不存在**
    （正确写法 `tl.program_id(0)`）。结果是**假阴性**：明明可用却报"不通"，
    于是 Skill 自动降级 triton→ascendc→eager，**白扔性能最优路径**。
  · 坑 60：★ 把 `import triton.language as tl` 写在**函数内部**同样会失败 ——
    `@triton.jit` 编译 kernel 时按**模块全局命名空间**(`fn.__globals__`)解析名字，
    函数局部的 `tl` 不在其中 → `NameError('tl is not defined')` → **又一个假阴性**。
    因此：**所有被 jit 的函数与其依赖的导入，必须位于模块顶层。**

退出码：0 = TRITON_NPU_OK；1 = 不可用（打印 TRITON_NPU_FAIL + 原因）
输出：
  TRITON_NPU_OK backend=<name> triton=<ver>
  TRITON_NPU_FAIL reason=<...> backend=<...> triton=<...>

自检（不需要 NPU）：本文件被 `--selfcheck` 调用时只检查结构约束（导入在顶层）。
"""
import os
import sys

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/_triton_min_cache")

# ---- ★ 必须位于模块顶层（坑 60）：jit 编译依赖模块全局命名空间 ----
torch = None
triton = None
tl = None
_IMPORT_ERR = ""
try:
    import torch                                    # noqa: F811
    import torch_npu                                # noqa: F401
    import triton                                   # noqa: F811
    import triton.language as tl                    # noqa: F811
except Exception as _e:                             # noqa: BLE001
    _IMPORT_ERR = "%s: %s" % (type(_e).__name__, _e)


if triton is not None:
    @triton.jit
    def _probe_kernel(X, Y, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)                      # ★ tl.program_id，不是 triton.program_id
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        x = tl.load(X + offs, mask=m, other=0.0)
        tl.store(Y + offs, x * 2.0, mask=m)
else:
    _probe_kernel = None


def _backend_names():
    try:
        from triton.backends import backends
        return ",".join(sorted(backends.keys()))
    except Exception as e:                          # noqa: BLE001
        return "probe_error:%s" % type(e).__name__


def main():
    ver = getattr(triton, "__version__", "?") if triton is not None else "?"

    if _IMPORT_ERR or triton is None:
        print("TRITON_NPU_FAIL reason=import:%s triton=%s" % (_IMPORT_ERR or "triton=None", ver))
        return 1

    backend = _backend_names()
    if "ascend" not in backend:
        print("TRITON_NPU_FAIL reason=no_ascend_backend backends=%s triton=%s" % (backend, ver))
        return 1

    try:
        n = 1024
        x = torch.randn(n, dtype=torch.float32).npu()
        y = torch.empty_like(x)
        _probe_kernel[(triton.cdiv(n, 256),)](x, y, n, BLOCK=256)
        torch.npu.synchronize()
        if bool(torch.allclose(y.cpu(), x.cpu() * 2.0, atol=1e-3)):
            print("TRITON_NPU_OK backend=%s triton=%s" % (backend, ver))
            return 0
        print("TRITON_NPU_FAIL reason=wrong_result backend=%s triton=%s" % (backend, ver))
        return 1
    except Exception as e:                          # noqa: BLE001
        msg = str(e).replace("\n", " ")[:200]
        print("TRITON_NPU_FAIL reason=%s:%s backend=%s triton=%s"
              % (type(e).__name__, msg, backend, ver))
        return 1


def structure_selfcheck():
    """★ 不需要 NPU 的结构自检：守住坑 46 与坑 60 这两类"判据自身写错"。

    这是把"教训"变成"机制"：以后任何人在 kernel 里写 `triton.program_id`，
    或把 `tl` 导入/kernel 定义放进**函数内部**（而非模块顶层），这里都会失败。

    ★ 坑 62：**用 AST 而不是文本匹配**。第一版用"行首是否缩进"判断，结果把模块顶层
      `try:` 块里缩进的 import 误判为"在函数内"，自检自己产生 7 条假阳性。
      缩进 ≠ 作用域 —— 语义判断必须交给解析器（与坑 42/56 同族：不要用脆弱文本启发式代替语义）。
    """
    import ast

    path = os.path.abspath(__file__)
    src = open(path, encoding="utf-8").read()
    fails = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print("TRITON_PROBE_SELFCHECK_FAIL 无法解析自身: %s" % e)
        return 1

    def module_scope(body):
        """展开模块顶层语句（穿过 try/if/with，**不进入** 函数/类）。"""
        out = []
        for node in body:
            out.append(node)
            if isinstance(node, (ast.Try, ast.If, ast.With, ast.AsyncWith)):
                out.extend(module_scope(getattr(node, "body", []) or []))
                out.extend(module_scope(getattr(node, "orelse", []) or []))
                out.extend(module_scope(getattr(node, "finalbody", []) or []))
                for h in getattr(node, "handlers", []) or []:
                    out.extend(module_scope(h.body))
        return out

    top = module_scope(tree.body)

    # 1) 不得出现 `triton.program_id`（坑 46）—— 语义级：属性访问，不是字符串/注释
    misuse = [n for n in ast.walk(tree)
              if isinstance(n, ast.Attribute) and n.attr == "program_id"
              and isinstance(n.value, ast.Name) and n.value.id == "triton"]
    if misuse:
        fails.append("坑46：出现 triton.program_id（应为 tl.program_id），行 %s"
                     % ",".join(str(getattr(n, "lineno", "?")) for n in misuse))

    # 2) `import triton.language as tl` 必须在**模块顶层作用域**（坑 60）
    has_tl = False
    for n in top:
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name == "triton.language" and a.asname == "tl":
                    has_tl = True
    if not has_tl:
        fails.append("坑60：`import triton.language as tl` 不在模块顶层作用域"
                     "（jit 编译按模块全局命名空间解析名字）")

    # 3) @triton.jit 装饰的函数必须定义在模块顶层（坑 60）
    jit_funcs = []
    for n in top:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in n.decorator_list:
                nm = d.attr if isinstance(d, ast.Attribute) else getattr(d, "id", "")
                if nm == "jit":
                    jit_funcs.append(n.name)
    if not jit_funcs:
        fails.append("坑60：没有模块顶层的 @triton.jit 函数（kernel 不能定义在函数内部）")

    # 4) 顶层必须有 triton 导入（否则无 NPU 时也要能给出清晰错误）
    has_triton_imp = any(isinstance(n, ast.Import) and
                         any(a.name == "triton" for a in n.names) for n in top)
    if not has_triton_imp:
        fails.append("顶层未导入 triton")

    if fails:
        for f in fails:
            print("TRITON_PROBE_SELFCHECK_FAIL %s" % f)
        return 1
    print("TRITON_PROBE_SELFCHECK_OK 结构约束满足"
          "（AST 校验：tl 在模块顶层 / jit 在模块顶层 [%s] / 无 triton.program_id）"
          % ",".join(jit_funcs))
    return 0


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        sys.exit(structure_selfcheck())
    sys.exit(main())
