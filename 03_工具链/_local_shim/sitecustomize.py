"""本机沙箱垫片（**不属于任何交付物**，仅用于在本机跑闸门）—— 见坑 214。

问题
----
在本机这个 DSH Windows 沙箱里，用 mode `0o700` 建出来的目录**后续无法写入任何文件**：

    os.mkdir(p, 0o700)  -> 建目录成功，但往里写文件必 PermissionError
    os.mkdir(p)         -> 正常      （0o755 / 0o777 也正常）
    PowerShell New-Item -> 正常

而 `tempfile.mkdtemp()` 内部**恰好**就是 `_os.mkdir(file, 0o700)`
⇒ 所有用 `tempfile` 的自检脚本在本机都会失败。

后果（坑 214 的现场）
--------------------
`prepush_check.py` 里 6 个闸门（G2/G3/G9/G24/G5/G6）的自检用到 `tempfile`
⇒ 本机输出 **PASS=19 FAIL=6 ⇒ RELEASE_BLOCKED**，看起来像"交付物回归了"，
实际上**交付物一个字节都没坏**（zip 122 条目逐文件哈希一致）。

用法（把本目录放进 PYTHONPATH，等价于给每个子进程打补丁）
--------------------------------------------------------
    $env:PYTHONPATH = "<工具_重建>\_local_shim"
    python prepush_check.py

去掉 `PYTHONPATH` 就是"原始宿主行为"，用于**对照**（证明差异确实来自宿主）：

    Remove-Item Env:PYTHONPATH ; python prepush_check.py

⛔ 不要**为了让它变绿去改交付物**：这是宿主行为，不是交付物的缺陷。
   与坑 213 的 `sitecustomize.py` 区分：213 那个是**注入被测机制**（torch 确定性开关），
   本文件是**修宿主文件系统行为**，用途不同、不得互相套用。
"""
import os as _os
import tempfile as _tempfile

if not getattr(_tempfile, "_dsh_defaultmode_patched", False):

    def _mkdtemp(suffix=None, prefix=None, dir=None):
        suffix = "" if suffix is None else suffix
        prefix = _tempfile.gettempprefix() if prefix is None else prefix
        dir = _tempfile.gettempdir() if dir is None else dir
        names = _tempfile._get_candidate_names()
        for _ in range(_tempfile.TMP_MAX):
            name = _os.path.join(dir, prefix + next(names) + suffix)
            try:
                _os.mkdir(name)          # 默认 mode —— 本机可写（0o700 不可写，坑 214）
            except FileExistsError:
                continue
            return name
        raise FileExistsError("no usable temporary directory name found")

    _tempfile.mkdtemp = _mkdtemp
    _tempfile._dsh_defaultmode_patched = True
    _os.environ["DSH_MKDTEMP_SHIM"] = "1"
