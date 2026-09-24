# -*- coding: utf-8 -*-
r"""_patch_session_read.py —— 修会话守护的**大输出截断** bug

现象（2026-09-22 实测）：拉取 555 KB 的 `train.log` 时，本地哈希 ≠ 远端哈希（远端稳定不变、无进程），
而 221 KB 的 `fp.log` 正常 ⇒ 不是"文件还在写"，是**我的读取逻辑丢了尾部**。

根因：第一版手写"轮询 `recv_ready()` + `exit_status_ready()`"的收尾：
`exit_status_ready()` 可能先于"stdout 数据全部到达"为真 ⇒ 我只排空了**当时已到**的缓冲就返回。
正确做法：`stdout.read()`（paramiko 标准用法）—— 阻塞到通道 EOF，不截断。

谁拦住了这个错误：**下游的逐文件 SHA256 校验**（`pull_evidence.py`）⇒ 报"拉取失败"而不是
把截断日志落盘当证据。**这是 fail-closed 判据真正的价值所在**：错误变成显式的失败。
"""
import io
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "ssh_session.py")

OLD = '''            out, err, rc = b"", b"", None
            deadline = time.time() + timeout
            while True:
                if ch.recv_ready():
                    out += ch.recv(65536)
                    continue
                if ch.recv_stderr_ready():
                    err += ch.recv_stderr(65536)
                    continue
                if ch.exit_status_ready():
                    while ch.recv_ready():
                        out += ch.recv(65536)
                    while ch.recv_stderr_ready():
                        err += ch.recv_stderr(65536)
                    rc = ch.recv_exit_status()
                    break
                if time.time() > deadline:
                    ch.close()
                    return {"rc": None, "stdout": out.decode("utf-8", "replace"),
                            "stderr": err.decode("utf-8", "replace") + "\\n[本地超时 %ds]" % timeout,
                            "timeout": True}
                time.sleep(0.02)'''

NEW = '''            # ★★ 2026-09-22 修 bug：第一版手写"轮询 recv_ready + exit_status_ready"的收尾，
            #   在**大输出**时丢尾部（`exit_status_ready()` 可能先于"stdout 全部到达"为真，
            #   于是只排空了当时已到的缓冲就返回）。221 KB 侥幸没事，555 KB 必踩。
            #   拦住它的不是我的自检，而是**下游的逐文件 SHA256**（pull_evidence）⇒ 报拉取失败。
            #   正确做法是 paramiko 标准 `read()`：阻塞到通道 EOF，不截断。
            try:
                out = stdout.read()
                err = stderr.read()
            except Exception as exc:
                try:
                    ch.close()
                except Exception:
                    pass
                return {"rc": None, "stdout": "",
                        "stderr": "[本地读取失败 %s: %s]" % (type(exc).__name__, exc),
                        "timeout": True}
            rc = ch.recv_exit_status()'''

src = io.open(TARGET, encoding="utf-8").read()
if "out = stdout.read()" in src:
    print("PATCH_SKIP 已修过")
    sys.exit(0)
if OLD not in src:
    print("PATCH_FAIL 找不到待替换的读取块")
    sys.exit(2)
bak = TARGET + ".bak_read_%s" % time.strftime("%Y%m%d_%H%M%S")
shutil.copy2(TARGET, bak)
io.open(TARGET, "w", encoding="utf-8").write(src.replace(OLD, NEW, 1))
back = io.open(TARGET, encoding="utf-8").read()
assert "out = stdout.read()" in back and "exit_status_ready()" not in back
print("PATCH_READ_OK 已改为 stdout.read()；备份=%s" % bak)
