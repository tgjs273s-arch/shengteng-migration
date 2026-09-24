# -*- coding: utf-8 -*-
r"""_rsh.py —— 把"远端命令"放进**文件**再执行，彻底消除 PowerShell 的引号/管道/美元符问题

**为什么需要它（今天第 N 次被同一个坑咬）**
Windows PowerShell 5.1 在向原生 exe 传参时**不转义参数里的双引号**：
`python pull_evidence.py --run 'echo "PKG=$P"; ls $P | head'` 会被拆成多个参数，
argparse 报 `unrecognized arguments: ... | head`。

所以凡是远端命令里含 `"`、`|`、`$`、`<`、`&` 的，都**不要再走命令行**：
写进一个 .sh 文件（本工具读文件、base64 后走既有 `RUN <base64>` 协议），
远端执行的是**文件内容的原文**，中间没有任何一层会重写它。

用法：
  python _rsh.py --file cmd.sh              # 执行文件里的命令（只读/写远端，本地不改）
  python _rsh.py --file cmd.sh --timeout 110
  python _rsh.py --file cmd.sh --out 本地文件   # 把 stdout 原文落盘（便于逐字比对）

判据（与 pull_evidence 同规矩）：远端 rc==0 且 stdout 非空才打印 RSH_OK；
否则打印 RSH_FAIL 并**原样**给出 stderr 尾部（不吞错）。
"""
import argparse
import base64
import io
import json
import os
import socket
import sys

HOST, PORT = "127.0.0.1", 8792


def call(line, timeout):
    s = socket.create_connection((HOST, PORT), timeout=timeout)
    s.settimeout(timeout)
    try:
        s.sendall((line + "\n").encode("ascii"))
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.decode("utf-8", "replace").strip() or "{}")
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="含远端命令的本地文件（原样上传，不做任何改写）")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--out", default="")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    cmd = io.open(a.file, encoding="utf-8").read()
    # ★ 2026-09-22 实测踩到：Windows PowerShell 5.1 的 `Set-Content -Encoding UTF8` **会写 BOM**，
    #   而远端 bash 看到的是 `\357\273\277set` ⇒ 报 `command not found: $'\357\273\277set'`，
    #   **脚本第一行静默失效**（其余行照跑，所以最容易骗过"看起来跑过了"）。
    #   这里统一剥掉 BOM 并**显式打印**，免得下次又靠肉眼去发现。
    if cmd.startswith("\ufeff"):
        cmd = cmd.lstrip("\ufeff")
        print("RSH_NOTE 已剥除命令文件开头的 BOM（PowerShell 5.1 Set-Content -Encoding UTF8 的产物）")
    if not cmd.strip():
        print("RSH_FAIL 命令文件为空")
        return 2
    b64 = base64.b64encode(cmd.encode("utf-8")).decode("ascii")
    r = call("RUN " + b64, a.timeout)
    out, err, rc = r.get("stdout") or "", r.get("stderr") or "", r.get("rc")
    if not a.quiet:
        if out:
            print(out.rstrip())
        if err.strip():
            print("---- stderr ----\n" + err.rstrip())
    if a.out:
        io.open(a.out, "w", encoding="utf-8").write(out)
        print("已落盘 %s（%d 字符）" % (a.out, len(out)))
    if rc == 0 and out.strip():
        # ★ 实跑当场暴露：`rc` 是**远端 shell 的**返回码 ⇒ 脚本内部失败但最后一条命令成功时仍是 0。
        #   所以命令文件里必须自己 `set -e`（或显式 exit 非 0），否则 RSH_OK 只代表"传输成功"。
        print("RSH_OK rc=0 stdout=%d 字符（= 远端 shell 返回 0；**命令内部失败需脚本自己 set -e/exit 非 0**）"
              % len(out))
        return 0
    print("RSH_FAIL rc=%s stderr_tail=%r" % (rc, err.strip()[-200:]))
    return 1


if __name__ == "__main__":
    sys.exit(main())
