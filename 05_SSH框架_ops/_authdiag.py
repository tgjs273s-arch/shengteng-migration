# -*- coding: utf-8 -*-
r"""_authdiag.py —— 诊断"到 Go 网关跳板的认证方式"（不猜，逐个试并把失败原因打全）

背景：重建的会话守护用 paramiko 密码认证跳板，连续两次（含一份**新** token）都报
`AuthenticationException: Authentication failed: transport shut down or saw EOF`
⇒ 说明问题不在 token 是否过期，而在**认证方式/协议细节**。
跳板的 banner 是 `remote software version Go`（Go 写的网关），这类网关常见两种差异：
  ① 只提供 `keyboard-interactive`（不提供 `password`）—— paramiko 的 `password=` 只走 `password`；
  ② 对 auth 失败**直接关连接**（而不是回 "failure"），于是 paramiko 只看到 EOF、看不到方法列表。

本脚本对**每一种方式用一条全新 transport**（失败后连接会被关，复用没意义）：
`auth_none` → `auth_password` → `auth_interactive_dumb`，
并打印 `BadAuthenticationType.allowed_types`（这才是权威的"服务器支持什么"）。
"""
import io
import json
import os
import socket
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "_token", "current.json")


def load():
    rec = json.load(io.open(STORE, encoding="utf-8"))
    jhost, jport = rec["jump"].rsplit(":", 1)
    juser, jpass = rec["token"].split(":", 1)
    return jhost, int(jport), juser, jpass, rec


def attempt(label, fn):
    try:
        fn()
        print("  [OK]   %s ⇒ 认证成功" % label)
        return True
    except Exception as exc:
        extra = ""
        allowed = getattr(exc, "allowed_types", None)
        if allowed:
            extra = " allowed_types=%s" % (allowed,)
        print("  [FAIL] %s ⇒ %s: %s%s" % (label, type(exc).__name__,
                                          str(exc).replace("\n", " ")[:160], extra))
        return False


def main():
    import paramiko
    jhost, jport, juser, jpass, rec = load()
    print("目标 store: target=%s jump=%s:%d token=%s…(密码长度 %d)"
          % (rec["target"], jhost, jport, juser[:14], len(jpass)))
    print("paramiko=%s" % paramiko.__version__)

    for label, do in (
        ("auth_none（探方法列表）",
         lambda t: t.auth_none(juser)),
        ("auth_password",
         lambda t: t.auth_password(juser, jpass)),
        ("auth_interactive_dumb",
         lambda t: t.auth_interactive_dumb(juser, handler=lambda *a, **k: [jpass])),
    ):
        t = paramiko.Transport((jhost, jport))
        try:
            t.banner_timeout = 20
            t.auth_timeout = 20
            t.start_client(timeout=20)
            print("  banner=%r 本地 kex=%s" % (t.remote_version, t.get_security_options().kex[:3]))
        except Exception as exc:
            print("  [FAIL] %s 之前 start_client 就失败：%s: %s" % (label, type(exc).__name__, exc))
            t.close()
            continue
        attempt(label, lambda: do(t))
        t.close()

    # 直接看 TCP 层：网关是否在挥手前就断开
    try:
        s = socket.create_connection((jhost, jport), timeout=10)
        banner = s.recv(256)
        print("  raw banner=%r" % banner[:120])
        s.close()
    except Exception as exc:
        print("  raw TCP 失败：%s: %s" % (type(exc).__name__, exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
